"""GraspGen ZMQ-client grasp pose sampler.

Talks to a running GraspGen ZMQ server (``NVlabs/GraspGen``,
diffusion-based 6-DoF grasp generation, Franka panda checkpoint) and
returns ``(N, 4, 4)`` grasp poses in mesh-local frame.

Why ZMQ instead of importing the model in-process: GraspGen's pip
extras pin ``torch==2.1.0+cu121`` and a custom ``pointnet2_ops`` C
extension that conflict with OmniGibson's ``torch==2.6.0+cu124``. The
server runs in its own ``uv`` venv with the right stack; we send NumPy
point clouds over a local TCP socket and get back ``(N, 4, 4)`` grasp
poses + per-grasp confidence.

Server start (one-time, in a separate terminal)::

    cd /path/to/GraspGen && source .venv/bin/activate
    python client-server/graspgen_server.py \\
        --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_franka_panda.yml \\
        --port 5556

Convention: GraspGen returns grasp poses in the same frame as the input
point cloud. We feed the visual mesh's surface points as-is (mesh-local
frame), so the returned poses are also in mesh-local frame and can be
composed with the object's world pose exactly like the antipodal sampler.

GraspGen's 6-DoF pose convention follows the Franka panda gripper:
+Z = approach (toward fingertips), +Y = closing axis, +X = perpendicular.
That matches our existing eef-pose conventions, so no axis swap needed.

Pose origin: GraspGen's grasp pose origin is at the panda gripper's
``hand`` link (i.e. the wrist body, not at the fingertips). Our IK
loop expects ``eef_link``, which sits ~0.10 m forward of ``panda_hand``
along approach. We shift the returned origins forward along +Z by
``hand_to_eef_offset`` (default 0.1034 m, matching NV's gripper depth)
so the poses can be plugged directly into the existing render pipeline.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


_DEFAULT_HOST = os.environ.get("GRASPGEN_HOST", "localhost")
_DEFAULT_PORT = int(os.environ.get("GRASPGEN_PORT", "5556"))
_CLIENT_CACHE: dict = {}


def _get_client(host: str, port: int):
    """Cache one ZMQ client per (host, port) — opening the socket per
    inference call would force a TCP handshake every time, costing
    ~10 ms+ on localhost and dominating the model latency."""
    key = (host, port)
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]
    # Lazy import — pyzmq + msgpack-numpy aren't always installed in the
    # OG env; fail loudly with a setup hint if they're missing.
    try:
        import zmq  # noqa: F401
        import msgpack  # noqa: F401
        import msgpack_numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "GraspGen client needs pyzmq + msgpack-numpy. "
            "In the behavior conda env: pip install pyzmq msgpack msgpack-numpy"
        ) from exc

    # Build a minimal client (mirrors GraspGen's GraspGenClient API but
    # without depending on the upstream package, which would drag in
    # pointnet2_ops etc.).
    client = _MinimalGraspGenClient(host=host, port=port)
    _CLIENT_CACHE[key] = client
    return client


class _MinimalGraspGenClient:
    """In-process wrapper around the GraspGen ZMQ wire protocol.

    Minimal so the OG env doesn't have to import the full ``grasp_gen``
    package (which pulls in pointnet2_ops + torch 2.1). We replicate
    the request shape used by ``client-server/graspgen_client.py``.
    """

    def __init__(self, host: str = "localhost", port: int = 5556,
                 timeout_ms: int = 30000):
        import zmq
        import msgpack_numpy as m
        m.patch()

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._addr = f"tcp://{host}:{port}"
        self.server_metadata = self._request({"action": "metadata"})

    def _request(self, payload: dict) -> dict:
        import msgpack
        self._sock.send(msgpack.packb(payload, use_bin_type=True))
        raw = self._sock.recv()
        return msgpack.unpackb(raw, raw=False)

    def infer(self, point_cloud: np.ndarray, num_grasps: int = 200,
              topk_num_grasps: int = -1, grasp_threshold: float = -1.0,
              min_grasps: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        pcd = np.ascontiguousarray(point_cloud, dtype=np.float32)
        resp = self._request({
            "action": "infer",
            "point_cloud": pcd,
            "num_grasps": int(num_grasps),
            "topk_num_grasps": int(topk_num_grasps),
            "grasp_threshold": float(grasp_threshold),
            "min_grasps": int(min_grasps),
        })
        if "error" in resp:
            raise RuntimeError(f"GraspGen server returned error: {resp['error']}")
        # Copy because msgpack-numpy returns read-only views and downstream
        # callers (sample_graspgen_grasps) edit the pose translations in
        # place when applying the hand→eef offset.
        grasps = np.array(resp["grasps"], dtype=np.float32, copy=True)
        scores = np.array(resp["confidences"], dtype=np.float32, copy=True)
        return grasps, scores

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:  # noqa: BLE001
            pass


def sample_graspgen_grasps(
    mesh,
    n_points: int = 8000,
    num_grasps: int = 200,
    topk_num_grasps: int = 100,
    confidence_threshold: float = 0.0,
    hand_to_eef_offset: float = 0.1034,
    host: str | None = None,
    port: int | None = None,
    rng: np.random.Generator | None = None,
):
    """Predict grasp poses on ``mesh`` via the GraspGen ZMQ server.

    Args:
        mesh: ``trimesh.Trimesh`` (visual mesh in object-local frame).
        n_points: surface points to sample for the input cloud.
        num_grasps: how many diffusion samples the server should draw.
        topk_num_grasps: server returns the top-K by discriminator score.
        confidence_threshold: drop scores below this client-side
            (server's top-K already enforces a soft threshold).
        hand_to_eef_offset: how far to shift the GraspGen pose origin
            forward along +Z so it lands on Franka's ``eef_link``
            instead of ``panda_hand``. Default 0.1034 m (NV's
            ``gripper_depth``). Set to 0 if the upstream pose origin
            should be used as-is.
        host / port: override ``GRASPGEN_HOST`` / ``GRASPGEN_PORT``.
        rng: optional numpy ``Generator`` for reproducible point sampling.

    Returns:
        Tuple ``(poses, scores)`` of dtype float32. ``poses`` is shape
        ``(N, 4, 4)``, ranked by descending discriminator score.
    """
    import trimesh

    if rng is None:
        rng = np.random.default_rng()

    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts = np.asarray(pts, dtype=np.float32)

    client = _get_client(host or _DEFAULT_HOST, port or _DEFAULT_PORT)
    grasps, scores = client.infer(
        pts, num_grasps=num_grasps, topk_num_grasps=topk_num_grasps,
    )
    if confidence_threshold > 0.0:
        keep = scores >= confidence_threshold
        grasps = grasps[keep]
        scores = scores[keep]
    if len(grasps) == 0:
        return np.empty((0, 4, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    if hand_to_eef_offset != 0.0:
        # +Z column = approach axis; shift origin forward along approach.
        approach = grasps[:, :3, 2]
        grasps[:, :3, 3] += approach * hand_to_eef_offset

    order = np.argsort(-scores)
    return grasps[order].astype(np.float32), scores[order].astype(np.float32)
