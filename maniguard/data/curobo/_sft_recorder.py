"""SFT recorder shared between pick_and_place_from_dataset and render_pnp_for_sft.

Provides:
  - ``install_wrist_camera_patch`` — monkey-patch FrankaPanda._load_sensors so
    a Camera prim is injected under ``panda_hand`` before sensor discovery.
    The pose is copied verbatim from ``franka_mounted.usda`` so the captured
    wrist view matches the one the task-generation pipeline already uses.
  - ``SFTRecorder`` — captures (image_left, image_right, wrist, state, action)
    per env.step into an HDF5 plus three review MP4s.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_WRIST_CAM_PATCHED = False


def install_wrist_camera_patch() -> None:
    """Relocate the panda_hand wrist Camera before sensor discovery.

    The FrankaPanda USD ships with a wrist Camera at translate=(0.05, 0,
    -0.05) (behind panda_hand origin, toward the wrist joint). With the
    maniguard longfinger patch the fingers extend further in +Z and the
    stock camera ends up framing the back of the gripper. We flip the
    Z to +0.05 so the camera sits between the wrist and the finger
    tips, looking out at the grasping zone. Idempotent.
    """
    global _WRIST_CAM_PATCHED
    if _WRIST_CAM_PATCHED:
        return
    from omnigibson.robots.franka import FrankaPanda
    import omnigibson.lazy as lazy

    _orig = FrankaPanda._load_sensors
    TARGET_TRANSLATE = (0.05, 0.0, 0.05)
    TARGET_ORIENT = (-0.0923, -0.701, -0.701, -0.0923)  # (w, x, y, z)

    def _patched(self):
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        hand = self._links.get("panda_hand") if self._links else None
        if hand is None:
            return _orig(self)

        cam_path = f"{hand.prim_path}/Camera"
        prim = stage.GetPrimAtPath(cam_path)
        if not prim.IsValid():
            # USD didn't ship with a wrist Camera (e.g. FrankaMounted
            # before chassis update): create a fresh one.
            cam_prim = lazy.pxr.UsdGeom.Camera.Define(stage, cam_path)
            prim = cam_prim.GetPrim()
            cam_prim.CreateFocalLengthAttr().Set(17.0)
            cam_prim.CreateClippingRangeAttr().Set(
                lazy.pxr.Gf.Vec2f(0.001, 1000000.0))
            print(f"[sft_recorder] created wrist Camera at {cam_path}",
                  flush=True)
        else:
            print(f"[sft_recorder] using existing wrist Camera at {cam_path}",
                  flush=True)

        # Rewrite translate + orient on every load — the USD's stock
        # pose puts the camera behind the wrist with longfinger off, and
        # we want a single canonical pose either way.
        xf = lazy.pxr.UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        t_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeTranslate,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        t_op.Set(lazy.pxr.Gf.Vec3d(*TARGET_TRANSLATE))
        r_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeOrient,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        r_op.Set(lazy.pxr.Gf.Quatd(*TARGET_ORIENT))
        s_op = xf.AddXformOp(
            lazy.pxr.UsdGeom.XformOp.TypeScale,
            lazy.pxr.UsdGeom.XformOp.PrecisionDouble, "")
        s_op.Set(lazy.pxr.Gf.Vec3d(1.0, 1.0, 1.0))
        xf.SetXformOpOrder([t_op, r_op, s_op])
        print(f"[sft_recorder] wrist Camera pose -> translate="
              f"{TARGET_TRANSLATE} orient={TARGET_ORIENT}", flush=True)
        return _orig(self)

    FrankaPanda._load_sensors = _patched
    _WRIST_CAM_PATCHED = True


def _quat_canonical_t(q_xyzw):
    """Return ±q with w >= 0 (canonical shortest-path rep) for a torch quat."""
    if float(q_xyzw[3].item()) < 0.0:
        return -q_xyzw
    return q_xyzw


def _quat2axisangle_np(q_xyzw) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    q = np.clip(q, -1.0, 1.0)
    w = float(q[3])
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-6:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arccos(max(-1.0, min(1.0, w))))
    axis = q[:3] / sin_half
    return (axis * angle).astype(np.float32)


def find_wrist_sensor(robot):
    from omnigibson.sensors import VisionSensor
    if not getattr(robot, "sensors", None):
        return None
    for name, sensor in robot.sensors.items():
        if isinstance(sensor, VisionSensor) and "hand" in name.lower():
            return sensor
    for sensor in robot.sensors.values():
        if isinstance(sensor, VisionSensor):
            return sensor
    return None


def _sensor_rgb_uint8(sensor, resolution: int) -> np.ndarray:
    try:
        obs_tuple = sensor.get_obs()
        rgb = obs_tuple[0].get("rgb") if isinstance(obs_tuple, tuple) else obs_tuple.get("rgb")
    except Exception:  # noqa: BLE001
        rgb = None
    if rgb is None:
        return np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rgb = rgb[..., :3]
    if hasattr(rgb, "detach"):
        rgb = rgb.detach()
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    return np.asarray(rgb, dtype=np.uint8)


def grab_state(robot) -> np.ndarray:
    """8D SFT state: [eef_pos(3) | eef_aa(3) | gripper_q(2)] float32."""
    eef_pos = robot.get_relative_eef_position().detach().cpu().numpy().astype(np.float32)
    eef_quat = robot.get_relative_eef_orientation().detach().cpu().numpy().astype(np.float32)
    eef_aa = _quat2axisangle_np(eef_quat)
    gripper_idx = robot.gripper_control_idx[robot.default_arm]
    if hasattr(gripper_idx, "cpu"):
        gripper_idx_np = gripper_idx.cpu().numpy()
    else:
        gripper_idx_np = np.asarray(gripper_idx)
    gripper_q = (robot.get_joint_positions().detach().cpu().numpy()[gripper_idx_np]
                 .astype(np.float32))
    return np.concatenate([eef_pos, eef_aa, gripper_q]).astype(np.float32)


class SFTRecorder:
    """Capture (obs, action) per env.step + stream three review MP4s.

    Usage::

        recorder = SFTRecorder(out_dir, resolution=256, fps=30)
        recorder.attach(env, robot)
        # ... in the OSC step loop after env.step(action):
        recorder.record_step(action_np, done=False)
        # at the end:
        recorder.finalize(success=True, attrs={"task_dir": ..., "seed": ...})
    """

    def __init__(self, out_dir, resolution: int = 256, fps: int = 30,
                 ext_cam_names=("cam_left", "cam_right"),
                 record_sim_states: bool = True,
                 lerobot_writer=None,
                 lerobot_prompt: str | None = None,
                 ltl_monitor=None):
        self.out_dir = Path(out_dir)
        self.resolution = int(resolution)
        self.fps = int(fps)
        self._ext_cam_names = list(ext_cam_names)
        # If True, ``og.sim.dump_state(serialized=True)`` is snapshotted per
        # step and persisted under ``data/demo_0/states`` — useful for any
        # consumer that needs to replay the rollout in OG. Set False to
        # suppress (saves ~10% disk on SFT-only runs).
        self.record_sim_states = bool(record_sim_states)
        # Optional maniguard.data.lerobot.lerobot_writer.LeRobotEpisodeWriter — when
        # set, MP4s are streamed directly to the writer's target paths and
        # per-step (state, action) is buffered for commit on success. In
        # this mode the HDF5 no longer stores image arrays (LeRobot owns the
        # pixels) but still carries sim states.
        self._lerobot_writer = lerobot_writer
        self._lerobot_prompt = lerobot_prompt
        # Optional maniguard.utils.safety_monitor.TaskLTLMonitor — when set,
        # ``record_step`` advances it once per recorded step and ``finalize``
        # surfaces the summary into HDF5 attrs + LeRobot episodes.jsonl. The
        # monitor itself is owned by the caller (one per task, reset per
        # variant via ``attach``).
        self._ltl_monitor = ltl_monitor
        # Phase A cache mode — when True, ``record_step`` also tees its
        # per-step data into self._phase_a_cache so a later variant can
        # replay it via ``append_cached_step`` instead of re-running the
        # physics. Set via ``start_phase_a_cache``/``end_phase_a_cache``.
        self._phase_a_cache_mode = False
        self._phase_a_cache: list = []
        self.env = None
        self.robot = None
        self._left = self._right = self._wrist = None
        self._writers = None
        self._frames_l: list[np.ndarray] = []
        self._frames_r: list[np.ndarray] = []
        self._frames_w: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._sim_states: list[np.ndarray] = []
        self._dones: list[bool] = []
        self.n_video_frames = 0
        self.n_record_steps = 0

        # Previous eef pose, for FK-derived action recording during Phase A
        # (where the underlying sim step is a kinematic teleport, not an OSC
        # command). First call after reset_eef_history() records action = 0.
        self._prev_eef_pos = None
        self._prev_eef_quat = None

    def reset_eef_history(self) -> None:
        """Forget the previous eef pose. First call to ``record_fk_step``
        after this records action = zeros."""
        self._prev_eef_pos = None
        self._prev_eef_quat = None

    def attach(self, env, robot) -> None:
        import imageio.v2 as imageio
        self.env = env
        self.robot = robot
        sens = env.external_sensors or {}
        self._left = sens.get(self._ext_cam_names[0])
        self._right = (sens.get(self._ext_cam_names[1])
                       if len(self._ext_cam_names) > 1 else None)
        self._wrist = find_wrist_sensor(robot)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # When a LeRobotEpisodeWriter is attached, imageio writes MP4s
        # directly to the LeRobot dataset's target paths so save_episode
        # can skip re-encoding. Otherwise (legacy), write to out_dir.
        if self._lerobot_writer is not None:
            tp = self._lerobot_writer.target_mp4_paths
            writer_paths = {
                "image_left":  tp["image_left"],
                "image_right": tp["image_right"],
                "wrist":       tp["wrist_image"],
            }
            # Cheap symlinks for local debug review. Resolve to absolute
            # paths so the symlinks work from any cwd — the dataset root is
            # often a relative path which would otherwise produce broken
            # links interpreted relative to out_dir.
            for wkey, target in writer_paths.items():
                local = self.out_dir / f"rollout_{'wrist' if wkey == 'wrist' else wkey}.mp4"
                if local.exists() or local.is_symlink():
                    local.unlink()
                try:
                    local.symlink_to(target.resolve())
                except OSError:
                    pass
        else:
            writer_paths = {
                "image_left":  self.out_dir / "rollout_image_left.mp4",
                "image_right": self.out_dir / "rollout_image_right.mp4",
                "wrist":       self.out_dir / "rollout_wrist.mp4",
            }
        self._writers = {
            name: imageio.get_writer(str(p), fps=self.fps, codec="libx264", quality=7)
            for name, p in writer_paths.items()
        }
        wrist_name = getattr(self._wrist, "name", None) if self._wrist else None
        print(f"[sft_recorder] attached  left={self._ext_cam_names[0]}  "
              f"right={self._ext_cam_names[1] if len(self._ext_cam_names) > 1 else '-'}  "
              f"wrist={wrist_name}", flush=True)
        if self._ltl_monitor is not None:
            self._ltl_monitor.reset()

    def _grab_frames(self):
        z = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        l = _sensor_rgb_uint8(self._left, self.resolution) if self._left is not None else z
        r = _sensor_rgb_uint8(self._right, self.resolution) if self._right is not None else z
        w = _sensor_rgb_uint8(self._wrist, self.resolution) if self._wrist is not None else z
        return l, r, w

    def record_video_only(self) -> None:
        """Append a frame to the three MP4s but skip HDF5 buffers (Phase A)."""
        l, r, w = self._grab_frames()
        self._writers["image_left"].append_data(l)
        self._writers["image_right"].append_data(r)
        self._writers["wrist"].append_data(w)
        self.n_video_frames += 1

    def record_step(self, action_np, done: bool = False) -> None:
        l, r, w = self._grab_frames()
        self._writers["image_left"].append_data(l)
        self._writers["image_right"].append_data(r)
        self._writers["wrist"].append_data(w)
        state = grab_state(self.robot)
        self._states.append(state)
        action_arr = np.asarray(action_np, dtype=np.float32)
        self._actions.append(action_arr)
        sim_state = None
        if self.record_sim_states:
            sim_state = self._dump_sim_state()
            self._sim_states.append(sim_state)
        ap_labels = None
        if self._ltl_monitor is not None:
            try:
                ltl_info = self._ltl_monitor.step(self.n_record_steps)
                ap_labels = ltl_info.get("ap")
            except Exception as e:  # noqa: BLE001
                # Disable monitor on first step error so a buggy proposition
                # doesn't kill data collection. The summary will reflect what
                # we got before the failure.
                print(f"[sft_recorder] LTL step failed at idx={self.n_record_steps}: {e}; "
                      f"disabling monitor for this variant", flush=True)
                self._ltl_monitor = None
        if self._lerobot_writer is not None:
            self._lerobot_writer.add_step(state, action_arr)
        else:
            # Legacy: store raw image arrays for HDF5. With LeRobot the MP4s
            # in <root>/videos/.../episode_NNNNNN.mp4 are the sole pixel
            # carrier and these 294 MB/variant arrays become redundant.
            self._frames_l.append(l)
            self._frames_r.append(r)
            self._frames_w.append(w)
        if self._phase_a_cache_mode:
            self._phase_a_cache.append({
                "image_left": l, "image_right": r, "wrist_image": w,
                "state": state, "action": action_arr,
                "sim_state": sim_state,
                "ap_labels": ap_labels,
            })
        self._dones.append(bool(done))
        self.n_video_frames += 1
        self.n_record_steps += 1

    def start_phase_a_cache(self) -> None:
        """Begin teeing per-step data into ``self._phase_a_cache``. Call
        before ``_record_phase_a_replay`` runs."""
        self._phase_a_cache_mode = True
        self._phase_a_cache = []

    def end_phase_a_cache(self) -> list:
        """Stop teeing and return the captured frame list (a snapshot)."""
        self._phase_a_cache_mode = False
        out = self._phase_a_cache
        self._phase_a_cache = []
        return out

    def append_cached_step(self, cached: dict) -> None:
        """Push a cached Phase A frame into the recorder's buffers + MP4
        without advancing sim or stepping the LTL monitor. Used by
        variants 1..N-1 to reuse variant 0's Phase A capture."""
        l = cached["image_left"]
        r = cached["image_right"]
        w = cached["wrist_image"]
        self._writers["image_left"].append_data(l)
        self._writers["image_right"].append_data(r)
        self._writers["wrist"].append_data(w)
        state = cached["state"]
        action_arr = cached["action"]
        self._states.append(state)
        self._actions.append(action_arr)
        if self.record_sim_states and cached.get("sim_state") is not None:
            self._sim_states.append(cached["sim_state"])
        if self._lerobot_writer is not None:
            self._lerobot_writer.add_step(state, action_arr)
        else:
            self._frames_l.append(l)
            self._frames_r.append(r)
            self._frames_w.append(w)
        self._dones.append(False)
        self.n_video_frames += 1
        self.n_record_steps += 1

    @staticmethod
    def _dump_sim_state() -> np.ndarray:
        """Serialise the current OG sim state for downstream replay."""
        import omnigibson as og

        s = og.sim.dump_state(serialized=True)
        if hasattr(s, "detach"):
            s = s.detach().cpu().numpy()
        return np.asarray(s, dtype=np.float32)

    def record_fk_step(self, gripper_cmd: float, done: bool = False) -> None:
        """Record a Phase-A step. Derives the 7D action from the eef-pose
        delta between the previous and current state (matches the format
        Phase B produces via OSC commands, so the HDF5 is homogeneous).

        ``gripper_cmd`` should be ``+1.0`` for open, ``-1.0`` for close.
        First call after ``reset_eef_history()`` records action zeros.
        """
        import omnigibson.utils.transform_utils as T
        import torch as th

        arm = self.robot.default_arm
        cur_pos, cur_quat = self.robot.get_relative_eef_pose(arm)
        cur_pos = cur_pos.float()
        cur_quat = cur_quat.float()
        if self._prev_eef_pos is None:
            dpos = np.zeros(3, dtype=np.float32)
            daa = np.zeros(3, dtype=np.float32)
        else:
            dpos_t = cur_pos - self._prev_eef_pos
            q_target = _quat_canonical_t(cur_quat)
            q_prev = _quat_canonical_t(self._prev_eef_quat)
            q_inv = T.quat_inverse(q_prev)
            q_delta = _quat_canonical_t(T.quat_multiply(q_target, q_inv))
            daa_t = T.quat2axisangle(q_delta)
            dpos = dpos_t.detach().cpu().numpy().astype(np.float32)
            daa = daa_t.detach().cpu().numpy().astype(np.float32)
        self._prev_eef_pos = cur_pos.clone()
        self._prev_eef_quat = cur_quat.clone()
        act7 = np.zeros(7, dtype=np.float32)
        act7[:3] = dpos
        act7[3:6] = daa
        act7[6] = float(gripper_cmd)
        self.record_step(act7, done=done)

    def finalize(self, success: bool, attrs: dict | None = None):
        """Close MP4 writers. On success, also write rollout.hdf5 and (if a
        LeRobot writer is attached) commit the LeRobot episode. On miss/
        no-steps, delete the empty MP4 files and abort the LeRobot writer
        so the dataset doesn't carry orphaned MP4s."""
        for w in (self._writers or {}).values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        self._writers = None
        if not success or self.n_record_steps == 0:
            for name in ("rollout_image_left.mp4", "rollout_image_right.mp4",
                         "rollout_wrist.mp4"):
                p = self.out_dir / name
                if p.exists() or p.is_symlink():
                    try:
                        p.unlink()
                    except Exception:  # noqa: BLE001
                        pass
            if self._lerobot_writer is not None:
                self._lerobot_writer.abort()
            return None
        if self._dones:
            self._dones[-1] = True
        attrs = dict(attrs) if attrs else {}
        ltl_summary = None
        if self._ltl_monitor is not None:
            ltl_summary = self._ltl_monitor.summary()
            attrs.setdefault("ltl_violated", bool(ltl_summary["violated"]))
            attrs.setdefault(
                "ltl_violation_step",
                int(ltl_summary["violation_step"]) if ltl_summary["violation_step"] is not None else -1,
            )
            attrs.setdefault("ltl_formula", ltl_summary["formula"] or "")
            attrs.setdefault("ltl_violation_count", int(ltl_summary["violation_count"]))
        hdf5_path = self._write_hdf5(attrs)
        if self._lerobot_writer is not None:
            ep_idx = self._lerobot_writer.episode_index
            extra_meta = None
            if ltl_summary is not None:
                extra_meta = {
                    "ltl_violated": bool(ltl_summary["violated"]),
                    "ltl_violation_step": (
                        int(ltl_summary["violation_step"])
                        if ltl_summary["violation_step"] is not None else None
                    ),
                }
            n = self._lerobot_writer.commit(self._lerobot_prompt or "",
                                            extra_episode_meta=extra_meta)
            tag = ""
            if ltl_summary is not None:
                tag = f"  ltl_violated={ltl_summary['violated']}"
            print(f"[sft_recorder] committed LeRobot episode_{ep_idx:06d} "
                  f"({n} frames){tag}", flush=True)
        return hdf5_path

    def _write_hdf5(self, attrs: dict) -> Path:
        import h5py
        path = self.out_dir / "rollout.hdf5"
        data = {
            "state": np.stack(self._states, axis=0),
            "action": np.stack(self._actions, axis=0),
            "done": np.array(self._dones, dtype=bool),
        }
        # In LeRobot mode the MP4 in <root>/videos/... is the pixel carrier;
        # skip the gzipped image arrays that otherwise take ~290 MB/variant.
        if self._lerobot_writer is None:
            data["image_left"] = np.stack(self._frames_l, axis=0)
            data["image_right"] = np.stack(self._frames_r, axis=0)
            data["wrist_image"] = np.stack(self._frames_w, axis=0)
        with h5py.File(str(path), "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["n_steps"] = int(self.n_record_steps)
            f.attrs["n_video_frames"] = int(self.n_video_frames)
            data_grp = f.create_group("data")
            grp = data_grp.create_group("demo_0")
            obs = grp.create_group("obs")
            if "image_left" in data:
                obs.create_dataset("image_left", data=data["image_left"],
                                   compression="gzip", compression_opts=4)
                obs.create_dataset("image_right", data=data["image_right"],
                                   compression="gzip", compression_opts=4)
                obs.create_dataset("wrist_image", data=data["wrist_image"],
                                   compression="gzip", compression_opts=4)
            obs.create_dataset("state", data=data["state"])
            grp.create_dataset("action", data=data["action"])
            grp.create_dataset("done", data=data["done"])
            if self.record_sim_states and self._sim_states:
                sim_states = np.stack(self._sim_states, axis=0)
                grp.create_dataset("states", data=sim_states,
                                   compression="gzip", compression_opts=4)
        tags = []
        if self.record_sim_states and self._sim_states:
            tags.append("+sim states")
        suffix = " " + " ".join(tags) if tags else ""
        print(f"[sft_recorder] wrote {path}  ({self.n_record_steps} steps){suffix}",
              flush=True)
        return path
