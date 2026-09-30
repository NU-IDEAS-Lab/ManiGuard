"""Local conversion integration tests; run in the LeRobot 0.5.1 training environment."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(os.environ.get("MANIGUARD_TEST_ROOT", Path(__file__).resolve().parents[2]))
CAMERAS = ["image_left", "image_right", "image_opposite", "image_left_shoulder", "wrist_image"]


def make_source(root):
    (root / "meta").mkdir(parents=True)
    features = {k: {"dtype": "video", "shape": [32, 32, 3], "names": ["height", "width", "channel"],
                    "info": {"video.fps": 30, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "has_audio": False}} for k in CAMERAS}
    features.update({k: {"dtype": "float32", "shape": [8], "names": None}
                     for k in ["state", "actions", "actions_commanded"]})
    features.update({k: {"dtype": "float32" if k == "timestamp" else "int64", "shape": [1], "names": None}
                     for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]})
    info = dict(codebase_version="v2.1", robot_type="franka", fps=30, total_episodes=2,
                total_frames=8, total_tasks=1, total_videos=10, total_chunks=1, chunks_size=1000,
                splits={"train": "0:2"}, features=features,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Move jar"}) + "\n")
    episodes, stats = [], []
    for ep in range(2):
        values = (np.arange(32).reshape(4, 8) / 100 + ep).astype(np.float32)
        data = {"state": values.tolist(), "actions": (values + .25).tolist(),
                "actions_commanded": (values + .5).tolist(), "timestamp": np.arange(4, dtype=np.float32) / 30,
                "frame_index": list(range(4)), "episode_index": [ep] * 4,
                "index": list(range(ep * 4, ep * 4 + 4)), "task_index": [0] * 4}
        parquet = root / info["data_path"].format(episode_chunk=0, episode_index=ep)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), parquet)
        for cam in CAMERAS:
            path = root / info["video_path"].format(episode_chunk=0, episode_index=ep, video_key=cam)
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width = stream.height = 32
                stream.pix_fmt = "yuv420p"
                for i in range(4):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), ep * 50 + i, np.uint8), format="rgb24")
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
        episodes.append({"episode_index": ep, "tasks": ["Move jar"], "length": 4})
        stats.append({"episode_index": ep, "stats": {
            k: {"mean": np.asarray(v).mean(axis=0).reshape(-1).tolist(),
                "std": np.asarray(v).std(axis=0).reshape(-1).tolist(),
                "min": np.asarray(v).min(axis=0).reshape(-1).tolist(),
                "max": np.asarray(v).max(axis=0).reshape(-1).tolist(), "count": [4]}
            for k, v in data.items()}})
    for name, rows in [("episodes", episodes), ("episodes_stats", stats)]:
        (root / f"meta/{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


try:
    TRAINING_RUNTIME = version("lerobot") == "0.5.1"
except PackageNotFoundError:
    TRAINING_RUNTIME = False


@unittest.skipUnless(TRAINING_RUNTIME, "requires the LeRobot 0.5.1 training environment")
class SmolVLADeliveryTest(unittest.TestCase):
    def test_training_ready_conversion_preserves_source_and_values(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as temp:
            src, out = Path(temp) / "source", Path(temp) / "prepared"
            make_source(src)
            original = hashes(src)
            cmd = [sys.executable, str(ROOT / "tools/smolvla_sft/prepare_dataset.py"),
                   "--src", str(src), "--out", str(out), "--repo-id", "maniguard/test"]
            result = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(hashes(src), original)
            info = json.loads((out / "meta/info.json").read_text())
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(set(k for k in info["features"] if k.startswith("observation.images.")),
                             {"observation.images.top", "observation.images.wrist"})
            self.assertEqual(len(list((out / "videos").rglob("*.mp4"))), 4)
            for source_camera, target_camera in [("image_left", "observation.images.top"),
                                                  ("wrist_image", "observation.images.wrist")]:
                converted_videos = sorted((out / "videos" / target_camera).rglob("*.mp4"))
                for episode, path in enumerate(converted_videos):
                    source_video = src / f"videos/chunk-000/{source_camera}/episode_{episode:06d}.mp4"
                    with av.open(str(source_video)) as a, av.open(str(path)) as b:
                        original_frames = [f.to_ndarray(format="rgb24") for f in a.decode(video=0)]
                        converted_frames = [f.to_ndarray(format="rgb24") for f in b.decode(video=0)]
                    np.testing.assert_array_equal(original_frames, converted_frames)
            old = pa.concat_tables([pq.read_table(p) for p in sorted((src / "data").rglob("*.parquet"))])
            new = pa.concat_tables([pq.read_table(p) for p in sorted((out / "data").rglob("*.parquet"))])
            for a, b in [("state", "observation.state"), ("actions", "action")]:
                self.assertEqual(old[a].to_pylist(), new[b].to_pylist())
            dataset = LeRobotDataset("maniguard/test", root=out, video_backend="pyav", tolerance_s=.01)
            self.assertEqual(len(dataset), 8)
            self.assertEqual(tuple(dataset[0]["observation.state"].shape), (8,))
            self.assertEqual(tuple(dataset[7]["observation.images.wrist"].shape), (3, 32, 32))
            # Existing output must never be overwritten just because its count matches.
            saved = hashes(out)
            rerun = subprocess.run(cmd, text=True, capture_output=True, cwd=ROOT)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(hashes(out), saved)


if __name__ == "__main__":
    unittest.main()
