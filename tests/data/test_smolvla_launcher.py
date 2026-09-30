"""Check the delivered launch command without starting training or contacting services."""
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(os.environ.get("MANIGUARD_TEST_ROOT", Path(__file__).resolve().parents[2]))


@pytest.mark.parametrize("gpus", [None, "1", "2"])
def test_training_command_includes_camera_mapping_and_device_count(tmp_path, gpus):
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text('{"codebase_version": "v3.0"}')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ["lerobot-train", "accelerate"]:
        executable = bin_dir / name
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
        executable.chmod(0o755)
    capture = tmp_path / "args.txt"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               CAPTURE_ARGS=str(capture), MANIGUARD_SFT_DATA_ROOT=str(tmp_path / "cache"),
               HF_TOKEN="test-only-unused", WANDB_API_KEY="test-only-unused")
    for key in ["RENAME_MAP", "VIDEO_BACKEND", "TOLERANCE_S"]:
        env.pop(key, None)
    command = ["bash", str(ROOT / "tools/smolvla_sft/run_sft.sh"), "--dataset", str(dataset),
               "--repo-id", "maniguard/test", "--output", str(tmp_path / "checkpoints"),
               "--steps", "2", "--batch", "4"]
    if gpus is not None:
        command += ["--gpus", gpus]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=15)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not (tmp_path / "checkpoints").exists()  # LeRobot creates its output directory.
    args = capture.read_text().splitlines()
    mapping = [a.split("=", 1)[1] for a in args if a.startswith("--rename_map=")]
    assert len(mapping) == 1
    assert json.loads(mapping[0]) == {"observation.images.top": "observation.images.camera1",
                                    "observation.images.wrist": "observation.images.camera2"}
    assert "--policy.push_to_hub=false" in args
    assert "--batch_size=4" in args
    if gpus == "2":
        assert args[args.index("--num_processes") + 1] == "2"
    else:
        assert "--num_processes" not in args
