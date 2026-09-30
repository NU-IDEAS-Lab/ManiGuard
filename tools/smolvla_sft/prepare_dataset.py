#!/usr/bin/env python
"""Prepare a two-camera LeRobot v3.0 copy for SmolVLA training.

Read ManiGuard's v2.1 export, rename observation/action keys, retain the selected
external camera and wrist camera, and convert the copy with LeRobot 0.5.1.
State/action values are preserved and video streams are copied without re-encoding.
Use a new output directory; the source dataset is never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from maniguard.smolvla_sft import embodiment as emb


def prepare(src: Path, out: Path, external_cam: str, repo_id: str) -> dict:
    src, out = src.resolve(), out.resolve()
    if src == out or src in out.parents or out in src.parents:
        raise ValueError("Source and output must be separate, non-nested directories")
    # The upstream converter uses these two sibling paths while converting.
    for candidate in (out, Path(f"{out}_old"), Path(f"{out}_v30")):
        if candidate.exists():
            raise FileExistsError(f"Use a fresh output path; already exists: {candidate}")
    info = json.loads((src / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Expected a ManiGuard LeRobot v2.1 source dataset")
    rename = emb.rename_map(external_cam)
    drop = set(emb.dropped_streams(external_cam))
    from lerobot.scripts.convert_dataset_v21_to_v30 import convert_dataset

    episodes = int(info["total_episodes"])
    chunk_size = int(info.get("chunks_size", 1000))
    videos = {s: t for s, t in rename.items() if t.startswith("observation.images.")}
    columns = {s: t for s, t in rename.items() if s not in videos}
    (out / "meta").mkdir(parents=True)
    new_info = dict(info)
    new_info.update(
        features={rename.get(k, k): v for k, v in info["features"].items() if k not in drop},
        total_videos=episodes * len(videos), repo_id=repo_id,
    )
    (out / "meta/info.json").write_text(json.dumps(new_info, indent=2) + "\n")
    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copyfile(src / "meta" / name, out / "meta" / name)

    def rekey(stats):
        return {rename.get(k, k): v for k, v in stats.items() if k not in drop}

    with (out / "meta/episodes_stats.jsonl").open("w") as target:
        for line in (src / "meta/episodes_stats.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["stats"] = rekey(row["stats"])
                target.write(json.dumps(row) + "\n")
    aggregate = src / "meta/stats.json"
    if aggregate.exists():
        (out / "meta/stats.json").write_text(json.dumps(rekey(json.loads(aggregate.read_text()))) + "\n")

    for episode in range(episodes):
        fields = {"episode_chunk": episode // chunk_size, "episode_index": episode}
        relative = info["data_path"].format(**fields)
        table = pq.read_table(src / relative)
        table = table.select([key for key in table.column_names if key not in drop])
        table = table.rename_columns([columns.get(key, key) for key in table.column_names])
        (out / relative).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / relative)
        for source_key, target_key in videos.items():
            source = src / info["video_path"].format(**fields, video_key=source_key)
            target = out / info["video_path"].format(**fields, video_key=target_key)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if (episode + 1) % 100 == 0 or episode + 1 == episodes:
            print(f"[prepare] {episode + 1}/{episodes} episodes", flush=True)

    # Keep one episode per video file so timestamps stay episode-relative.
    convert_dataset(repo_id=repo_id, root=out, push_to_hub=False,
                    force_conversion=True, video_file_size_in_mb=0)
    converted = json.loads((out / "meta/info.json").read_text())
    if converted["codebase_version"] != "v3.0" or converted["total_episodes"] != episodes:
        raise RuntimeError("Converted dataset version or episode count does not match")
    shutil.rmtree(Path(f"{out}_old"))  # intermediate copy created by this conversion
    result = {"repo_id": repo_id, "episodes": episodes, "root": str(out),
              "external_cam": external_cam, "codebase_version": "v3.0"}
    print(f"[prepare] DONE {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="source five-camera LeRobot v2.1 directory")
    parser.add_argument("--out", required=True, help="new two-camera LeRobot v3.0 directory")
    parser.add_argument("--repo-id", required=True, help="local dataset identifier")
    parser.add_argument("--external-cam", default=emb.DEFAULT_EXTERNAL_CAM, choices=emb.EXTERNAL_CAM_CHOICES)
    args = parser.parse_args()
    prepare(Path(args.src), Path(args.out), args.external_cam, args.repo_id)


if __name__ == "__main__":
    main()
