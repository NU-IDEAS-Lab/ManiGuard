"""Field-level diff of two LeRobot datasets (reference vs candidate).

Compares, for byte/field-level identity:
  * ``meta/info.json`` (all keys),
  * ``meta/tasks.jsonl`` / ``episodes.jsonl`` / ``episodes_stats.jsonl``,
  * every episode's parquet (ALL columns, exact values),
  * every episode's videos (md5).

Used to prove a parallel / merged dataset byte-identical to a serial ``to_lerobot`` reference
(the ``--verify`` self-check in ``to_lerobot_parallel`` and ad-hoc gold-standard checks).
``diff_datasets`` returns the list of differences (empty list == identical).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jl(path):
    return [json.loads(x) for x in open(path)]


def diff_datasets(a, b) -> list[str]:
    """Return a list of human-readable differences between LeRobot datasets ``a`` and ``b``
    (empty == identical). Parquet comparison is exact over every column of every episode."""
    a, b = Path(a), Path(b)
    diffs: list[str] = []

    ia = json.load(open(a / "meta" / "info.json"))
    ib = json.load(open(b / "meta" / "info.json"))
    for k in sorted(set(ia) | set(ib)):
        if ia.get(k) != ib.get(k):
            diffs.append(f"info.json[{k}]: {str(ia.get(k))[:80]} != {str(ib.get(k))[:80]}")

    for m in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"):
        la, lb = _jl(a / "meta" / m), _jl(b / "meta" / m)
        if len(la) != len(lb):
            diffs.append(f"{m}: len {len(la)} != {len(lb)}")
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                diffs.append(f"{m}[{i}] first differs")
                break

    ne = ia.get("total_episodes")
    ch = ia.get("chunks_size", 1000)
    vkeys = [k for k, v in ia["features"].items() if v.get("dtype") == "video"]
    for e in range(ne):
        c = e // ch
        rel_pq = f"data/chunk-{c:03d}/episode_{e:06d}.parquet"
        ta, tb = pq.read_table(a / rel_pq), pq.read_table(b / rel_pq)
        if ta.schema.names != tb.schema.names:
            diffs.append(f"ep{e}: parquet columns differ")
        else:
            for col in ta.schema.names:
                if ta.column(col).to_pylist() != tb.column(col).to_pylist():
                    diffs.append(f"ep{e}: parquet column '{col}' differs")
                    break
        for k in vkeys:
            rel_v = f"videos/chunk-{c:03d}/{k}/episode_{e:06d}.mp4"
            if _md5(a / rel_v) != _md5(b / rel_v):
                diffs.append(f"ep{e}: video '{k}' md5 differs")
    return diffs


if __name__ == "__main__":
    d = diff_datasets(sys.argv[1], sys.argv[2])
    if d:
        print(f"DIFFS: {len(d)}")
        for x in d[:40]:
            print("  " + x)
        print("VERDICT: NOT IDENTICAL")
        sys.exit(1)
    print("VERDICT: IDENTICAL")
