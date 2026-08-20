#!/usr/bin/env bash
# Fixture check for prepare_dataset.py (RUN in the fork's gr00t venv; it imports gr00t
# via the embodiment). Verifies: videos/ + data/ are symlinked, meta/ is a real copy
# with modality.json added, and the SOURCE dataset is left untouched.
set -euo pipefail
S=$(mktemp -d); O=$(mktemp -d)/out
mkdir -p "$S/videos/chunk-000/image_left" "$S/data/chunk-000" "$S/meta"
echo x > "$S/videos/chunk-000/image_left/episode_000000.mp4"
echo x > "$S/data/chunk-000/episode_000000.parquet"
printf '{"features":{"state":{"dtype":"float32"},"actions":{"dtype":"float32"}}}' > "$S/meta/info.json"
# stats step is allowed to fail on the fake parquet (|| true); structure is written first.
python tools/gr00t_sft/prepare_dataset.py --src "$S" --out "$O" --stats-dir /nonexistent_stats 2>/dev/null || true
[ -L "$O/videos" ]         && echo "videos symlinked OK" || { echo "FAIL videos not symlink"; exit 1; }
[ -L "$O/data" ]           && echo "data symlinked OK"   || { echo "FAIL data not symlink"; exit 1; }
[ -f "$O/meta/info.json" ] && echo "meta copied OK"      || { echo "FAIL meta"; exit 1; }
[ -f "$O/meta/modality.json" ] && echo "modality OK"     || { echo "FAIL modality"; exit 1; }
[ ! -e "$S/meta/modality.json" ] && echo "src untouched OK" || { echo "FAIL src mutated"; exit 1; }
rm -rf "$S" "$(dirname "$O")"
echo "FIXTURE PASS"
