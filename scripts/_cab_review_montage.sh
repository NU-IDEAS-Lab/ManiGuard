#!/usr/bin/env bash
# Build a review montage for a cabinet datagen traj dir: a 2x2 grid of the four external 256x256
# cameras (NO aspect squish — kept square) beside the wrist view, with blue stream labels.
# Usage: scripts/_cab_review_montage.sh <traj_dir> [out_name]
set -euo pipefail
D="${1:?usage: _cab_review_montage.sh <traj_dir> [out_name]}"
OUT="${2:-_review.mp4}"
LBL="fontcolor=white:fontsize=22:box=1:boxcolor=black:boxborderw=7:x=10:y=10"
ffmpeg -y -loglevel error \
  -i "$D/image_left.mp4" -i "$D/image_right.mp4" -i "$D/image_opposite.mp4" \
  -i "$D/image_left_shoulder.mp4" -i "$D/wrist_image.mp4" \
  -filter_complex \
  "[0:v]scale=384:384,drawtext=text='left':$LBL[a];\
   [1:v]scale=384:384,drawtext=text='right':$LBL[b];\
   [2:v]scale=384:384,drawtext=text='opposite':$LBL[c];\
   [3:v]scale=384:384,drawtext=text='left_shoulder':$LBL[d];\
   [a][b]hstack[top];[c][d]hstack[bot];[top][bot]vstack[grid];\
   [4:v]scale=768:768,drawtext=text='wrist':$LBL[w];[grid][w]hstack[out]" \
  -map "[out]" -r 20 "$D/$OUT"
echo "montage -> $D/$OUT"
