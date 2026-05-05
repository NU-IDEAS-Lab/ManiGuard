# Vast 5090 BEHAVIOR / SENTINEL Handoff

Date: 2026-05-01 UTC

## Goal

Continue SENTINEL-Lite + BEHAVIOR-1K setup on the new Vast AI RTX 5090 machine, then run RL PPO
speed benchmark.

Primary repo path expected on new machine:

```bash
/workspace/SENTINEL-Lite

## New Vast Instance

Instance / contract:

contract id: 35933568

Image used:

nvidia/cuda:12.8.1-devel-ubuntu24.04

Create command used:

vastai create instance 23733483 \
  --image nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --disk 800 \
  --ssh \
  --direct \
  --onstart-cmd "nvidia-smi"

Verified GPU / driver:

GPU: NVIDIA GeForce RTX 5090
Driver: 580.65.06
CUDA shown by nvidia-smi: 13.0
VRAM: 32607 MiB

This driver is important. Previous host had driver 595.45.04, and Isaac Sim / RTX renderer crashed
in librtx.scenedb.plugin.so. Use this R580 host for Isaac Sim 5.1.

Disk:

/ is 800G overlay
/workspace was manually created

## Base System Packages Already Installed

Already ran:

apt-get update
apt-get install -y git git-lfs wget curl bzip2 ca-certificates build-essential tmux htop \
  libxt6 libglu1-mesa libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libx11-6
git lfs install

## Repo

Expected clone:

cd /workspace
git clone --recursive git@github.com:NU-IDEAS-Lab/SENTINEL-Lite.git
cd /workspace/SENTINEL-Lite
git checkout feat/behavior-main-isaac5
git submodule update --init --recursive

## Conda / BEHAVIOR Setup

Conda path on this machine is expected to be:

/workspace/miniconda3

BEHAVIOR env:

/workspace/miniconda3/envs/behavior

Activation:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior

BEHAVIOR setup has completed successfully with:

✓ Installed OmniGibson + Isaac Sim
✓ Installed BDDL
✓ Installed JoyLo
✓ Installed OmniGibson with primitives support
✓ Installed evaluation support
✓ Downloaded datasets

Earlier issue was:

./setup.sh: line 265: pip: command not found

Fix was to install pip into the env and rerun setup without --new-env:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda install -y -n behavior -c conda-forge pip
conda activate behavior
hash -r

Then:

cd /workspace/SENTINEL-Lite/behavior-1k

export OMNI_KIT_ACCEPT_EULA=YES
export TORCH_CUDA_ARCH_LIST=12.0

./setup.sh \
  --omnigibson \
  --bddl \
  --joylo \
  --dataset \
  --eval \
  --primitives \
  --accept-conda-tos \
  --accept-nvidia-eula \
  --accept-dataset-tos

## Recommended Env Hooks

Run once after setup:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior

mkdir -p /workspace/miniconda3/envs/behavior/etc/conda/activate.d
mkdir -p /workspace/miniconda3/envs/behavior/etc/conda/deactivate.d

cat > /workspace/miniconda3/envs/behavior/etc/conda/activate.d/behavior_cuda.sh <<'EOF'
export _OLD_CUDA_HOME="${CUDA_HOME:-}"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDA_ROOT="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="12.0"
export OMNI_KIT_ACCEPT_EULA=YES
EOF

cat > /workspace/miniconda3/envs/behavior/etc/conda/deactivate.d/behavior_cuda.sh <<'EOF'
if [ -n "${_OLD_CUDA_HOME+x}" ]; then
  export CUDA_HOME="$_OLD_CUDA_HOME"
else
  unset CUDA_HOME
fi
unset CUDA_PATH CUDA_ROOT CUDACXX TORCH_CUDA_ARCH_LIST OMNI_KIT_ACCEPT_EULA _OLD_CUDA_HOME
EOF

conda deactivate
conda activate behavior

Verify:

echo $CUDA_HOME
echo $TORCH_CUDA_ARCH_LIST
which python
which pip
which nvcc
python --version
python -m pip --version
nvcc -V

## SENTINEL Python Install

Run inside behavior env:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior

cd /workspace/SENTINEL-Lite
pip install -e ".[rl,serve]"

## Benchmark Dataset

Need this HF dataset under:

/workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-full-
perturbed_with_base-20260426

Download command:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior

pip install -U huggingface_hub

cd /workspace/SENTINEL-Lite

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "IDEAS-Lab-Northwestern/final_unique_accepted-goal_region_sphere-full-
perturbed_with_base-20260426"
target = Path("/workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-full-
perturbed_with_base-20260426")
target.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(target),
)
print(path)
PY

Expected useful file for PPO smoke:

/workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-full-
perturbed_with_base-20260426/table/task_0022/base/diagnostics.jsonl

## Isaac Sim Smoke Test

First verify Vulkan ICD:

ls -l /etc/vulkan/icd.d/nvidia_icd.json

Then run:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /workspace/SENTINEL-Lite

OMNI_KIT_ACCEPT_EULA=YES \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
OMNIGIBSON_HEADLESS=1 \
python - <<'PY'
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "multi_gpu": False})
print("SimulationApp OK")
app.close()
PY

If this segfaults on R580, collect full log. On the previous R595 host this crashed inside RTX
renderer.

## PPO Smoke Test

Run short test first:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /workspace/SENTINEL-Lite

OMNI_KIT_ACCEPT_EULA=YES \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
OMNIGIBSON_HEADLESS=1 \
python -m sentinel.rl.algorithms.ppo \
  --diagnostics-file /workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-
full-perturbed_with_base-20260426/table/task_0022/base/diagnostics.jsonl \
  --num-envs 4 \
  --total-timesteps 2048 \
  --n-steps 128 \
  --batch-size 256 \
  --save-freq 1024 \
  --output-dir outputs/rl_ppo_5090_n4_smoke

## PPO 200k Benchmark

If smoke passes:

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate behavior
cd /workspace/SENTINEL-Lite

OMNI_KIT_ACCEPT_EULA=YES \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
OMNIGIBSON_HEADLESS=1 \
python -m sentinel.rl.algorithms.ppo \
  --diagnostics-file /workspace/SENTINEL-Lite/datasets/final_unique_accepted-goal_region_sphere-
full-perturbed_with_base-20260426/table/task_0022/base/diagnostics.jsonl \
  --num-envs 4 \
  --total-timesteps 200000 \
  --n-steps 128 \
  --batch-size 256 \
  --save-freq 50000 \
  --output-dir outputs/rl_ppo_5090_n4

For raw speed testing, omit wandb unless explicitly needed.

## Codex CLI Setup On New Machine

Install Node/npm if needed:

apt-get update
apt-get install -y nodejs npm
node -v
npm -v

Install Codex CLI:

npm install -g @openai/codex
codex --version
which codex

Login:

codex --login

Then start in repo:

cd /workspace/SENTINEL-Lite
codex

Prefer codex --login. Do not paste API keys into chat. If login is not usable, set key locally
only:

export OPENAI_API_KEY="..."

## Key Notes

- Use branch feat/behavior-main-isaac5.
- Use R580 driver host. Avoid R595 for Isaac Sim 5.1 RTX rendering.
- Keep everything under /workspace for path consistency.
- BEHAVIOR setup should not be rerun with --new-env once env exists.
- If pip resolves to base Python 3.13, the env is not activated correctly or pip is missing inside
  behavior.
- Important env vars for tests:
    - OMNI_KIT_ACCEPT_EULA=YES
    - VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
    - OMNIGIBSON_HEADLESS=1
    - TORCH_CUDA_ARCH_LIST=12.0

