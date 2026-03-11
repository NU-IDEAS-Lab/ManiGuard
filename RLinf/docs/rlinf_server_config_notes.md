# RLinf Server Configuration Notes

> Deploy Env: IDEAS2 Lab Server & Northwestern Quest HPC (Main)

Key reference: [RL with BEHAVIOR benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/behavior.html). 

This section covers PPO fine-tuning of **Pi0.5** base model on BEHAVIOR manipulation tasks. The workflow: 

- install RLinf dependencies 
- configure environment variables (especially Vulkan for headless rendering) 
- download and convert Pi0.5 weights 
- compute dataset normalization stats 
- launch distributed RL training.

**Prerequisites**: BEHAVIOR environment must be configured first (see above section).

### Dataset Configuration

Update dataset paths in `b1k-baselines/baselines/openpi/src/openpi/training/config.py`:

```python
# For both pi0_b1k and pi05_b1k configs, change:
behavior_dataset_root="/home/<net_id>/SENTINEL-Lite/datasets"
```

This ensures OpenPI can locate the BEHAVIOR-1K dataset when computing normalization stats.

### Dependency Configuration

RLinf is based on [uv](https://docs.astral.sh/uv/) (install it first).

```bash
# clone RLinf repository inside SENTINEL-Lite workspace
cd ~/SENTINEL-Lite
git clone https://github.com/RLinf/RLinf.git
cd RLinf

# Configure UV paths in .bashrc
echo 'export UV_CACHE_DIR="/data/<net_id>/.cache/uv"' >> ~/.bashrc
echo 'export UV_PYTHON_INSTALL_DIR="/data/<net_id>/.local/share/uv/python"' >> ~/.bashrc
echo 'export UV_TOOL_DIR="/data/<net_id>/.local/share/uv/tools"' >> ~/.bashrc
source ~/.bashrc

# Symlink venv to /data for space
mkdir -p /data/<net_id>/venvs
ln -s /data/<net_id>/venvs/SENTINEL-Lite-RLinf .venv

# Set BEHAVIOR_PATH to reuse existing BEHAVIOR installation
export BEHAVIOR_PATH=/home/<net_id>/SENTINEL-Lite

# Install dependencies (--no-root because sys_deps already installed)
bash requirements/install.sh embodied --model openpi --env behavior --no-root
source .venv/bin/activate

# Fix package versions for compatibility
uv pip install numpy==1.26.4 protobuf==3.20.3 ml_dtypes==0.5.3
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install flash-attn (check Python 3.10, Torch 2.5.1+cu124, CUDA 12.4)
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# Install lerobot "forcefully" due to version mismatch (needed for norm stats computation)
uv pip install lerobot --no-deps

# Lerobot import compatibility (choose one depending on which lerobot you use):
# - If using a lerobot that HAS lerobot.common (e.g. a fork): patch OmniGibson to use it:
#   sed -i 's/from lerobot\.datasets/from lerobot.common.datasets/g' OmniGibson/omnigibson/learning/datas/lerobot_dataset.py
#   sed -i 's/from lerobot\.constants/from lerobot.common.constants/g' OmniGibson/omnigibson/learning/datas/lerobot_dataset.py
# - If using official PyPI lerobot (no lerobot.common): patch OmniGibson and openpi to use lerobot.*:
#   sed -i 's/lerobot\.common\./lerobot./g' OmniGibson/omnigibson/learning/datas/lerobot_dataset.py
#   sed -i 's/lerobot\.common\./lerobot./g' OmniGibson/omnigibson/learning/utils/lerobot_utils.py
#   sed -i 's/lerobot\.common\./lerobot./g' RLinf/.venv/lib/python3.10/site-packages/openpi/training/data_loader.py

# Symlink standalone Isaac Sim kit folder (if using pip-installed isaacsim)
# you need to have the standalone Isaac Sim installation first. Refer to IsaacSim official doc for installation.
ln -s /data/<net_id>/isaacsim/kit .venv/lib/python3.10/site-packages/isaacsim/kit
```

### Environment Variables

Add to `~/.bashrc`:

```bash
# Isaac Sim (use pip-installed version in RLinf venv)
export ISAAC_PATH="/home/<net_id>/SENTINEL-Lite/RLinf/.venv/lib/python3.10/site-packages/isaacsim"

# BEHAVIOR dataset path
export OMNIGIBSON_DATA_PATH="/home/<net_id>/SENTINEL-Lite/datasets"

# RLinf BEHAVIOR integration
export BEHAVIOR_PATH="/home/<net_id>/SENTINEL-Lite"

# Headless rendering (critical for server)
export OMNIGIBSON_HEADLESS=1
export NVIDIA_DRIVER_CAPABILITIES=all
```

**Vulkan ICD Fix** (for headless rendering without sudo):

Create local ICD file at `~/SENTINEL-Lite/RLinf/nvidia_icd_local.json` (this is already provided in our repository):

```json
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "api_version" : "1.4.303"
    }
}
```

Then fix the uv venv activation script (already fixed in our repository. If necessary, follow the steps below):

```bash
# Edit RLinf/.venv/bin/activate, replace lines 132-133:
export VK_DRIVER_FILES="/home/<net_id>/SENTINEL-Lite/RLinf/nvidia_icd_local.json"
export VK_ICD_FILENAMES="/home/<net_id>/SENTINEL-Lite/RLinf/nvidia_icd_local.json"
```

And update `run_embodiment.sh` (lines 27-29):

```bash
export OMNIGIBSON_HEADLESS=1
export NVIDIA_DRIVER_CAPABILITIES=all
export VK_ICD_FILENAMES="/home/<net_id>/SENTINEL-Lite/RLinf/nvidia_icd_local.json"
```

### Model Download & Configuration

Download Pi0.5 base model (refer to [openpi repo](https://github.com/Physical-Intelligence/openpi?tab=readme-ov-file#base-models)) from **Google Cloud Storage (gs)**. The π₀.₅ base checkpoint is hosted at `gs://openpi-assets/checkpoints/pi05_base`, not on Hugging Face. Then convert JAX weights to PyTorch if needed:

```bash
# Download Pi0.5 base model from GCS (requires gsutil: pip install gsutil or Google Cloud SDK)
mkdir -p /data/<net_id>/SENTINEL-Lite/checkpoints
cd /data/<net_id>/SENTINEL-Lite/checkpoints
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base .

# clone BEHAVIOR Baselines repository
git clone https://github.com/StanfordVL/b1k-baselines.git --recurse-submodules



# check whether the base model folder has the <model.safetensors> file which is the PyTorch format of the checkpoint. If not, MUST convert it first:
cd ~/SENTINEL-Lite/b1k-baselines/baselines/openpi
# Double check that you have transformers 4.53.2 installed: uv pip show transformers
# Apply the transformers library patches:
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/

# Convert JAX weights to PyTorch format (Pi0.5 base)
# - config_name: use pi05_libero (same Pi0.5 model arch; RLinf uses pi05_behavior for BEHAVIOR data).
# - checkpoint_dir: directory that contains params/ (and optionally assets/), i.e. the pi05_base folder.
# - output_path: where to write model.safetensors and config.json; can be the same as checkpoint_dir for in-place conversion.
python examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_libero \
    --checkpoint_dir /data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base \
    --output_path /data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base

# Compute normalization stats for BEHAVIOR dataset (this has been done in our repository: ~/SENTINEL-Lite/b1k-baselines/outputs/assets/pi05_b1k/behavior-1k/2025-challenge-demos/norm_stats.json)
export PYTHONPATH=$PWD/b1k-baselines/baselines/openpi/src:$PYTHONPATH
cd ~/SENTINEL-Lite/b1k-baselines
python baselines/openpi/scripts/compute_norm_stats.py --config-name pi05_b1k

# Copy norm_stats.json to model directory
mkdir -p /data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base/physical-intelligence/behavior
cp norm_stats.json /data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base/physical-intelligence/behavior/

# Add Pi0.5 config to RLinf (if not already present)
# Edit RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py
# Add TrainConfig for "pi05_behavior" with:
#   - pi05=True, action_horizon=10, discrete_state_input=False
#   - asset_path and weight_path pointing to /data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base
```

Update model paths in `RLinf/examples/embodiment/config/behavior_ppo_openpi.yaml`:

```yaml
rollout:
  model:
    model_path: "/data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base"

actor:
  model:
    model_path: "/data/<net_id>/SENTINEL-Lite/checkpoints/pi05_base"
    openpi:
      config_name: "pi05_behavior"
```

#### Quest Server environment

On Quest (or when pi05_base lives under GPFS), keep the above steps and in addition:

- **Pi0.5 base path**: Set the env so dataconfig and yaml overrides resolve correctly:
  ```bash
  export SENTINEL_PI05_BASE=/gpfs/projects/p33203/checkpoints/pi05_base
  ```
  The `pi05_behavior` config in `RLinf/rlinf/models/embodiment/openpi/dataconfig/__init__.py` uses this for `assets_dir`, `weight_loader`, and `pytorch_weight_path` when loading the Pi0.5 base model.

- **YAML**: In `RLinf/examples/embodiment/config/behavior_ppo_openpi.yaml`, set both `rollout.model.model_path` and `actor.model.model_path` to the same path, e.g. `/gpfs/projects/p33203/checkpoints/pi05_base`, or override at run time, e.g.:
  ```bash
  actor.model.model_path=/gpfs/projects/p33203/checkpoints/pi05_base rollout.model.model_path=/gpfs/projects/p33203/checkpoints/pi05_base
  ```

### Running Scripts

**Configuration**: `RLinf/examples/embodiment/config/behavior_ppo_openpi.yaml`

Key settings for multi-GPU server:

```yaml
cluster:
  num_nodes: 1
  component_placement: # can use 0-3 to show multi-GPU assignment. Refer to the [RLinf documentation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/behavior.html#running-scripts) for more details.
    env: 0       # Simulation environments
    rollout: 2   # Policy inference
    actor: 3     # Policy training

env:
  train:
    total_num_envs: 1  # Must be divisible by num GPUs assigned to env
    video_cfg:
      save_video: False  # Disable during training for performance
  eval:
    total_num_envs: 1
    video_cfg:
      save_video: True   # Enable for eval rollouts
      video_base_dir: ${runner.logger.log_path}/video/eval
```

**GPU Selection** (if some GPUs are occupied):

```bash
# Check GPU usage
nvitop

# Mask occupied GPUs (e.g., GPU 0 and 4 are busy)
export CUDA_VISIBLE_DEVICES=1,2,3,5,6,7

# Then component_placement uses logical GPU IDs (0-5 after masking)
```

**Launch Training**:

```bash
cd ~/SENTINEL-Lite/RLinf
source .venv/bin/activate

# Verify environment variables
echo $ISAAC_PATH
echo $OMNIGIBSON_DATA_PATH
echo $VK_ICD_FILENAMES

# Start training
bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi
```

### Debug Tips

**Common Issues**:

1. `**ModuleNotFoundError: No module named 'spot'`**
  - Spot (LTL library) is optional. Make imports conditional in `omnigibson/utils/ltl_utils.py` and `omnigibson/tasks/behavior_task.py`
2. `**RuntimeError: Failed to acquire interface: omni::kit::IApp**`
  - Isaac Sim pip package missing `kit` folder. Symlink from standalone installation:
3. `**VkResult: ERROR_INCOMPATIBLE_DRIVER**` (Vulkan error)
  - Root cause: `VK_ICD_FILENAMES` pointing to non-existent file
  - Solution: Create local ICD file (see Environment Variables section) and update `.venv/bin/activate`
4. `**torch.OutOfMemoryError: CUDA out of memory**`
  - Reduce `total_num_envs` in YAML config
  - Use `CUDA_VISIBLE_DEVICES` to mask busy GPUs
  - Ensure `component_placement` values don't overlap on same GPU
  - ***PLEASE use QUEST***
5. **Video writer `TypeError: cannot unpack non-iterable NoneType`**
  - Fixed in `behavior_env.py` by reinitializing writer after `flush_video()`
  - Disable training video (`save_video: False`) to reduce overhead
6. `**IndexError: list index out of range` in `isaacsim/__init__.py**`
  - Pip-installed Isaac Sim has nested `simulation_app` directory
  - Fix: Add extra `"simulation_app"` to glob path in `__init__.py` line 98
7. `**ImportError: libcusparse.so.12: undefined symbol**`
  - Caused by incorrect `LD_LIBRARY_PATH` interfering with PyTorch's CUDA libs
  - Solution: Use local Vulkan ICD file with absolute paths instead of modifying `LD_LIBRARY_PATH`


### RLinf HPC/Quest Deployment Debug Notes

This section documents the specific configurations required to run RLinf with BEHAVIOR/OmniGibson on the Northwestern Quest HPC (H100 GPUs, Headless).

#### 1. Transformers & OpenPi Compatibility
- **Issue**: `openpi` patches are strictly tied to specific `transformers` internal structures (`LossKwargs`, `masking_utils`). Newer versions (e.g., 5.2.0) or older versions (e.g., 4.40.1) often lead to `ImportError`.
- **Solution**: Forced `transformers==4.53.2`. This version contains both `AutoModelForVision2Seq` and the modular `masking_utils` required by OpenPi's `modeling_gemma.py`.
- **Action**: Re-applied patches from `openpi/models_pytorch/transformers_replace/` to the `transformers` installation path in the venv.

#### 2. BEHAVIOR Model Assets
- **Issue**: Missing `norm_stats.json` for the Pi0.5 model in the expected OpenPi directory.
- **Solution**: Copied the validated `norm_stats.json` to `/gpfs/projects/p33203/checkpoints/pi05_base/physical-intelligence/behavior/norm_stats.json`.

#### 3. Headless Vulkan Rendering (H100 GPUs)
- **Issue**: Isaac Sim fails to initialize Vulkan on headless H100 nodes, often reporting `ERROR_INCOMPATIBLE_DRIVER`.
- **Solution**: Created a local Vulkan ICD configuration (`nvidia_icd_local.json`) and used specific environment variables to bypass device selection issues.
- **Environment Variables**:
  - `VK_DRIVER_FILES`: Points to the local ICD JSON.
  - `MESA_VK_DEVICE_SELECT`: Set to `10de:2330` (H100 PCI ID) to force selection of the NVIDIA hardware.
  - `OMNIGIBSON_HEADLESS=1`: Mandatory for server deployment.

#### 4. System Library Dependencies
- **Issue**: Missing `libGLU.so.1` causes segfaults in Omniverse Kit extensions.
- **Solution**: Loaded the HPC module `mesa-glu` and ensured it is present in `.bashrc` and the startup script.


### Results & Visualization

**TensorBoard**:

```bash
tensorboard --logdir ~/SENTINEL-Lite/RLinf/logs --port 6006
```

**Video Output**:

- Eval videos saved to `logs/<timestamp>-behavior_ppo_openpi/video/eval/behavior_video_N.mp4`
- Videos show robot's egocentric view: left wrist + right wrist + head camera (448×672 resolution)
- Each eval epoch generates a separate numbered video file
