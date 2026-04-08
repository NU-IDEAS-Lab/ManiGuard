#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a local Pi0.5 PyTorch checkpoint tree for RLinf Sentinel evaluation.
#
# Required env vars:
#   OPENPI_REPO: local clone of https://github.com/Physical-Intelligence/openpi
# Optional env vars:
#   PI05_BASE_DIR: checkpoint output dir, default: /home/yiyanpeng/data/SENTINEL-Lite/checkpoints/pi05_base
#   PI05_GCS_URI:  default: gs://openpi-assets/checkpoints/pi05_base
#   NORM_STATS_SRC: path to a prepared norm_stats.json for the target data contract
#   NORM_STATS_ASSET_ID: checkpoint-relative asset id under assets/, default: franka
#   PYTHON_BIN:    interpreter inside the openpi environment, default: python

OPENPI_REPO="${OPENPI_REPO:-}"
PI05_BASE_DIR="${PI05_BASE_DIR:-/home/yiyanpeng/data/SENTINEL-Lite/checkpoints/pi05_base}"
PI05_GCS_URI="${PI05_GCS_URI:-gs://openpi-assets/checkpoints/pi05_base}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NORM_STATS_SRC="${NORM_STATS_SRC:-}"
NORM_STATS_ASSET_ID="${NORM_STATS_ASSET_ID:-franka}"

if [[ -z "${OPENPI_REPO}" ]]; then
  echo "OPENPI_REPO is required."
  echo "Example:"
  echo "  export OPENPI_REPO=/home/yiyanpeng/work/openpi"
  exit 1
fi

if [[ ! -d "${OPENPI_REPO}" ]]; then
  echo "OPENPI_REPO does not exist: ${OPENPI_REPO}"
  exit 1
fi

mkdir -p "$(dirname "${PI05_BASE_DIR}")"

download_pi05_base() {
  if [[ -f "${PI05_BASE_DIR}/model.safetensors" ]]; then
    echo "Pi0.5 PyTorch checkpoint already exists: ${PI05_BASE_DIR}/model.safetensors"
    return
  fi

  if [[ -d "${PI05_BASE_DIR}/params" ]]; then
    echo "Pi0.5 JAX checkpoint already exists: ${PI05_BASE_DIR}/params"
    return
  fi

  if command -v gsutil >/dev/null 2>&1; then
    mkdir -p "$(dirname "${PI05_BASE_DIR}")"
    pushd "$(dirname "${PI05_BASE_DIR}")" >/dev/null
    gsutil -m cp -r "${PI05_GCS_URI}" .
    popd >/dev/null
    return
  fi

  if command -v gcloud >/dev/null 2>&1; then
    mkdir -p "${PI05_BASE_DIR}"
    gcloud storage cp --recursive "${PI05_GCS_URI}" "$(dirname "${PI05_BASE_DIR}")"
    return
  fi

  echo "Neither gsutil nor gcloud is available. Install one of them to download ${PI05_GCS_URI}."
  exit 1
}

patch_transformers() {
  local transformers_dir
  transformers_dir="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
import pathlib
spec = importlib.util.find_spec("transformers")
if spec is None or spec.origin is None:
    raise SystemExit(1)
print(pathlib.Path(spec.origin).resolve().parent)
PY
)"

  if [[ -z "${transformers_dir}" ]]; then
    echo "Could not locate the transformers package with ${PYTHON_BIN}."
    exit 1
  fi

  cp -r "${OPENPI_REPO}/src/openpi/models_pytorch/transformers_replace/"* "${transformers_dir}/"
}

convert_checkpoint() {
  local tmp_output_dir
  if [[ -f "${PI05_BASE_DIR}/model.safetensors" ]]; then
    echo "Skipping conversion: ${PI05_BASE_DIR}/model.safetensors already exists."
    return
  fi

  if [[ ! -d "${PI05_BASE_DIR}/params" ]]; then
    echo "Missing JAX checkpoint params directory: ${PI05_BASE_DIR}/params"
    exit 1
  fi

  tmp_output_dir="$(mktemp -d "${PI05_BASE_DIR}.convert.XXXXXX")"
  pushd "${OPENPI_REPO}" >/dev/null
  "${PYTHON_BIN}" examples/convert_jax_model_to_pytorch.py \
    --config_name pi05_libero \
    --checkpoint_dir "${PI05_BASE_DIR}" \
    --output_path "${tmp_output_dir}"
  popd >/dev/null

  if [[ ! -f "${tmp_output_dir}/model.safetensors" ]]; then
    echo "Conversion did not produce ${tmp_output_dir}/model.safetensors"
    rm -rf "${tmp_output_dir}"
    exit 1
  fi

  mv "${tmp_output_dir}/model.safetensors" "${PI05_BASE_DIR}/model.safetensors"
  if [[ -f "${tmp_output_dir}/config.json" ]]; then
    mv "${tmp_output_dir}/config.json" "${PI05_BASE_DIR}/config.json"
  fi
  rm -rf "${tmp_output_dir}"
}

install_norm_stats() {
  if [[ -z "${NORM_STATS_SRC}" ]]; then
    echo "NORM_STATS_SRC not provided. Skipping norm_stats.json copy."
    return
  fi

  if [[ ! -f "${NORM_STATS_SRC}" ]]; then
    echo "NORM_STATS_SRC does not exist: ${NORM_STATS_SRC}"
    exit 1
  fi

  mkdir -p "${PI05_BASE_DIR}/assets/${NORM_STATS_ASSET_ID}"
  cp "${NORM_STATS_SRC}" "${PI05_BASE_DIR}/assets/${NORM_STATS_ASSET_ID}/norm_stats.json"
}

check_python_stack() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
required = ["torch", "transformers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages for conversion: {missing}")
PY
}

download_pi05_base
check_python_stack
patch_transformers
convert_checkpoint
install_norm_stats

echo "Pi0.5 local bootstrap complete."
echo "Set SENTINEL_PI05_BASE=${PI05_BASE_DIR} before running RLinf Sentinel eval."
