#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_model.sh [CONFIG_PATH]
#
# Optional env vars:
#   METHOD=MT|UAMT
#   USE_SAM=0|1
#   USE_FN=0|1
#   SAM_CKPT=/abs/path/to/sam_checkpoint.pth
#   SAM_MODEL=vit_b|vit_l|vit_h
#   SAM_BACKEND=sam|medsam
#   SAM_AUTO_DOWNLOAD=0|1
#   SAM_URL=https://...
#   RESUME=0|1
#   GPU_ID=0                      # physical GPU id to expose
#   CUDA_VISIBLE_DEVICES=0        # optional explicit override

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${1:-configs/train_config_3d.yaml}"
METHOD="${METHOD:-MT}"
USE_SAM="${USE_SAM:-0}"
USE_FN="${USE_FN:-0}"
SAM_CKPT="${SAM_CKPT:-./pretrained/sam_vit_h_4b8939.pth}"
SAM_MODEL="${SAM_MODEL:-vit_h}"
SAM_BACKEND="${SAM_BACKEND:-sam}"
SAM_AUTO_DOWNLOAD="${SAM_AUTO_DOWNLOAD:-0}"
SAM_URL="${SAM_URL:-}"
RESUME="${RESUME:-0}"
GPU_ID="${GPU_ID:-0}"

if [[ "${SAM_BACKEND}" == "medsam" ]]; then
  if [[ "${SAM_MODEL}" != "vit_b" ]]; then
    SAM_MODEL="vit_b"
  fi
  if [[ "${SAM_CKPT}" == "./pretrained/sam_vit_h_4b8939.pth" ]]; then
    SAM_CKPT="./pretrained/medsam_vit_b.pth"
  fi
fi

# Keep only one visible GPU to avoid CUDA init scanning bad/blocked devices.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

mkdir -p config/generated
RUN_TAG="$(date +%Y-%m-%d=%H-%M-%S)_${METHOD}_sam${USE_SAM}_fn${USE_FN}"
TMP_CONFIG="config/generated/${RUN_TAG}_train_config.yaml"

python - <<PY
import yaml

cfg_path = r"${CONFIG_PATH}"
out_path = r"${TMP_CONFIG}"

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["method"] = r"${METHOD}"
# With CUDA_VISIBLE_DEVICES set, trainer should always use local index 0.
cfg["gpu"] = "0"

sam_cfg = cfg.get("sam", {}) or {}
sam_cfg["use_sam_refinement"] = bool(int(r"${USE_SAM}"))
sam_cfg["use_fn_recovery"] = bool(int(r"${USE_FN}"))
sam_cfg["sam_checkpoint"] = r"${SAM_CKPT}"
sam_cfg["sam_model_type"] = r"${SAM_MODEL}"
sam_cfg["sam_backend"] = r"${SAM_BACKEND}"
sam_cfg["auto_download"] = bool(int(r"${SAM_AUTO_DOWNLOAD}"))
sam_cfg["download_url"] = r"${SAM_URL}"
cfg["sam"] = sam_cfg

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "Project root: ${PROJECT_ROOT}"
echo "Base config : ${CONFIG_PATH}"
echo "Run config  : ${TMP_CONFIG}"
echo "Method      : ${METHOD}"
echo "Use SAM     : ${USE_SAM}"
echo "Use FN rec. : ${USE_FN}"
echo "SAM backend : ${SAM_BACKEND}"
echo "SAM download: ${SAM_AUTO_DOWNLOAD}"
echo "GPU (phys)  : ${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Start time  : $(date)"

if [[ "${RESUME}" == "1" ]]; then
  python train_semi-supervised.py --config "${TMP_CONFIG}" --c
else
  python train_semi-supervised.py --config "${TMP_CONFIG}"
fi

echo "End time    : $(date)"
