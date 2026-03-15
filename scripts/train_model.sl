#!/bin/bash
#SBATCH --job-name=train_ssl3d
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb
#SBATCH --partition=weilab
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liupen@bc.edu
#SBATCH --output=/projects/weilab/liupeng/semi-seg/log/train_ssl3d_%j.out

set -euo pipefail

source ~/miniconda3/bin/activate
conda activate ssl_seg

PROJECT_ROOT="/projects/weilab/liupeng/projects/semi-seg/SSL-Seg"
cd "${PROJECT_ROOT}"

# Optional overrides (pass with: sbatch --export=ALL,VAR=... scripts/train_model.sl)
CONFIG_PATH="${CONFIG_PATH:-configs/train_config_3d.yaml}"
METHOD="${METHOD:-MT}"
USE_SAM="${USE_SAM:-0}"
USE_FN="${USE_FN:-0}"
SAM_CKPT="${SAM_CKPT:-./pretrained/sam_vit_h_4b8939.pth}"
SAM_MODEL="${SAM_MODEL:-vit_h}"
SAM_AUTO_DOWNLOAD="${SAM_AUTO_DOWNLOAD:-0}"
SAM_URL="${SAM_URL:-}"
RESUME="${RESUME:-0}"
GPU_ID="${GPU_ID:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Config      : ${CONFIG_PATH}"
echo "Method      : ${METHOD}"
echo "Use SAM     : ${USE_SAM}"
echo "Use FN rec. : ${USE_FN}"
echo "SAM download: ${SAM_AUTO_DOWNLOAD}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "Start time  : $(date)"

bash scripts/train_model.sh "${CONFIG_PATH}"

echo "End time    : $(date)"
