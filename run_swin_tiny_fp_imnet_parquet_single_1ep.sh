#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
DATA_PATH="${1:-/mlx_devbox/users/quyanyi/playground/QAT/imagenet1k_40g_parquet}"
OUT_DIR="${2:-/mlx_devbox/users/quyanyi/playground/QAT/out/swin_tiny_fp_single_1ep}"

mkdir -p "${OUT_DIR}"

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 nohup python3 -u main.py \
  --model swin_tiny_patch4_window7_224 \
  --pretrained \
  --data-set IMNET_PARQUET \
  --data-path "${DATA_PATH}" \
  --epochs 1 \
  --warmup-epochs 0 \
  --weight-decay 0.05 \
  --drop-path 0.2 \
  --batch-size 64 \
  --num_workers 16 \
  --output_dir "${OUT_DIR}" \
  > "${OUT_DIR}/console.log" 2>&1 &

echo $! > "${OUT_DIR}/pid.txt"
echo "launched_pid=$(cat "${OUT_DIR}/pid.txt")"
echo "out_dir=${OUT_DIR}"
