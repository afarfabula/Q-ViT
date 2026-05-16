#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-/mlx_devbox/users/quyanyi/playground/QAT/imagenet1k_40g_parquet}"
OUT_DIR="${2:-/mlx_devbox/users/quyanyi/playground/QAT/out/swin_tiny_fp_ddp}"

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  --master_port=29541 \
  main.py \
  --model swin_tiny_patch4_window7_224 \
  --pretrained \
  --data-set IMNET_PARQUET \
  --data-path "${DATA_PATH}" \
  --epochs 300 \
  --warmup-epochs 20 \
  --weight-decay 0.05 \
  --batch-size 64 \
  --num_workers 16 \
  --drop-path 0.2 \
  --output_dir "${OUT_DIR}"
