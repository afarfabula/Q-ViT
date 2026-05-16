#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

HF_ENDPOINT="${HF_ENDPOINT:-http://huggingface-proxy-sg.byted.org}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HOME="${HF_HOME:-/tmp/huggingface}"

IMNET_DIR="${IMNET_DIR:-/tmp/imagenet1k_full_parquet}"
OUT_DIR="${OUT_DIR:-/tmp/qvit_out_imnet1k_full_train10}"

MODEL="${MODEL:-fourbits_deit_small_patch16_224}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10}"
MAX_EVAL_STEPS="${MAX_EVAL_STEPS:-2}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export HF_ENDPOINT HF_HUB_DISABLE_XET HF_HOME

mkdir -p "${HF_HOME}" "${IMNET_DIR}" "${OUT_DIR}"

echo "repo_dir=${REPO_DIR}"
echo "imagenet_dir=${IMNET_DIR}"
echo "out_dir=${OUT_DIR}"
echo "hf_endpoint=${HF_ENDPOINT}"

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "missing_token=1"
  echo "set env: HF_TOKEN or HUGGINGFACE_HUB_TOKEN"
  exit 2
fi

echo "python=$(${PYTHON_BIN} -V 2>&1)"

${PYTHON_BIN} - <<'PY'
import importlib, sys
mods = ["torch"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("missing_python_modules=" + ",".join(missing))
    sys.exit(3)
print("torch_ok=1")
PY

maybe_install() {
  pkg="$1"
  ${PYTHON_BIN} - <<PY || (${PYTHON_BIN} -m pip install --user -U "$pkg")
import importlib
importlib.import_module("$2")
print("ok")
PY
}

maybe_install "huggingface_hub[cli]" "huggingface_hub"
maybe_install "pyarrow" "pyarrow"
maybe_install "Pillow" "PIL"
maybe_install "timm==0.4.12" "timm"

echo "download_start=1"
huggingface-cli download ILSVRC/imagenet-1k \
  --repo-type dataset \
  --local-dir "${IMNET_DIR}" \
  --local-dir-use-symlinks False \
  --resume-download
echo "download_done=1"

if [ ! -d "${IMNET_DIR}/data" ]; then
  echo "missing_data_dir=${IMNET_DIR}/data"
  exit 4
fi

train_files_count="$(find "${IMNET_DIR}/data" -maxdepth 1 -type f -name 'train-*.parquet' | wc -l | tr -d ' ')"
val_files_count="$(find "${IMNET_DIR}/data" -maxdepth 1 -type f -name 'validation-*.parquet' | wc -l | tr -d ' ')"
echo "train_parquet_files=${train_files_count}"
echo "validation_parquet_files=${val_files_count}"

cd "${REPO_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 ${PYTHON_BIN} -u main.py \
  --model "${MODEL}" \
  --data-set IMNET_PARQUET \
  --data-path "${IMNET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --epochs 1 \
  --warmup-epochs 0 \
  --weight-decay 0.0 \
  --batch-size "${BATCH_SIZE}" \
  --batch-size-eval "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max-train-steps "${MAX_TRAIN_STEPS}" \
  --max-eval-steps "${MAX_EVAL_STEPS}"
