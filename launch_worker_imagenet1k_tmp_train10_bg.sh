#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${LOG:-/tmp/qvit_worker_imnet1k_train10.log}"
PID_FILE="${PID_FILE:-/tmp/qvit_worker_imnet1k_train10.pid}"

rm -f "${LOG}"

cd "${REPO_DIR}"

nohup bash ./run_worker_imagenet1k_tmp_train10.sh > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "launched_pid=$(cat "${PID_FILE}")"
echo "log=${LOG}"
echo "pid_file=${PID_FILE}"
