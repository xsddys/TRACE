#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=""
STEPS=(${*:-120})

for step in "${STEPS[@]}"; do
  LOCAL_DIR="${BASE_DIR}/global_step_${step}/actor"
  python ./scripts/model_merge.py \
    --local_dir "${LOCAL_DIR}"
done