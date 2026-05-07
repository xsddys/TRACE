#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/checkpoints/Ablation/Semantic-wo_refulsal-A-q253i-T-l318i-bigger_validation__20260418_194625"
STEPS=(${*:-120})

for step in "${STEPS[@]}"; do
  LOCAL_DIR="${BASE_DIR}/global_step_${step}/actor"
  python /mnt/shared-storage-user/wenxiaoyu/hezhida/TROJail/scripts/model_merge.py \
    --local_dir "${LOCAL_DIR}"
done