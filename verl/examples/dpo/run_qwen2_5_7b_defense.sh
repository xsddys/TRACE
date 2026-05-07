#!/bin/bash
set -euo pipefail
set -x

if [ "$#" -lt 4 ]; then
    echo "Usage: run_qwen2_5_7b_defense.sh <nproc_per_node> <dataset_dir> <save_path> <train_mode:lora|full> [extra_overrides...]"
    exit 1
fi

nproc_per_node=$1
dataset_dir=$2
save_path=$3
train_mode=$4
shift 4

train_file="${dataset_dir}/train.jsonl"
val_file="${dataset_dir}/val.jsonl"
log_dir="${DEFENSE_TRAIN_LOG_DIR:-${save_path}/torchrun_logs}"
repo_root="./verl"
helpful_sft_train_file="${HELPFUL_SFT_TRAIN_FILE:-}"
helpful_sft_val_file="${HELPFUL_SFT_VAL_FILE:-}"
helpful_sft_dataset_cls="${HELPFUL_SFT_DATASET_CLS:-defense_sft}"
helpful_sft_train_batch_size="${HELPFUL_SFT_BATCH_SIZE:-64}"
helpful_sft_micro_batch_size="${HELPFUL_SFT_MICRO_BATCH_SIZE:-2}"
helpful_sft_prompt_key="${HELPFUL_SFT_PROMPT_KEY:-prompt}"
helpful_sft_response_key="${HELPFUL_SFT_RESPONSE_KEY:-response}"
helpful_sft_prompt_text_key="${HELPFUL_SFT_PROMPT_TEXT_KEY:-prompt_text}"
helpful_sft_response_text_key="${HELPFUL_SFT_RESPONSE_TEXT_KEY:-response_text}"
joint_sft_lambda="${JOINT_SFT_LAMBDA:-0.3}"
mkdir -p "$log_dir"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
cd "$repo_root"

python - <<'PY'
import importlib

for module_name in ["ray", "hydra", "omegaconf", "verl.trainer.fsdp_dpo_trainer"]:
    importlib.import_module(module_name)
print("Preflight import check passed.")
PY


common_overrides=(
  data.dataset_cls=defense_dpo
  data.train_files="$train_file"
  data.val_files="$val_file"
  data.prompt_key=prompt
  data.chosen_key=chosen
  data.rejected_key=rejected
  data.train_batch_size=128
  data.max_prompt_length=4096
  data.max_response_length=1024
  data.max_length=5120
  data.num_workers=0
  data.val_num_workers=0
  data.prompt_truncation=left
  data.response_truncation=right
  data.truncation=truncate
  model.partial_pretrain=./models/Qwen2.5-7B-Instruct
  model.enable_gradient_checkpointing=true
  trainer.default_local_dir="$save_path"
  trainer.project_name=multiturn-defense-dpo
  trainer.total_epochs=1
  trainer.evals_per_epoch=10
  trainer.default_hdfs_dir=null
  dpo.beta=0.3
  dpo.loss_type=sigmoid
  dpo.label_smoothing=0.0
  dpo.reference_free=false
  dpo.average_log_prob=true
  optim.weight_decay=0.01
  optim.warmup_steps_ratio=0.03
  optim.max_grad_norm=1.0
)

if [ -n "$helpful_sft_train_file" ] || [ -n "$helpful_sft_val_file" ]; then
  if [ -z "$helpful_sft_train_file" ] || [ -z "$helpful_sft_val_file" ]; then
    echo "Both HELPFUL_SFT_TRAIN_FILE and HELPFUL_SFT_VAL_FILE must be set." >&2
    exit 1
  fi
  common_overrides+=(
    helpful_data.enabled=true
    helpful_data.dataset_cls="$helpful_sft_dataset_cls"
    helpful_data.train_files="$helpful_sft_train_file"
    helpful_data.val_files="$helpful_sft_val_file"
    helpful_data.train_batch_size="$helpful_sft_train_batch_size"
    helpful_data.micro_batch_size_per_gpu="$helpful_sft_micro_batch_size"
    helpful_data.prompt_key="$helpful_sft_prompt_key"
    helpful_data.response_key="$helpful_sft_response_key"
    helpful_data.prompt_text_key="$helpful_sft_prompt_text_key"
    helpful_data.response_text_key="$helpful_sft_response_text_key"
    joint.sft_lambda="$joint_sft_lambda"
  )
fi

if [ "$train_mode" = "lora" ]; then
  mode_overrides=(
    data.micro_batch_size_per_gpu=2
    model.lora_rank=64
    model.lora_alpha=128
    model.lora_dropout=0.03
    model.target_modules=all-linear
    optim.lr=5e-6
    trainer.experiment_name=qwen2.5-7b-defense-dpo-lora
  )
elif [ "$train_mode" = "full" ]; then
  mode_overrides=(
    data.micro_batch_size_per_gpu=1
    model.lora_rank=0
    model.lora_alpha=16
    model.lora_dropout=0.0
    model.target_modules=all-linear
    optim.lr=7e-7
    trainer.experiment_name=qwen2.5-7b-defense-dpo-full
  )
else
  echo "Unknown train_mode: $train_mode"
  exit 1
fi

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
  --log-dir "$log_dir" \
  --redirects 3 \
  --tee 3 \
  -m verl.trainer.fsdp_dpo_trainer --config-name dpo_defense_trainer \
  "${common_overrides[@]}" \
  "${mode_overrides[@]}" \
  "$@"
