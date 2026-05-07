PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

model_path="./models/Qwen2.5-3B-Instruct"

# Ensure correct GPU visibility for this node
export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_ROOT}/wandb"

export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED=1
judger_llm_tokenizer_path="./models/HarmBench-Llama-2-13b-cls"


trainer_project_name="trace_project"
experiment_name="trace_experiment"
timestamp=$(date +"%Y%m%d_%H%M%S")
experiment_name="${experiment_name}__${timestamp}"

export PYTHONPATH="${PROJECT_ROOT}/verl:${PROJECT_ROOT}:${PYTHONPATH}"

CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${trainer_project_name}/${experiment_name}"
TRAIN_ROLLOUT_DIR="${PROJECT_ROOT}/matric-rollout/Training/${experiment_name}/train_rollout"
VAL_ROLLOUT_DIR="${PROJECT_ROOT}/matric-rollout/Training/${experiment_name}/val_rollout"
TENSORBOARD_DIR="${PROJECT_ROOT}/tensorboard_log/${experiment_name}"
RUN_LOG_FILE="${PROJECT_ROOT}/nohup_logs/run_logs/training/${experiment_name}.log"

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${TRAIN_ROLLOUT_DIR}"
mkdir -p "${VAL_ROLLOUT_DIR}"
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "${PROJECT_ROOT}/nohup_logs/run_logs/training"
python train.py --config-name _7_jailbreak.yaml \
  model_path=$model_path \
  judger_llm.tokenizer_path=$judger_llm_tokenizer_path judger_llm.model_path=$judger_llm_tokenizer_path \
  algorithm.heuristic_process_adv_lambda=0.1 \
  experiment_name=${experiment_name} trainer.project_name=${trainer_project_name} trainer.experiment_name=${experiment_name} \
  trainer.default_local_dir="${CHECKPOINT_DIR}" trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
  trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" trainer.tensorboard_dir="${TENSORBOARD_DIR}" \
  trainer.total_training_steps=130 trainer.test_freq=10 trainer.save_freq=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
  > "${RUN_LOG_FILE}" 2>&1

echo "Training ${experiment_name} finished."
