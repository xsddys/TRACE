#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_7b_multiturn.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_dpo_trainer \
    data.train_files=$HOME/data/multiturn_dpo/train.parquet \
    data.val_files=$HOME/data/multiturn_dpo/test.parquet \
    data.messages_key=messages \
    data.chosen_key=chosen \
    data.rejected_key=rejected \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.max_length=5120 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=multiturn-dpo \
    trainer.experiment_name=qwen2.5-7b-instruct-multiturn-dpo \
    trainer.total_epochs=2 \
    trainer.logger=['console'] \
    trainer.default_hdfs_dir=null $@
