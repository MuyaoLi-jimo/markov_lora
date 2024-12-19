#!/bin/bash

batch=16
gradient_accumulation_steps=4
logging_step=1
epoch=1
learning_rate=1.4e-5
card_number=4
cuda_visible_devices=0,1,2,3
max_seq_length=512
training_port=20015
version="Qwen2.5_0.5B_Instruct-bias_finetune-q_proj-12_18_01"
WANDB_NAME="$version-c$card_number-e$epoch-b$batch-a$gradient_accumulation_steps"
export WANDB_MODE="offline"
export WANDB_API_KEY=998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d
export WANDB_PROJECT=BIAS_MODEL
export WANDB_NOTES="[24-12-18]TRAIN A BIAS MODEL"

deepspeed  --include=localhost:$cuda_visible_devices --master_port=$training_port deepspeed_train2.py \
    --deepspeed "configs/deepspeed_config_s2.v2.json" \
    --model_name_or_path /scratch/limuyao/checkpoints/Qwen2.5-0.5B-Instruct \
    --dataset_name /scratch/limuyao/datas/MetaMathQA \
    --dataloader_num_workers 16 \
    --dataloader_pin_memory True \
    --output_dir "/scratch/limuyao/checkpoints/$WANDB_NAME" \
    --ranks_per_gpu 64  \
    --run_name $WANDB_NAME \
    --report_to "wandb" \
    --logging_strategy "steps" \
    --logging_steps $logging_step \
    --num_train_epochs $epoch \
    --gradient_checkpointing \
    --torch_dtype float32 \
    --max_seq_length $max_seq_length \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size $batch \
    --per_device_eval_batch_size $batch \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "steps" \
    --eval_steps 200 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 20