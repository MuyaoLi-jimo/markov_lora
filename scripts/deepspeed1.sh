#!/bin/bash

card_number=1
cuda_visible_devices=4
training_port=20015

deepspeed  --include=localhost:$cuda_visible_devices --master_port=$training_port deepspeed_train.py \
    --deepspeed --deepspeed_config "configs/deepspeed_config_s2.json" \
    --model_path /scratch/mc_lmy/models/Qwen2.5-0.5B-Instruct \
    --dataset_name /scratch/mc_lmy/datas/MetaMathQA \
    --output_path /scratch/mc_lmy/models/JARVIS/checkpoints \
    --training_mode 'lora_finetune' \
    --ranks_per_gpu 64  \
    --batch_size 32  \
    --device_num $cuda_visible_devices \
    --accumulation_steps  4  \
    --num_epochs  1 

