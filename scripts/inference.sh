#!/bin/bash

python inference.py \
    --finetune_model_path "/scratch/limuyao/checkpoints/Qwen2.5_0.5B_Instruct-bias_finetune-q_proj-12_18_01-c4-e1-b16-a4/checkpoint-928" \
    --device 6 \
    --ranks_per_gpu 64 \
    --target_module "[q_proj]"