#!/bin/bash

python inference.py \
    --finetune_model_path "/scratch/limuyao/checkpoints/Qwen2.5_0.5B_Instruct-normal-12_19_01-c4-e1-b32-a4/checkpoint-464" \
    --device 7 \
    --ranks_per_gpu 64 \
    --target_module "[]"