#! /bin/bash

#SBATCH -J longhorizon
#SBATCH -p IAI_SLURM_HGX
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -c 16
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u bias_tuning.py \
    --model_path  "/ceph/home/muhan01/huggingfacemodels/Qwen2.5-1.5B-Instruct" \
    --output_path "./outputs/Qwen2.5-1.5B-4gpu-16ranks-GSM240k-1epochs-lora"  \
    --world_size 4  \
    --training_mode 'lora_finetune' \
    --ranks_per_gpu 16  \
    --batch_size  1  \
    --accumulation_steps  128  \
    --num_epochs  1
