#! /bin/bash
  
#SBATCH -J wyd
#SBATCH -p IAI_SLURM_HGX
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -c 16
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u inference.py

