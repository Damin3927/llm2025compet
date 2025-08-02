#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=my_tutorial_job
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

export VLLM_ATTENTION_BACKEND=FLASHINFER  # Use FlashInfer for better attention performance

python3 run_inference.py
