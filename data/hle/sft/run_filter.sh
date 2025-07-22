#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=filter
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # GPUが必要な場合
#SBATCH --time=01:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hf
pip install transformers datasets huggingface-hub tqdm vllm

python OpenMathReasoningFiltering.py
