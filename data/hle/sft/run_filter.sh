#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=filter
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # GPUが必要な場合
#SBATCH --time=01:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=/home/Competition2025/P02/P02U007/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/P02U007/logs/%x-%j.err

# bash /home/Competition2025/P02/shareP02/scripts/scancel.sh <job_id>
# scp -r comp:/home/Competition2025/P02/P02U007/logs/filter-281969.* ~/Desktop
# Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate base
pip install torch transformers datasets huggingface-hub tqdm
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

CUDA_VISIBLE_DEVICES=0 python OpenMathReasoningFiltering.py --start-from-percentage 0.0 --end-at-percentage 0.5
# CUDA_VISIBLE_DEVICES=1 python OpenMathReasoningFiltering.py --start-from-percentage 0.5 --end-at-percentage 1.0
# CUDA_VISIBLE_DEVICES=2 python OpenMathReasoningFiltering.py --run-index 2 --start-from-percentage 0.0 --end-at-percentage 0.5
# CUDA_VISIBLE_DEVICES=3 python OpenMathReasoningFiltering.py --run-index 2 --start-from-percentage 0.5 --end-at-percentage 1.0
# CUDA_VISIBLE_DEVICES=4 python OpenMathReasoningFiltering.py --run-index 3 --start-from-percentage 0.0 --end-at-percentage 0.5
# CUDA_VISIBLE_DEVICES=5 python OpenMathReasoningFiltering.py --run-index 3 --start-from-percentage 0.5 --end-at-percentage 1.0
# CUDA_VISIBLE_DEVICES=6 python OpenMathReasoningFiltering.py --run-index 4 --start-from-percentage 0.0 --end-at-percentage 0.5
# CUDA_VISIBLE_DEVICES=7 python OpenMathReasoningFiltering.py --run-index 4 --start-from-percentage 0.5 --end-at-percentage 1.0
