#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=filter
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu54
#SBATCH --gres=gpu:1 # GPUが必要な場合
#SBATCH --time=5:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=/home/Competition2025/P02/P02U007/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/P02U007/logs/%x-%j.err

# bash /home/Competition2025/P02/shareP02/scripts/scancel.sh <job_id>
# scp -r comp:/home/Competition2025/P02/P02U007/logs/filter-281969.out ~/Desktop
# Activate the correct conda environment
module load cuda/12.4
source /home/Competition2025/P02/P02U007/llm2025compet/data/hle/sft/hfenv/bin/activate

# pip install torch transformers datasets huggingface-hub tqdm
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

nvidia-smi

cd /home/Competition2025/P02/P02U007/llm2025compet/data/hle/sft

CUDA_VISIBLE_DEVICES=0 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --start-from-percentage 0.0 --end-at-percentage 0.5
CUDA_VISIBLE_DEVICES=1 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --start-from-percentage 0.5 --end-at-percentage 1.0
CUDA_VISIBLE_DEVICES=2 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 2 --start-from-percentage 0.0 --end-at-percentage 0.5
CUDA_VISIBLE_DEVICES=3 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 2 --start-from-percentage 0.5 --end-at-percentage 1.0
CUDA_VISIBLE_DEVICES=4 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 3 --start-from-percentage 0.0 --end-at-percentage 0.5
CUDA_VISIBLE_DEVICES=5 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 3 --start-from-percentage 0.5 --end-at-percentage 1.0
CUDA_VISIBLE_DEVICES=6 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 4 --start-from-percentage 0.0 --end-at-percentage 0.5
CUDA_VISIBLE_DEVICES=7 python OpenMathReasoningFiltering.py --inference-model Qwen/Qwen3-32B --judgement-model Qwen/Qwen3-32B --run-index 4 --start-from-percentage 0.5 --end-at-percentage 1.0
