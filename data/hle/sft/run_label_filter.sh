#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=filter_by_label
#SBATCH --partition=P02
#SBATCH --time=5:00:00 # 実行に時間がかかる可能性を考慮して設定
#SBATCH --output=/home/Competition2025/P02/P02U007/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/P02U007/logs/%x-%j.err

# bash /home/Competition2025/P02/shareP02/scripts/scancel.sh <job_id>
# scp -r comp:/home/Competition2025/P02/P02U007/logs/filter-281969.out ~/Desktop
# Activate the correct conda environment
# Load CUDA and activate environment
module load cuda/12.4
source /home/Competition2025/P02/P02U007/llm2025compet/data/hle/sft/hfenv/bin/activate

# Set environment variables for better GPU memory management
# export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_USE_FLASH_ATTENTION=1
# export VLLM_HIDDEN_SIZE=4096  # Adjust based on model architecture
export HF_HOME="/home/Competition2025/P02/P02U007/.cache/huggingface"

# pip install torch transformers datasets huggingface-hub tqdm
# pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124

# Check GPU status
nvidia-smi

cd /home/Competition2025/P02/P02U007/llm2025compet/data/hle/sft

# # Run with reduced batch size and optimized memory settings
# python OpenMathReasoningFiltering.py \
#     --inference-model Qwen/Qwen3-32B \
#     --judgement-model Qwen/Qwen3-32B \
#     --inference-batch-size 1 \
#     --judgement-batch-size 1 \
#     --start-from-percentage 0.0 \
#     --end-at-percentage 0.5

# Commented out parallel runs
python OpenMathReasoningFiltering_bylabel.py --filter-by-pass-rate 0.1 --save-per-iteration 100000 --batch-size 1
echo "All processes completed!"
