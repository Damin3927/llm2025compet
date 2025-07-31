#!/bin/bash

# --- Slurm ジョブ設定 ---
#SBATCH --job-name=filter
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu54
#SBATCH --gres=gpu:8 # GPUが必要な場合
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

cd /home/Competition2025/P02/P02U007/llm2025compet/data/hle/sft/chemistry/chempile
python extract_chempile_qa.py \
    --model "Qwen/Qwen3-32B" \
    --output "chempile_qa_pairs.json" \
    --validate \
    --tp 2

echo "Job completed at: $(date)"

# Print some statistics
if [ -f "chempile_qa_pairs.json" ]; then
    echo "Output file size: $(du -h chempile_qa_pairs.json)"
    echo "Number of records: $(python -c "import json; data=json.load(open('chempile_qa_pairs.json')); print(len(data))")"
fi

# Deactivate virtual environment
deactivate

echo "ChemPile QA extraction job completed!"