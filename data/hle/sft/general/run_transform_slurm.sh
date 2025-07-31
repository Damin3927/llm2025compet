#!/bin/bash
#SBATCH --job-name=strategyqa_transform
#SBATCH --output=logs/strategyqa_transform_%j.out
#SBATCH --error=logs/strategyqa_transform_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster setup)
module load python/3.9
module load cuda/11.8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/qian.niu/explore/data/hle/sft/general"

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Run the transformation script
echo "Starting StrategyQA transformation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python transform_strategyqa.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output "strategyqa_transformed_$(date +%Y%m%d_%H%M%S).json" \
    --batch-size 16 \
    --max-tokens 256 \
    --temperature 0.1 \
    --tensor-parallel-size 1

echo "StrategyQA transformation completed at $(date)"
echo "Exit code: $?"