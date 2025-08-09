#!/bin/bash
#SBATCH --job-name=medcalc_convert
#SBATCH --output=medcalc_convert_%j.out
#SBATCH --error=medcalc_convert_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Load necessary modules
module load cuda/12.1
module load python/3.9

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/tmp/huggingface_cache_$SLURM_JOB_ID
export TRANSFORMERS_CACHE=/tmp/transformers_cache_$SLURM_JOB_ID
export HF_DATASETS_CACHE=/tmp/datasets_cache_$SLURM_JOB_ID

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

# Navigate to script directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install vllm torch transformers datasets tqdm accelerate

echo "========================================"
echo "Starting medcalc conversion"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"
echo "Start time: $(date)"
echo "========================================"

# Check GPU availability
nvidia-smi

# Run the conversion script with LLM transformation
python convert_medcalc.py --use-llm --model "Qwen/Qwen2.5-32B-Instruct"

# Check exit status
EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output saved to: ~/explore/data/hle/sft/medical/results/medcalc_cot.json"
echo "Timing data saved to: ~/explore/data/hle/sft/medical/results/medcalc_timing.json"
echo "Validation report saved to: ~/explore/data/hle/sft/medical/results/medcalc_validation_report.json"
echo "========================================"

# Clean up cache directories
rm -rf $HF_HOME
rm -rf $TRANSFORMERS_CACHE 
rm -rf $HF_DATASETS_CACHE

exit $EXIT_CODE