#!/bin/bash
#SBATCH --job-name=chempile_qa_extraction
#SBATCH --output=chempile_extraction_%j.out
#SBATCH --error=chempile_extraction_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a10g:1
#SBATCH --partition=gpu

# Load necessary modules (adjust based on your cluster)
module load python/3.9
module load cuda/11.8

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm
pip install datasets
pip install transformers
pip install accelerate

# Set memory and performance optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.6"  # For A10G GPU
export VLLM_USE_MODELSCOPE=False

# Run the extraction script
echo "Starting ChemPile QA extraction..."
echo "Job started at: $(date)"

python extract_chempile_qa.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --output "chempile_qa_pairs.json" \
    --validate

echo "Job completed at: $(date)"

# Print some statistics
if [ -f "chempile_qa_pairs.json" ]; then
    echo "Output file size: $(du -h chempile_qa_pairs.json)"
    echo "Number of records: $(python -c "import json; data=json.load(open('chempile_qa_pairs.json')); print(len(data))")"
fi

# Deactivate virtual environment
deactivate

echo "ChemPile QA extraction job completed!"