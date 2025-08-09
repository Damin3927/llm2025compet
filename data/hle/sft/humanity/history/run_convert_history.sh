
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

# Define paths
INPUT_FILE="./history/results/history.json"
OUTPUT_FILE="./history/results/history_reasoning.json"
MODEL_NAME="Qwen/Qwen3-32B"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found!"
    exit 1
fi

echo "Starting history data conversion..."
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Install required packages if not already installed
pip install --user vllm transformers torch

# Run the conversion script
python convert_history.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_NAME"

echo "Conversion completed!"
echo "Output saved to: $OUTPUT_FILE"

# Print some statistics
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file size: $(du -h $OUTPUT_FILE | cut -f1)"
    echo "Number of processed samples: $(python -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))")"
fi