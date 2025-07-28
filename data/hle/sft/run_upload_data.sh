#!/bin/bash
#SBATCH --job-name=upload_data
#SBATCH --output=logs/upload_data_%j.out
#SBATCH --error=logs/upload_data_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

# Configuration - Edit these variables as needed
DATASET_PATH="./results/selected_data/OpenMathReasoning"  # Path to folder containing data subfolders
REPO_ID="neko-llm/SFT_OpenMathReasoninge"  # HuggingFace repository ID
CREATE_DATASET_CARD=true  # Set to false to skip dataset card creation

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/upload_data.py"

# Create necessary directories
mkdir -p logs

# Detect if running under SLURM or locally
if [ -n "$SLURM_JOB_ID" ]; then
    RUNNING_MODE="SLURM"
    JOB_INFO="Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
else
    RUNNING_MODE="Local"
    JOB_INFO="Running locally on $(hostname)"
fi

echo "=========================================="
echo "Upload Data Job ($RUNNING_MODE)"
echo "=========================================="
echo "$JOB_INFO"
echo "Started at: $(date)"
echo "Dataset path: $DATASET_PATH"
echo "Repository ID: $REPO_ID"
echo "Create dataset card: $CREATE_DATASET_CARD"
echo "=========================================="

# Validate dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset path does not exist: $DATASET_PATH"
    echo "Please ensure you have run the data selection/generation scripts first."
    exit 1
fi

# Check if dataset path contains data
if [ -z "$(ls -A "$DATASET_PATH" 2>/dev/null)" ]; then
    echo "❌ ERROR: Dataset path is empty: $DATASET_PATH"
    echo "Please ensure there are data folders in the dataset path."
    exit 1
fi

# Show dataset structure
echo "Dataset structure:"
find "$DATASET_PATH" -type f -name "*.json" | head -10 | while read file; do
    echo "  $file"
done
if [ $(find "$DATASET_PATH" -type f -name "*.json" | wc -l) -gt 10 ]; then
    echo "  ... and $(expr $(find "$DATASET_PATH" -type f -name "*.json" | wc -l) - 10) more files"
fi

# Activate conda environment (adjust as needed)
# source ~/.bashrc
# conda activate hf  # Replace with your environment name

# Or use module system if available
# module load python/3.9

# Build command
CMD="python ${SCRIPT_PATH}"
CMD="${CMD} --dataset_path ${DATASET_PATH}"
CMD="${CMD} --repo_id ${REPO_ID}"

if [ "$CREATE_DATASET_CARD" = true ]; then
    CMD="${CMD} --create_dataset_card"
fi

echo "Command to execute:"
echo "$CMD"
echo "=========================================="

# Execute the command
eval $CMD

# Check exit status
EXIT_CODE=$?
echo "=========================================="
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Data upload completed successfully"
    echo "Repository URL: https://huggingface.co/datasets/$REPO_ID"
    
    # Show upload summary if logs are available
    echo ""
    echo "Upload summary:"
    echo "- Dataset path: $DATASET_PATH"
    echo "- Repository: $REPO_ID"
    echo "- Dataset card created: $CREATE_DATASET_CARD"
else
    echo "❌ ERROR: Data upload failed with exit code $EXIT_CODE"
    echo ""
    echo "Common issues and solutions:"
    echo "1. Authentication: Make sure you're logged in to HuggingFace"
    echo "   Run: huggingface-cli login"
    echo "2. Repository: Ensure the repository exists and you have write access"
    echo "3. Data format: Check that JSON files are valid and properly structured"
    echo "4. Network: Verify internet connection and HuggingFace Hub access"
fi

echo "=========================================="