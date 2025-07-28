#!/bin/bash

# Universal upload data script - works both locally and on SLURM
# Usage: ./universal_upload_data.sh [dataset_path] [repo_id] [create_card]
# Example: ./universal_upload_data.sh ./results/my_data user/my-dataset true

# Default values
DEFAULT_DATASET_PATH="./results/selected_data"
DEFAULT_REPO_ID="your-username/your-dataset-name"
DEFAULT_CREATE_CARD="true"

# Parse command line arguments
DATASET_PATH="${1:-$DEFAULT_DATASET_PATH}"
REPO_ID="${2:-$DEFAULT_REPO_ID}"
CREATE_CARD="${3:-$DEFAULT_CREATE_CARD}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/upload_data.py"

# Create necessary directories
mkdir -p logs

# Detect running environment
if [ -n "$SLURM_JOB_ID" ]; then
    RUNNING_MODE="SLURM"
    JOB_INFO="Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
    LOG_FILE="logs/upload_data_${SLURM_JOB_ID}.log"
else
    RUNNING_MODE="Local"
    JOB_INFO="Running locally on $(hostname)"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="logs/upload_data_${TIMESTAMP}.log"
fi

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_with_timestamp "=========================================="
log_with_timestamp "Universal Upload Data Script ($RUNNING_MODE)"
log_with_timestamp "=========================================="
log_with_timestamp "$JOB_INFO"
log_with_timestamp "Script arguments:"
log_with_timestamp "  Dataset path: $DATASET_PATH"
log_with_timestamp "  Repository ID: $REPO_ID"
log_with_timestamp "  Create dataset card: $CREATE_CARD"
log_with_timestamp "  Log file: $LOG_FILE"
log_with_timestamp "=========================================="

# Validation checks
validation_failed=false

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    log_with_timestamp "❌ ERROR: Dataset path does not exist: $DATASET_PATH"
    log_with_timestamp "Available paths in current directory:"
    ls -la | grep "^d" | awk '{print "  " $9}' | tee -a "$LOG_FILE"
    validation_failed=true
fi

# Check if dataset path contains data
if [ -d "$DATASET_PATH" ] && [ -z "$(find "$DATASET_PATH" -name "*.json" 2>/dev/null)" ]; then
    log_with_timestamp "❌ ERROR: No JSON files found in dataset path: $DATASET_PATH"
    log_with_timestamp "Directory contents:"
    ls -la "$DATASET_PATH" | tee -a "$LOG_FILE"
    validation_failed=true
fi

# Check repository ID format
if [[ ! "$REPO_ID" =~ ^[^/]+/[^/]+$ ]]; then
    log_with_timestamp "❌ ERROR: Invalid repository ID format: $REPO_ID"
    log_with_timestamp "Expected format: username/dataset-name"
    validation_failed=true
fi

# Check if upload_data.py exists
if [ ! -f "$SCRIPT_PATH" ]; then
    log_with_timestamp "❌ ERROR: upload_data.py not found at: $SCRIPT_PATH"
    validation_failed=true
fi

if [ "$validation_failed" = true ]; then
    log_with_timestamp "=========================================="
    log_with_timestamp "❌ VALIDATION FAILED - Please fix the above errors"
    log_with_timestamp "=========================================="
    exit 1
fi

# Show dataset structure
log_with_timestamp "Dataset structure preview:"
json_files=$(find "$DATASET_PATH" -type f -name "*.json")
json_count=$(echo "$json_files" | wc -l)

if [ $json_count -gt 0 ]; then
    echo "$json_files" | head -5 | while read file; do
        file_size=$(du -h "$file" | cut -f1)
        log_with_timestamp "  $file ($file_size)"
    done
    
    if [ $json_count -gt 5 ]; then
        log_with_timestamp "  ... and $(expr $json_count - 5) more JSON files"
    fi
    
    log_with_timestamp "Total JSON files: $json_count"
else
    log_with_timestamp "  No JSON files found"
fi

# Show folder structure
log_with_timestamp ""
log_with_timestamp "Folder structure:"
find "$DATASET_PATH" -type d | while read dir; do
    file_count=$(find "$dir" -maxdepth 1 -name "*.json" | wc -l)
    if [ $file_count -gt 0 ]; then
        log_with_timestamp "  $dir/ ($file_count JSON files)"
    else
        log_with_timestamp "  $dir/ (no JSON files)"
    fi
done

# Environment setup
log_with_timestamp ""
log_with_timestamp "Environment setup..."

# Activate conda environment (uncomment and modify as needed)
# source ~/.bashrc
# conda activate hf  # Replace with your environment name
# log_with_timestamp "Activated conda environment: hf"

# Or use module system if available
# module load python/3.9
# log_with_timestamp "Loaded Python module"

# Check Python and required packages
python_version=$(python --version 2>&1)
log_with_timestamp "Python version: $python_version"

# Build and execute command
log_with_timestamp ""
log_with_timestamp "Building command..."

CMD="python \"${SCRIPT_PATH}\""
CMD="${CMD} --dataset_path \"${DATASET_PATH}\""
CMD="${CMD} --repo_id \"${REPO_ID}\""

if [ "$CREATE_CARD" = "true" ]; then
    CMD="${CMD} --create_dataset_card"
fi

log_with_timestamp "Command to execute:"
log_with_timestamp "$CMD"
log_with_timestamp "=========================================="

# Execute the command
log_with_timestamp "Starting upload process..."
start_time=$(date +%s)

eval $CMD 2>&1 | tee -a "$LOG_FILE"

# Check exit status
EXIT_CODE=${PIPESTATUS[0]}
end_time=$(date +%s)
duration=$((end_time - start_time))

log_with_timestamp "=========================================="
log_with_timestamp "Upload process completed"
log_with_timestamp "Duration: ${duration} seconds ($(($duration / 60))m $(($duration % 60))s)"
log_with_timestamp "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    log_with_timestamp "✅ SUCCESS: Data upload completed successfully"
    log_with_timestamp ""
    log_with_timestamp "🎉 Your dataset is now available at:"
    log_with_timestamp "   https://huggingface.co/datasets/$REPO_ID"
    log_with_timestamp ""
    log_with_timestamp "Next steps:"
    log_with_timestamp "1. Visit the dataset page to verify the upload"
    log_with_timestamp "2. Update the dataset description if needed"
    log_with_timestamp "3. Test loading the dataset:"
    log_with_timestamp "   from datasets import load_dataset"
    log_with_timestamp "   dataset = load_dataset('$REPO_ID')"
else
    log_with_timestamp "❌ ERROR: Data upload failed with exit code $EXIT_CODE"
    log_with_timestamp ""
    log_with_timestamp "🔧 Troubleshooting steps:"
    log_with_timestamp "1. Authentication:"
    log_with_timestamp "   huggingface-cli login"
    log_with_timestamp "   huggingface-cli whoami"
    log_with_timestamp ""
    log_with_timestamp "2. Repository access:"
    log_with_timestamp "   - Ensure repository exists: https://huggingface.co/datasets/$REPO_ID"
    log_with_timestamp "   - Check write permissions"
    log_with_timestamp "   - Try: huggingface-cli repo create --type dataset $REPO_ID"
    log_with_timestamp ""
    log_with_timestamp "3. Data validation:"
    log_with_timestamp "   - Check JSON file format"
    log_with_timestamp "   - Verify folder structure"
    log_with_timestamp ""
    log_with_timestamp "4. Network connectivity:"
    log_with_timestamp "   - Test: curl -I https://huggingface.co"
    log_with_timestamp ""
    log_with_timestamp "Check the full log for details: $LOG_FILE"
fi

log_with_timestamp "=========================================="

# Return the exit code
exit $EXIT_CODE