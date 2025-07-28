#!/bin/bash

# SLURM submission script for upload_data.py with parameters
# Usage: ./submit_upload_data.sh [dataset_path] [repo_id] [create_card]

# Default values
DEFAULT_DATASET_PATH="./results/selected_data"
DEFAULT_REPO_ID="your-username/your-dataset-name"
DEFAULT_CREATE_CARD="true"

# Parse command line arguments
DATASET_PATH="${1:-$DEFAULT_DATASET_PATH}"
REPO_ID="${2:-$DEFAULT_REPO_ID}"
CREATE_CARD="${3:-$DEFAULT_CREATE_CARD}"

# Validate inputs
if [[ ! "$REPO_ID" =~ ^[^/]+/[^/]+$ ]]; then
    echo "❌ ERROR: Invalid repository ID format: $REPO_ID"
    echo "Expected format: username/dataset-name"
    exit 1
fi

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Generate unique job name and timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPO_NAME=$(basename "$REPO_ID")
JOB_NAME="upload_${REPO_NAME}_${TIMESTAMP}"

echo "Submitting SLURM job with parameters:"
echo "  Dataset path: $DATASET_PATH"
echo "  Repository ID: $REPO_ID" 
echo "  Create dataset card: $CREATE_CARD"
echo "  Job name: $JOB_NAME"
echo ""

# Create the job script on the fly
cat > temp_upload_job_${TIMESTAMP}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/upload_data_%j.out
#SBATCH --error=logs/upload_data_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu

# Create necessary directories
mkdir -p logs

echo "=========================================="
echo "SLURM Upload Data Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Started at: \$(date)"
echo "Dataset path: $DATASET_PATH"
echo "Repository ID: $REPO_ID"
echo "Create dataset card: $CREATE_CARD"
echo "=========================================="

# Show dataset structure
echo "Dataset structure:"
find "$DATASET_PATH" -type f -name "*.json" | head -5 | while read file; do
    echo "  \$file"
done
json_count=\$(find "$DATASET_PATH" -type f -name "*.json" | wc -l)
if [ \$json_count -gt 5 ]; then
    echo "  ... and \$(expr \$json_count - 5) more files"
fi
echo "Total JSON files: \$json_count"
echo ""

# Activate environment (uncomment and modify as needed)
# source ~/.bashrc
# conda activate hf

# Run the upload script
python upload_data.py \\
    --dataset_path "$DATASET_PATH" \\
    --repo_id "$REPO_ID" \\
$([ "$CREATE_CARD" = "true" ] && echo "    --create_dataset_card \\")

EXIT_CODE=\$?
echo "=========================================="
echo "Finished at: \$(date)"
echo "Exit code: \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Data upload completed successfully"
    echo "🎉 Dataset available at: https://huggingface.co/datasets/$REPO_ID"
    
    # Show upload summary
    echo ""
    echo "Upload summary:"
    echo "- Repository: $REPO_ID"
    echo "- Source: $DATASET_PATH"
    echo "- Dataset card: $CREATE_CARD"
else
    echo "❌ ERROR: Data upload failed with exit code \$EXIT_CODE"
    echo ""
    echo "Check the logs and verify:"
    echo "1. HuggingFace authentication (huggingface-cli login)"
    echo "2. Repository exists and has write access"
    echo "3. Data format is valid JSON"
    echo "4. Network connectivity to HuggingFace Hub"
fi
echo "=========================================="
EOF

# Submit the job
echo "Submitting job to SLURM..."
job_id=$(sbatch temp_upload_job_${TIMESTAMP}.sh | grep -o '[0-9]*')

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "Job ID: $job_id"
    echo "Job name: $JOB_NAME"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -u \$USER"
    echo "  squeue -j $job_id"
    echo ""
    echo "View logs:"
    echo "  tail -f logs/upload_data_${job_id}.out"
    echo "  tail -f logs/upload_data_${job_id}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $job_id"
else
    echo "❌ Failed to submit job"
    exit 1
fi

# Clean up temporary job script
rm temp_upload_job_${TIMESTAMP}.sh

echo "Job submitted successfully!"