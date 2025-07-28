#!/bin/bash

# SLURM submission script with parameters
# Usage: ./submit_length_selector.sh [dataset_name] [total_samples] [output_prefix]

# Default values
DEFAULT_DATASET="neko-llm/SFT_OpenMathReasoning"
DEFAULT_SAMPLES=10000
DEFAULT_OUTPUT_PREFIX="selected_data"

# Parse command line arguments
DATASET_NAME="${1:-$DEFAULT_DATASET}"
TOTAL_SAMPLES="${2:-$DEFAULT_SAMPLES}"
OUTPUT_PREFIX="${3:-$DEFAULT_OUTPUT_PREFIX}"

# Additional parameters (can be modified as needed)
DATASET_CONFIG="cot"
SPLIT="train"
ANSWER_FIELD="generated_solution"
USE_SHUFFLE="--shuffle"  # Remove this for sequential processing

# Dynamic binning parameters
SAMPLE_SIZE_FOR_STATS=1000  # Number of samples to analyze for bin creation
NUM_BINS=6  # Number of length bins to create

# Generate unique job name and output file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="length_selector_${TOTAL_SAMPLES}_${TIMESTAMP}"
OUTPUT_DIR="./selected_data"
OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${TOTAL_SAMPLES}_${TIMESTAMP}.json"

echo "Submitting SLURM job with parameters:"
echo "  Dataset: $DATASET_NAME"
echo "  Config: $DATASET_CONFIG"
echo "  Split: $SPLIT"
echo "  Answer field: $ANSWER_FIELD"
echo "  Total samples: $TOTAL_SAMPLES"
echo "  Sample size for stats: $SAMPLE_SIZE_FOR_STATS"
echo "  Number of bins: $NUM_BINS"
echo "  Output: $OUTPUT_FILE"
echo "  Job name: $JOB_NAME"
echo ""

# Create the job script on the fly
cat > temp_job_${TIMESTAMP}.sh << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/length_selector_%j.out
#SBATCH --error=logs/length_selector_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Create necessary directories
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "SLURM Length Selector Job"
echo "=========================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Started at: \$(date)"
echo "Dataset: $DATASET_NAME"
echo "Config: $DATASET_CONFIG"
echo "Split: $SPLIT"
echo "Answer field: $ANSWER_FIELD"
echo "Target samples: $TOTAL_SAMPLES"
echo "Sample size for stats: $SAMPLE_SIZE_FOR_STATS"
echo "Number of bins: $NUM_BINS"
echo "Output: $OUTPUT_FILE"
echo "=========================================="

# Activate environment (uncomment and modify as needed)
# source ~/.bashrc
# conda activate hf

# Run the length selector
python length_selector.py \\
    --input "$DATASET_NAME" \\
    --dataset_config "$DATASET_CONFIG" \\
    --split "$SPLIT" \\
    --answer_field "$ANSWER_FIELD" \\
    --total_samples $TOTAL_SAMPLES \\
    --sample_size_for_stats $SAMPLE_SIZE_FOR_STATS \\
    --num_bins $NUM_BINS \\
    --output "$OUTPUT_FILE" \\
    $USE_SHUFFLE

EXIT_CODE=\$?
echo "=========================================="
echo "Finished at: \$(date)"
echo "Exit code: \$EXIT_CODE"

if [ \$EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Length selection completed successfully"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file size: \$(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "Number of selected items: \$(jq '. | length' "$OUTPUT_FILE" 2>/dev/null || echo 'jq not available')"
    fi
else
    echo "ERROR: Length selection failed with exit code \$EXIT_CODE"
fi
echo "=========================================="
EOF

# Submit the job
sbatch temp_job_${TIMESTAMP}.sh

# Clean up temporary job script
rm temp_job_${TIMESTAMP}.sh

echo "Job submitted successfully!"
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/"