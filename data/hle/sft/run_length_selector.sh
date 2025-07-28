#!/bin/bash
#SBATCH --job-name=length_selector
#SBATCH --output=logs/length_selector_%j.out
#SBATCH --error=logs/length_selector_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Configuration - Edit these variables as needed
DATASET_NAME="nvidia/OpenMathReasoning"
DATASET_CONFIG="default"
SPLIT="genselect"
ID_HEADER="OpenMathReasoning_genselect"  # Prefix for generated IDs (e.g., "problem_1", "problem_2", etc.)
QUESTION_FIELD="problem"
SOLUTION_FIELD="generated_solution"
ANSWER_FIELD="expected_answer"
TOTAL_SAMPLES=5000
OUTPUT_DIR="./results/selected_data/${ID_HEADER}"
OUTPUT_FILE="${OUTPUT_DIR}/${ID_HEADER}_${TOTAL_SAMPLES}_samples.json"
USE_SHUFFLE=true  # Set to false for sequential processing

# Dynamic binning parameters
SAMPLE_SIZE_FOR_STATS=5000  # Number of samples to analyze for bin creation
NUM_BINS=6  # Number of length bins to create
CURVE_SHARPNESS=2.0  # Sharpness of half-Gaussian curve (higher = more peaked at longest items)

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/length_selector.py"

# Create necessary directories
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# Detect if running under SLURM or locally
if [ -n "$SLURM_JOB_ID" ]; then
    RUNNING_MODE="SLURM"
    JOB_INFO="Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
else
    RUNNING_MODE="Local"
    JOB_INFO="Running locally on $(hostname)"
fi

echo "=========================================="
echo "Length Selector Job ($RUNNING_MODE)"
echo "=========================================="
echo "$JOB_INFO"
echo "Started at: $(date)"
echo "Dataset: $DATASET_NAME"
echo "Config: $DATASET_CONFIG"
echo "Split: $SPLIT"
echo "Question field: $QUESTION_FIELD"
echo "Solution field: $SOLUTION_FIELD"
echo "Answer field: $ANSWER_FIELD"
echo "Target samples: $TOTAL_SAMPLES"
echo "Output: $OUTPUT_FILE"
echo "Shuffle: $USE_SHUFFLE"
echo "Sample size for stats: $SAMPLE_SIZE_FOR_STATS"
echo "Number of bins: $NUM_BINS"
echo "Curve sharpness: $CURVE_SHARPNESS"
echo "ID header: $ID_HEADER"
echo "=========================================="

# Activate conda environment (adjust as needed)
# source ~/.bashrc
# conda activate hf  # Replace with your environment name

# Or use module system if available
# module load python/3.9
# module load cuda/11.8  # If needed

# Build command
CMD="python ${SCRIPT_PATH}"
CMD="${CMD} --input ${DATASET_NAME}"
CMD="${CMD} --output ${OUTPUT_FILE}"
CMD="${CMD} --answer_field ${ANSWER_FIELD}"
CMD="${CMD} --total_samples ${TOTAL_SAMPLES}"

if [ ! -z "$DATASET_CONFIG" ]; then
    CMD="${CMD} --dataset_config ${DATASET_CONFIG}"
fi

if [ ! -z "$SPLIT" ]; then
    CMD="${CMD} --split ${SPLIT}"
fi

if [ "$USE_SHUFFLE" = true ]; then
    CMD="${CMD} --shuffle"
fi

CMD="${CMD} --sample_size_for_stats ${SAMPLE_SIZE_FOR_STATS}"
CMD="${CMD} --num_bins ${NUM_BINS}"
CMD="${CMD} --curve_sharpness ${CURVE_SHARPNESS}"
CMD="${CMD} --question_field ${QUESTION_FIELD}"
CMD="${CMD} --solution_field ${SOLUTION_FIELD}"
CMD="${CMD} --id_header ${ID_HEADER}"

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
    echo "SUCCESS: Length selection completed successfully"
    
    # Show output file stats
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Output file size: $(du -h "$OUTPUT_FILE" | cut -f1)"
        echo "Number of lines: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo "ERROR: Length selection failed with exit code $EXIT_CODE"
fi

echo "=========================================="