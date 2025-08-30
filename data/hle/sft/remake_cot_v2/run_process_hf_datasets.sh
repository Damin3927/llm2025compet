#!/bin/bash

# Default values
CONFIG_FILE="sft_dataset_v1.json"
OUTPUT_DIR="./results/processed_datasets_v1"
EVALUATOR_MODELS="deepseek/deepseek-r1-0528:free,moonshotai/kimi-k2:free"
# REGENERATOR_MODELS="qwen/qwen3-235b-a22b:free,z-ai/glm-4.5-air:free"
REGENERATOR_MODELS="qwen/qwen3-235b-a22b:free"
GRADE_THRESHOLD="C"
# MAX_ITEMS=5
NO_ASYNC=false
CREATE_EXAMPLE=false

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config FILE              Path to JSON configuration file"
    echo "  --output-dir DIR           Output directory for processed datasets (default: ./processed_datasets)"
    echo "  --evaluator-models MODELS  Comma-separated list of models for evaluation"
    echo "  --regenerator-models MODELS Comma-separated list of models for regeneration"
    echo "  --grade-threshold GRADE    Regenerate items with this grade or lower (A,B,C,D, default: C)"
    echo "  --max-items N              Maximum items to process per dataset (for testing)"
    echo "  --no-async                 Disable async multi-model regeneration"
    echo "  --create-example-config    Create an example configuration file and exit"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --create-example-config"
    echo "  $0 --config datasets_config.json --max-items 10"
    echo "  $0 --config my_config.json --output-dir ./results --grade-threshold B"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --evaluator-models)
            EVALUATOR_MODELS="$2"
            shift 2
            ;;
        --regenerator-models)
            REGENERATOR_MODELS="$2"
            shift 2
            ;;
        --grade-threshold)
            GRADE_THRESHOLD="$2"
            shift 2
            ;;
        --max-items)
            MAX_ITEMS="$2"
            shift 2
            ;;
        --no-async)
            NO_ASYNC=true
            shift
            ;;
        --create-example-config)
            CREATE_EXAMPLE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the Python command
PYTHON_CMD="python3 process_hf_datasets.py"

# Add arguments
if [ "$CREATE_EXAMPLE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --create-example-config"
else
    if [ -n "$CONFIG_FILE" ]; then
        PYTHON_CMD="$PYTHON_CMD --config \"$CONFIG_FILE\""
    fi
    
    PYTHON_CMD="$PYTHON_CMD --output-dir \"$OUTPUT_DIR\""
    
    if [ -n "$EVALUATOR_MODELS" ]; then
        PYTHON_CMD="$PYTHON_CMD --evaluator-models \"$EVALUATOR_MODELS\""
    fi
    
    if [ -n "$REGENERATOR_MODELS" ]; then
        PYTHON_CMD="$PYTHON_CMD --regenerator-models \"$REGENERATOR_MODELS\""
    fi
    
    PYTHON_CMD="$PYTHON_CMD --grade-threshold \"$GRADE_THRESHOLD\""
    
    if [ -n "$MAX_ITEMS" ]; then
        PYTHON_CMD="$PYTHON_CMD --max-items $MAX_ITEMS"
    fi
    
    if [ "$NO_ASYNC" = true ]; then
        PYTHON_CMD="$PYTHON_CMD --no-async"
    fi
fi

# Print the command being executed
echo "Executing: $PYTHON_CMD"
echo ""

# Execute the Python script
eval "$PYTHON_CMD"