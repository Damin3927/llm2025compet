#!/bin/bash

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dataset> --prompt-version VERSION [--ids ID1,ID2,...]"
    echo ""
    echo "Required arguments:"
    echo "  dataset: 'phybench' or 'physreason'"
    echo "  --prompt-version: Prompt version (e.g., 3.0)"
    echo ""
    echo "Optional arguments:"
    echo "  --ids ID1,ID2,...  Process specific IDs"
    exit 1
fi

DATASET=$1
shift

# Parse arguments
PROMPT_VERSION=""
IDS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt-version)
            PROMPT_VERSION="$2"
            shift 2
            ;;
        --ids)
            IDS="--ids $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$PROMPT_VERSION" ]]; then
    echo "Error: --prompt-version is required"
    exit 1
fi

if [[ "$DATASET" != "phybench" && "$DATASET" != "physreason" ]]; then
    echo "Error: Dataset must be 'phybench' or 'physreason'"
    exit 1
fi

echo "Generating CoTs with prompt version $PROMPT_VERSION..."
uv run python 2_generate_cot.py "$DATASET" --prompt-version "$PROMPT_VERSION" $IDS

sleep 2

echo "Evaluating CoTs..."
uv run python 3_evaluate_cot.py "$DATASET" $IDS

echo "Done"