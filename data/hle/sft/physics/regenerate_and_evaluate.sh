#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset> [--ids ID1,ID2,...] [--force]"
    exit 1
fi

DATASET=$1
shift
OPTIONS="$@"

if [[ "$DATASET" != "phybench" && "$DATASET" != "physreason" ]]; then
    echo "Error: Dataset must be 'phybench' or 'physreason'"
    exit 1
fi

echo "Regenerating CoTs..."
uv run python 4_regenerate_cot.py "$DATASET" $OPTIONS

sleep 2

echo "Evaluating CoTs..."
uv run python 3_evaluate_cot.py "$DATASET" $OPTIONS

echo "Done"