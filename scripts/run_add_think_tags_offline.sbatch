#!/bin/bash
#SBATCH --job-name=think-tags-offline
#SBATCH --partition=P01
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euxo pipefail

# Condaを利用する場合の例
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate myenv || true
fi

cd /home/argo/llm2025compet

python -m pip install -r requirements-think-offline.txt --no-input --quiet || true

python data/dna/add_think_tags_offline.py \
  --start_index ${START_INDEX:-0} \
  --end_index ${END_INDEX:-10} \
  --dataset_name ${DATASET_NAME:-argo11/DNA_DPO_hh-rlhf} \
  --log_dir data/dna/logs \
  ${APPLY_NONPREF:+--apply_to_non_preferred} \
  ${USE_TRANSFORMERS:+--use_transformers} \
  ${TRANSFORMER_MODEL:+--transformer_model ${TRANSFORMER_MODEL}} \
  ${DEVICE:+--device ${DEVICE}} \
  --output_file ${OUTPUT_FILE:-data/dna/think_tagged_${START_INDEX:-0}-$(( ${END_INDEX:-10}-1 )).jsonl}


