#!/usr/bin/env bash
# SLURM launcher for LoRA merge on huge models (e.g., Qwen3-235B-A22B)
# - Designed to run on a single node even in multi-node clusters
# - Uses transformers accelerate-style offload (device_map=auto + offload_folder)
#
# Usage examples:
#   sbatch inference/merge_lora/run.sh \
#     --base-model Qwen/Qwen3-235B-A22B \
#     --adapters your/adapter_repo \
#     --out-dir /path/to/merged_out
#
#   # Multiple adapters (merged sequentially)
#   sbatch inference/merge_lora/run.sh \
#     --base-model Qwen/Qwen3-235B-A22B \
#     --adapters your/adapterA your/adapterB \
#     --out-dir /path/to/merged_out
#
#   # Upload to Hugging Face Hub
#   sbatch inference/merge_lora/run.sh \
#     --base-model Qwen/Qwen3-235B-A22B \
#     --adapters your/adapter \
#     --out-dir /path/to/merged_out \
#     --upload-to-hub \
#     --hub-repo-id username/merged-model

#SBATCH --job-name=lora-merge
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0               # all available memory
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# Default args (can be overridden by CLI flags passed to this script)
BASE_MODEL=""
OUT_DIR=""
DTYPE="bf16"
DEVICE_MAP="auto"
OFFLOAD_DIR="${OFFLOAD_DIR:-${HOME}/lora/offload_${SLURM_JOB_ID:-$$}}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5GB}"
TRUST_REMOTE_CODE="--trust-remote-code"
SAVE_TOKENIZER="--save-tokenizer"
SMOKE_TEST="--smoke-test"
UPLOAD_TO_HUB=""
HUB_REPO_ID=""
HUB_TOKEN=""
HUB_PRIVATE=""

ADAPTERS=()

# Parse args
while [[ $# -gt 0 ]]; do
	case "$1" in
		--base-model)
			BASE_MODEL="$2"; shift 2 ;;
		--out-dir)
			OUT_DIR="$2"; shift 2 ;;
		--dtype)
			DTYPE="$2"; shift 2 ;;
		--device-map)
			DEVICE_MAP="$2"; shift 2 ;;
		--offload-dir)
			OFFLOAD_DIR="$2"; shift 2 ;;
		--max-shard-size)
			MAX_SHARD_SIZE="$2"; shift 2 ;;
		--no-trust-remote-code)
			TRUST_REMOTE_CODE=""; shift ;;
		--no-save-tokenizer)
			SAVE_TOKENIZER=""; shift ;;
		--no-smoke-test)
			SMOKE_TEST=""; shift ;;
		--upload-to-hub)
			UPLOAD_TO_HUB="--upload-to-hub"; shift ;;
		--hub-repo-id)
			HUB_REPO_ID="$2"; shift 2 ;;
		--hub-token)
			HUB_TOKEN="$2"; shift 2 ;;
		--hub-private)
			HUB_PRIVATE="--hub-private"; shift ;;
		--adapters)
			shift
			while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
				ADAPTERS+=("$1"); shift
			done
			;;
		*)
			echo "[WARN] Unknown arg: $1"; shift ;;
	esac
done

if [[ -z "$BASE_MODEL" || -z "$OUT_DIR" || ${#ADAPTERS[@]} -eq 0 ]]; then
	echo "Usage: sbatch $0 --base-model <repo|path> --adapters <a> [b ...] --out-dir <dir> [--dtype bf16|fp16|fp32] [--device-map auto|cpu] [--offload-dir <dir>] [--max-shard-size 5GB] [--no-trust-remote-code] [--no-save-tokenizer] [--no-smoke-test] [--upload-to-hub] [--hub-repo-id <repo>] [--hub-token <token>] [--hub-private]"
	exit 2
fi

mkdir -p "$OFFLOAD_DIR"
mkdir -p "$OUT_DIR"

echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "[INFO] OFFLOAD_DIR=$OFFLOAD_DIR"

# Speed up HF hub downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_OFFLINE=0

module purge
module load cuda/12.4
module load cudnn/9.6.0
module load nccl/2.24.3

CMD=("python" "main.py"
	--base-model "$BASE_MODEL"
	--out-dir "$OUT_DIR"
	--dtype "$DTYPE"
	--device-map "$DEVICE_MAP"
	--offload-dir "$OFFLOAD_DIR"
	--max-shard-size "$MAX_SHARD_SIZE"
)

if [[ -n "$TRUST_REMOTE_CODE" ]]; then CMD+=("$TRUST_REMOTE_CODE"); fi
if [[ -n "$SAVE_TOKENIZER" ]]; then CMD+=("$SAVE_TOKENIZER"); fi
if [[ -n "$SMOKE_TEST" ]]; then CMD+=("$SMOKE_TEST"); fi
if [[ -n "$UPLOAD_TO_HUB" ]]; then CMD+=("$UPLOAD_TO_HUB"); fi
if [[ -n "$HUB_REPO_ID" ]]; then CMD+=(--hub-repo-id "$HUB_REPO_ID"); fi
if [[ -n "$HUB_TOKEN" ]]; then CMD+=(--hub-token "$HUB_TOKEN"); fi
if [[ -n "$HUB_PRIVATE" ]]; then CMD+=("$HUB_PRIVATE"); fi

# Add adapters at the end
CMD+=(--adapters "${ADAPTERS[@]}")

echo "[INFO] Launching: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Merge completed. Output: $OUT_DIR"
