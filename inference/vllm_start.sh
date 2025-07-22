#!/bin/bash

# VLLM Start Script
# Starts a distributed VLLM server using Singularity containers and Ray
# Supports model and container image downloading

set -euo pipefail  # Exit on errors, undefined vars, and pipe failures

# =============================================================================
# Configuration Variables
# =============================================================================

readonly MODEL_PATH="Qwen/Qwen3-235B-A22B"
readonly VLLM_TAG="v0.9.2"
readonly SINGULARITY_IMAGE_DIR="${HOME}/.singularity"
readonly TARGET_HF_HOME="/var/tmp/hf_home"
readonly RAY_HEAD_PORT="6379"
readonly TENSOR_PARALLEL_SIZE="8"
readonly VLLM_API_KEY="token-abc123"

# =============================================================================
# Functions
# =============================================================================

download_model() {
    echo "Downloading model..."
    huggingface-cli download "$MODEL_PATH"
    echo "Model downloaded to $(huggingface-cli info "$MODEL_PATH" | grep 'Local path' | awk '{print $3}')"
}

download_singularity_image() {
    echo "Downloading Singularity image..."
    mkdir -p "$SINGULARITY_IMAGE_DIR"
    singularity pull "${SINGULARITY_IMAGE_DIR}/vllm-openai_${VLLM_TAG}.sif" \
        "docker://vllm/vllm-openai:${VLLM_TAG}"
}

setup_environment() {
    # Add custom bin directory to PATH
    export PATH="$HOME/.bin:$PATH"
    
    # Set HuggingFace cache directory
    export HF_HOME="$HOME/.cache/huggingface"
    
    # Configure NCCL for multi-node communication
    export NCCL_DEBUG=INFO
    export NCCL_IB_HCA=mlx5
    export NCCL_SOCKET_IFNAME=ib0  # Run `ifconfig` to see your infiniband interface
    export NCCL_NET_GDR_LEVEL=SYS  # Modify based on your system needs
    export NCCL_P2P_LEVEL=SYS      # Modify based on your system needs
}

sync_hf_cache() {
    echo "Syncing HF cache..."
    rsync -a --links -r "$HF_HOME" "$TARGET_HF_HOME"
    echo "HF cache synced"
}

get_cluster_info() {
    # Get SLURM cluster information
    NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
    NODE0_IP=$(host -4 $(echo "$NODELIST" | awk '{print $1}') | awk '{print $4}')
    NNODES=$(echo "$NODELIST" | wc -w)
    NODE_RANK="$SLURM_NODEID"
}

build_ray_command() {
    local vllm_command="vllm serve --dtype auto --api-key $VLLM_API_KEY --tensor-parallel-size $TENSOR_PARALLEL_SIZE --pipeline-parallel-size $NNODES --enable-prefix-caching $MODEL_PATH"
    local ray_start_cmd="ray start"
    
    if [ "$NODE_RANK" == "0" ]; then
        # Head node: start Ray head, wait, check status, then start VLLM
        ray_start_cmd+=" --head --port=$RAY_HEAD_PORT && sleep 5 && ray status"
        ray_start_cmd+=" && $vllm_command"
    else
        # Worker node: connect to head node and block
        ray_start_cmd+=" --block --address=${NODE0_IP}:${RAY_HEAD_PORT}"
    fi
    
    echo "$ray_start_cmd"
}

run_singularity_container() {
    local ray_command="$1"
    
    echo "RAY_START_CMD: $ray_command"
    
    # Create temporary home directory
    mkdir -p /var/tmp/home
    
    # Run VLLM in Singularity container
    singularity exec \
        --nv \
        --no-home \
        -B /var/tmp/home:"$HOME" \
        -B "$HF_HOME:/hf_home" \
        --env HF_HOME=/hf_home \
        "${SINGULARITY_IMAGE_DIR}/vllm-openai_${VLLM_TAG}.sif" \
        bash -c "$ray_command"
}

# =============================================================================
# Main Script Logic
# =============================================================================

main() {
    # Handle command line arguments
    case "${1:-}" in
        "--download-model")
            download_model
            exit 0
            ;;
        "--download-image")
            download_singularity_image
            exit 0
            ;;
    esac
    
    # Setup environment and cluster
    setup_environment
    sync_hf_cache
    get_cluster_info
    
    # Build and execute Ray command
    local ray_command
    ray_command=$(build_ray_command)
    run_singularity_container "$ray_command"
}

# Execute main function with all arguments
main "$@"