#!/bin/bash

# VLLM Start Script
# Starts a distributed VLLM server using Singularity containers and Ray
# Supports model and container image downloading

set -euo pipefail

# =============================================================================
# Configuration Variables
# =============================================================================

readonly MODEL_PATH="Qwen/Qwen3-32B"
readonly VLLM_TAG="v0.9.2"
readonly SINGULARITY_IMAGE_DIR="${HOME}/.singularity"
readonly RAY_HEAD_PORT="6379"
readonly VLLM_API_KEY="token-abc123"

# =============================================================================
# Functions
# =============================================================================

download_model() {
    local model_path="${1:-$MODEL_PATH}"
    echo "Downloading model..."
    huggingface-cli download "$model_path"
    echo "Model downloaded to $(huggingface-cli info "$model_path" | grep 'Local path' | awk '{print $3}')"
}

download_singularity_image() {
    echo "Downloading Singularity image..."
    mkdir -p "$SINGULARITY_IMAGE_DIR"
    singularity pull "${SINGULARITY_IMAGE_DIR}/vllm-openai_${VLLM_TAG}.sif" \
        "docker://vllm/vllm-openai:${VLLM_TAG}"
}

validate_model() {
    local model_path="$1"
    echo "Validating model path..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        if huggingface-cli info "$model_path" >/dev/null 2>&1; then
            echo "Model $model_path validated successfully"
        else
            echo "Warning: Model $model_path not found locally. It will be downloaded during startup."
        fi
    else
        echo "Warning: huggingface-cli not found. Skipping model validation."
    fi
}

check_gpu_availability() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        local available_gpus
        available_gpus=$(nvidia-smi --list-gpus | wc -l)
        echo "Found $available_gpus GPU(s)"
        if [ "$available_gpus" -lt "$NGPUS" ]; then
            echo "Warning: Requested $NGPUS GPUs but only $available_gpus available"
        fi
    else
        echo "Warning: nvidia-smi not found. Cannot verify GPU availability."
    fi
}

setup_environment() {
    # Add custom bin directory to PATH
    export PATH="$HOME/.bin:$PATH"
    
    # Set HuggingFace cache directory
    export HF_HOME="$HOME/.cache/huggingface"
    
    # Configure NCCL for multi-node communication
    export NCCL_DEBUG=INFO
    export NCCL_NET_GDR_LEVEL=SYS
    export NCCL_P2P_LEVEL=SYS
    
    # Disable Ray usage stats collection for privacy and faster startup
    export RAY_DISABLE_USAGE_STATS=1
    
    # Set CUDA architecture list to optimize compilation (common architectures for H100/A100)
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
    
    # Optimize PyTorch performance
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=true
}

get_cluster_info() {
    # Get SLURM cluster information
    JOB_NODELIST_RAW=$(scontrol show job "$SLURM_JOBID" | grep NodeList=osk | cut -d'=' -f2)
    NODELIST=$(scontrol show hostnames "$JOB_NODELIST_RAW")
    echo nodelist $NODELIST
    NODE_RANK="$SLURM_NODEID"
    echo node_rank $NODE_RANK

    if [ "$NODE_RANK" == "0" ]; then
        echo "rank 0 node"
    else
	HEAD_NODE_HOSTNAME=$(echo "$NODELIST" | head -n 1 | awk '{print $1}')
	echo head_node $HEAD_NODE_HOSTNAME
	NODE0_IP=$(getent hosts "$HEAD_NODE_HOSTNAME" | awk '{print $1}' | grep -E '^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$' | head -n 1)
	echo node0-ip $NODE0_IP
    fi
}

build_ray_command() {
    local model_path="$1"
    local api_key="$2"
    local vllm_command="vllm serve --dtype auto --api-key $api_key --tensor-parallel-size $NGPUS --pipeline-parallel-size $NNODES --enable-prefix-caching $model_path"
    local ray_start_cmd="ray start --disable-usage-stats"
    
    if [ "$NODE_RANK" == "0" ]; then
        # Head node: start Ray head, wait, check status, then start VLLM
        ray_start_cmd+=" --head --port=$RAY_HEAD_PORT && sleep 10 && ray status"
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
    
    # Check if Singularity image exists
    if [ ! -f "${SINGULARITY_IMAGE_DIR}/vllm-openai_${VLLM_TAG}.sif" ]; then
        echo "Error: Singularity image not found. Run with --download-image first."
        exit 1
    fi
    
    # Create temporary home directory
    mkdir -p /var/tmp/home
    
    # Ensure HF_HOME directory exists
    mkdir -p "$HF_HOME"
    
    echo "Starting VLLM server in Singularity container..."
    
    # Run VLLM in Singularity container with error handling
    if ! singularity exec \
        --nv \
        --no-home \
        -B /var/tmp/home:"$HOME" \
        -B "$HF_HOME:/hf_home" \
        --env HF_HOME=/hf_home \
        --env RAY_DISABLE_USAGE_STATS=$RAY_DISABLE_USAGE_STATS \
        --env TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
        --env TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM \
        "${SINGULARITY_IMAGE_DIR}/vllm-openai_${VLLM_TAG}.sif" \
        bash -c "$ray_command"; then
        echo "Error: Singularity container execution failed"
        exit 1
    fi
}

# =============================================================================
# Main Script Logic
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --download-model        Download the model and exit
    --download-image        Download the Singularity image and exit
    --nnodes <number>       Number of nodes for distributed inference (required)
    --ngpus <number>        Number of GPUs per node (required)
    --model <path>          Model path to serve (default: $MODEL_PATH)
    --api-key <key>         API key for authentication (default: $VLLM_API_KEY)
    --help                  Show this help message

Examples:
    # Download model and image
    $0 --download-model
    $0 --download-image
    
    # Run single-node inference with 2 GPUs
    $0 --nnodes 1 --ngpus 2
    
    # Run with custom model and API key
    $0 --nnodes 1 --ngpus 2 --model "microsoft/DialoGPT-medium" --api-key "my-secret-key"
    
    # Run multi-node inference
    $0 --nnodes 2 --ngpus 4

Environment Variables:
    MODEL_PATH             Model to serve (default: $MODEL_PATH)
    VLLM_TAG              vLLM container version (default: $VLLM_TAG)
    VLLM_API_KEY          API key for authentication (default: $VLLM_API_KEY)
EOF
}

main() {
    # Initialize variables from defaults
    local model_path="$MODEL_PATH"
    local api_key="$VLLM_API_KEY"
    local download_model_flag=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "${1}" in
            "--download-model")
                download_model_flag=true
                shift
                ;;
            "--download-image")
                download_singularity_image
                exit 0
                ;;
            "--help"|"-h")
                show_usage
                exit 0
                ;;
            "--nnodes")
                if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
                    NNODES="$2"
                    shift 2
                else
                    echo "Error: --nnodes requires a positive integer argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--ngpus")
                if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
                    NGPUS="$2"
                    shift 2
                else
                    echo "Error: --ngpus requires a positive integer argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--model")
                if [[ -n "${2:-}" ]]; then
                    model_path="$2"
                    shift 2
                else
                    echo "Error: --model requires a model path argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--api-key")
                if [[ -n "${2:-}" ]]; then
                    api_key="$2"
                    shift 2
                else
                    echo "Error: --api-key requires an API key argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            *)
                echo "Unknown argument: $1" >&2
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Handle download model flag
    if [[ "$download_model_flag" == true ]]; then
        download_model "$model_path"
        exit 0
    fi
    
    # Check if required arguments are set
    if [[ -z "${NNODES:-}" ]]; then
        echo "Error: --nnodes argument is required" >&2
        show_usage
        exit 1
    fi
    
    if [[ -z "${NGPUS:-}" ]]; then
        echo "Error: --ngpus argument is required" >&2
        show_usage
        exit 1
    fi
    
    # Setup environment and cluster
    setup_environment
    echo "Validating configuration..."
    validate_model "$model_path"
    check_gpu_availability
    echo "Getting cluster info..."
    get_cluster_info
    
    # Build and execute Ray command
    echo "Building Ray command..."
    local ray_command
    ray_command=$(build_ray_command "$model_path" "$api_key")
    echo "Starting Singularity container..."
    run_singularity_container "$ray_command"
}

# Execute main function with all arguments
main "$@"
