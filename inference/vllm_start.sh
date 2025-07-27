#!/bin/bash

# VLLM Start Script
# Starts a distributed VLLM server using Ray
# Supports model and container image downloading

set -euo pipefail

# =============================================================================
# Configuration Variables
# =============================================================================

readonly MODEL_PATH="Qwen/Qwen3-32B"
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
    # Load necessary modules
    echo "Loading modules..."
    module purge
    module load cuda/12.4
    module load cudnn/9.6.0
    module load nccl/2.24.3
    module load hpcx/2.18.1-gcc-cuda12/hpcx-mt

    # Add custom bin directory to PATH
    export PATH="$HOME/.bin:$PATH"
    
    # Set HuggingFace cache directory
    export HF_HOME="$HOME/.cache/huggingface"
    
    # Configure NCCL for multi-node communication
    # export NCCL_NET_GDR_LEVEL=SYS
    # export NCCL_P2P_LEVEL=SYS

    export NCCL_DEBUG=TRACE
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    # export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=0
    export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1

    # Disable Ray usage stats collection for privacy and faster startup
    export RAY_DISABLE_USAGE_STATS=1
    
    # Set CUDA architecture list to optimize compilation (common architectures for H100/A100)
    # export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
    
    # Optimize PyTorch performance
    # export OMP_NUM_THREADS=1
    # export TOKENIZERS_PARALLELISM=true

    export VLLM_LOGGING_LEVEL=DEBUG
    # export CUDA_LAUNCH_BLOCKING=1
    export VLLM_TRACE_FUNCTION=1

    export NCCL_SOCKET_IFNAME=enp25s0np0
    export NVTE_FUSED_ATTN=0
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=0
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    #export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    # export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
    unset ROCR_VISIBLE_DEVICES
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

run_ray_command() {
    local model_path="$1"
    local api_key="$2"
    local total_gpus=$((NGPUS * NNODES))
    
    if [ "$NODE_RANK" == "0" ]; then
        echo "RANK: $NODE_RANK. Starting Ray head node..."
        ray start --disable-usage-stats --head --port=$RAY_HEAD_PORT
        echo "Ray head node started, waiting for worker nodes to connect..."
        sleep 30
        echo "Checking Ray cluster status..."
        ray status
        echo "Expected: ${NNODES} nodes with ${NGPUS} GPUs each (total: $total_gpus GPUs)"
        echo "Ray cluster ready!"
    else
        echo "RANK: $NODE_RANK. Connecting to Ray head node"
        ray start --disable-usage-stats --block --address="10.255.255.54:${RAY_HEAD_PORT}"
        echo "Ray worker node connected to head node"
    fi
}

run_vllm() {
    vllm serve --dtype auto --api-key "$VLLM_API_KEY" \
        --tensor-parallel-size "$NGPUS" \
        --pipeline-parallel-size "$NNODES" \
        --enable-prefix-caching \
        "$MODEL_PATH"
}

# =============================================================================
# Main Script Logic
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --download-model        Download the model and exit
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
    
    echo "RANK: $NODE_RANK. Starting Ray cluster..."
    run_ray_command "$model_path" "$api_key"
    
    if [[ "$NODE_RANK" == "0" ]]; then
        echo "RANK: 0. Starting VLLM server..."
        run_vllm
    fi
}

# Execute main function with all arguments
main "$@"
