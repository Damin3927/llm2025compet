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

# Args holders (will be filled when options are given)
declare -a lora_args=()
declare -a ep_args=()

# =============================================================================
# Functions
# =============================================================================

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

    export NCCL_DEBUG=INFO
    export GPU_MAX_HW_QUEUES=2
    export TORCH_NCCL_HIGH_PRIORITY=1
    export NCCL_CHECKS_DISABLE=1
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
    export NCCL_IB_GID_INDEX=3
    export NCCL_CROSS_NIC=1
    # export NCCL_PROTO=Simple
    export RCCL_MSCCL_ENABLE=0
    export TOKENIZERS_PARALLELISM=false
    export HSA_NO_SCRATCH_RECLAIM=1

    # Disable Ray usage stats collection for privacy and faster startup
    export RAY_DISABLE_USAGE_STATS=1

    # export VLLM_TRACE_FUNCTION=1
    
    # Optimize PyTorch performance
    # export OMP_NUM_THREADS=1
    # export TOKENIZERS_PARALLELISM=true

    export VLLM_LOGGING_LEVEL=DEBUG

    export NCCL_SOCKET_IFNAME=enp25s0np0
    export NVTE_FUSED_ATTN=0
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=0
    #export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
    unset ROCR_VISIBLE_DEVICES

    ulimit -v unlimited
    ulimit -m unlimited
}

get_cluster_info() {
    export VLLM_HOST_IP=$(ip -4 -o addr show bond0 | awk '{print $4}' | cut -d/ -f1)

    # Get SLURM cluster information
    NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    echo nodelist: $NODELIST
    NODE_RANK="$SLURM_NODEID"
    echo node_rank: $NODE_RANK

    if ! [ "$NODE_RANK" == "0" ]; then
        HEAD_NODE_HOSTNAME=$(echo "$NODELIST" | head -n 1 | awk '{print $1}')
        echo head_node: $HEAD_NODE_HOSTNAME
        NODE0_IP=$(getent hosts "${HEAD_NODE_HOSTNAME}gw" | awk '{print $1}')
        echo node0-ip: $NODE0_IP
    fi
}

run_ray_command() {
    local model_path="$1"
    local api_key="$2"
    export TOTAL_GPUS=$((NGPUS * NNODES))
    echo "cpus_per_task: $SLURM_CPUS_PER_TASK"
    
    if [ "$NODE_RANK" == "0" ]; then
        echo "RANK: $NODE_RANK. Starting Ray head node..."
        ray start --disable-usage-stats --head --port=$RAY_HEAD_PORT --node-ip-address=$VLLM_HOST_IP --num-cpus=$SLURM_CPUS_PER_TASK
        echo "Ray head node started, waiting for worker nodes to connect..."
        sleep 10
        echo "Checking Ray cluster status..."
        ray status
        echo "Expected: ${NNODES} nodes with ${NGPUS} GPUs each (total: $TOTAL_GPUS GPUs)"
        echo "Ray cluster ready!"
    else
        echo "RANK: $NODE_RANK. Connecting to Ray head node"
        ray start --disable-usage-stats --block --address="${NODE0_IP}:${RAY_HEAD_PORT}" --node-ip-address=$VLLM_HOST_IP --num-cpus=$SLURM_CPUS_PER_TASK
        echo "Ray worker node connected to head node"
    fi
}

run_vllm() {
    local model_path="$1"

    echo "Starting VLLM server with model: $model_path"

    vllm serve \
        "${lora_args[@]}" \
        "${ep_args[@]}" \
        --enable-lora --max-loras "${MAX_LORAS:-4}" --max-lora-rank "${MAX_LORA_RANK:-16}" \
        --dtype auto --api-key "$VLLM_API_KEY" \
        --download-dir "$HF_HOME" \
        --tensor-parallel-size $NGPUS \
        --pipeline-parallel-size $NNODES \
        --distributed-executor-backend ray \
        --trust-remote-code \
        "${model_path}"
}

# =============================================================================
# Main Script Logic
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --nnodes <number>       Number of nodes for distributed inference (required)
    --ngpus <number>        Number of GPUs per node (required)
    --model <path>          Model path to serve (default: $MODEL_PATH)
    --api-key <key>         API key for authentication (default: $VLLM_API_KEY)
    --lora "<name>=<repo_or_path>[,<name2>=...>]"   LoRA adapters (optional)
    --expert-parallel       Enable expert parallelism (for MoE models only; default: off)
    --help                  Show this help message

Examples:
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
    local lora_spec=""
    local enable_ep=0 # enable_ep is toggled by --expert-parallel
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "${1}" in
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
            "--lora")
                if [[ -n "${2:-}" ]]; then
                    lora_spec="$2"
                    shift 2
                else
                    echo "Error: --lora requires a spec like 'name=repo_or_path[,name2=...]'" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--expert-parallel")
                enable_ep=1
                # actual flag will be appended to ep_args below
                shift 1
                ;;
            *)
                echo "Unknown argument: $1" >&2
                show_usage
                exit 1
                ;;
        esac
    done
    
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
    
    # Build optional args for vLLM
    if [[ -n "$lora_spec" ]]; then
        # vLLM LoRA modules (comma-separated 'name=repo_or_path')
        lora_args+=( --lora-modules "$lora_spec" )
        # enable-lora and limits are added in run_vllm()
    fi
    if [[ "$enable_ep" -eq 1 ]]; then
        ep_args+=( --enable-expert-parallel )
    fi

    # Setup environment and cluster
    setup_environment
    check_gpu_availability
    echo "Getting cluster info..."
    get_cluster_info
    
    echo "RANK: $NODE_RANK. Starting Ray cluster..."
    run_ray_command "$model_path" "$api_key"
    
    if [[ "$NODE_RANK" == "0" ]]; then
        echo "RANK: 0. Starting VLLM server..."
        run_vllm "$model_path"
    fi
}

# Execute main function with all arguments
main "$@"
