#!/bin/bash

# VLLM SBATCH Script
# Submits a SLURM job for VLLM inference with configurable parameters

set -euo pipefail  # Exit on errors, undefined vars, and pipe failures

# =============================================================================
# Default Configuration
# =============================================================================

readonly DEFAULT_MODEL="Qwen/Qwen3-32B"
readonly DEFAULT_NODES=1
readonly DEFAULT_GPUS=2
readonly DEFAULT_NODELIST="osk-gpu54"
readonly DEFAULT_TIMEOUT="02:00:00"
readonly DEFAULT_PARTITION="P02"

# =============================================================================
# Functions
# =============================================================================

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --model <model>         Model name/path (default: $DEFAULT_MODEL)
    --nodes <number>        Number of nodes (default: $DEFAULT_NODES)
    --gpus <number>         Number of GPUs per node (default: $DEFAULT_GPUS)
    --nodelist <list>       SLURM nodelist (default: $DEFAULT_NODELIST)
    --timeout <time>        Job timeout in HH:MM:SS format (default: $DEFAULT_TIMEOUT)
    --partition <name>      SLURM partition (default: $DEFAULT_PARTITION)
    --api-key <key>         API key for vLLM server (optional)
    --help                  Show this help message

Examples:
    # Use defaults
    $0
    
    # Custom configuration
    $0 --model "Qwen/Qwen3-32B" --nodes 2 --gpus 4 --timeout "04:00:00" --nodelist "osk-gpu54,osk-gpu55"
EOF
}

download_model() {
    local model_path="$1"
    local node="${2:-$DEFAULT_NODELIST}"
    echo "Downloading model using srun on compute node..."

    srun \
        --job-name=download_model \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=32 \
        --nodelist="$node" \
        --partition="$DEFAULT_PARTITION" \
        --time=01:00:00 \
        --mem=32G \
        --pty bash -c "huggingface-cli download \"$model_path\""
}

generate_job_name() {
    local model="$1"
    local nodes="$2"
    local gpus="$3"
    
    # Extract model name from path (e.g., "Qwen/Qwen3-32B" -> "Qwen3-32B")
    local model_name=$(basename "$model")
    # Replace any problematic characters for SLURM job names
    model_name=$(echo "$model_name" | sed 's/[^a-zA-Z0-9_-]/_/g')
    
    echo "vllm-${model_name}-n${nodes}-g${gpus}"
}

generate_sbatch_script() {
    local model="$1"
    local nodes="$2"
    local gpus="$3"
    local nodelist="$4"
    local timeout="$5"
    local partition="$6"
    local api_key="$7"
    
    local job_name
    job_name=$(generate_job_name "$model" "$nodes" "$gpus")
    
    # Build vllm_start.sh arguments
    local vllm_args="--nnodes $nodes --ngpus $gpus --model \"$model\""
    if [[ -n "$api_key" ]]; then
        vllm_args+=" --api-key \"$api_key\""
    fi
    
    # Generate the SBATCH script content
    # NOTE: cpus-per-task がシビア。2*2 gpus の場合は 32 でうまくいったが、他の構成は未検証
    cat << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --nodes=$nodes
#SBATCH --gres=gpu:$gpus
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=$nodelist
#SBATCH --time=$timeout
#SBATCH --partition=$partition
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mem=0
#SBATCH --cpus-per-task=32

srun --ntasks=$nodes --gres=gpu:$gpus --exclusive ./vllm_start.sh $vllm_args
EOF
}

main() {
    # Initialize variables with defaults
    local model="$DEFAULT_MODEL"
    local nodes="$DEFAULT_NODES"
    local gpus="$DEFAULT_GPUS"
    local nodelist="$DEFAULT_NODELIST"
    local timeout="$DEFAULT_TIMEOUT"
    local partition="$DEFAULT_PARTITION"
    local api_key=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "${1}" in
            "--help"|"-h")
                show_usage
                exit 0
                ;;
            "--model")
                if [[ -n "${2:-}" ]]; then
                    model="$2"
                    shift 2
                else
                    echo "Error: --model requires a model path argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--nodes")
                if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
                    nodes="$2"
                    shift 2
                else
                    echo "Error: --nodes requires a positive integer argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--gpus")
                if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]+$ ]]; then
                    gpus="$2"
                    shift 2
                else
                    echo "Error: --gpus requires a positive integer argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--nodelist")
                if [[ -n "${2:-}" ]]; then
                    nodelist="$2"
                    shift 2
                else
                    echo "Error: --nodelist requires a nodelist argument" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--timeout")
                if [[ -n "${2:-}" ]] && [[ "${2}" =~ ^[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
                    timeout="$2"
                    shift 2
                else
                    echo "Error: --timeout requires a time in HH:MM:SS format" >&2
                    show_usage
                    exit 1
                fi
                ;;
            "--partition")
                if [[ -n "${2:-}" ]]; then
                    partition="$2"
                    shift 2
                else
                    echo "Error: --partition requires a partition name argument" >&2
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

    local head_node=$(scontrol show hostnames $nodelist | head -n 1)
    local head_node_ip=$(getent hosts "${head_node}gw" | awk '{print $1}')
    
    # Generate and display the configuration
    echo "=== VLLM SLURM Job Configuration ==="
    echo "Model: $model"
    echo "Nodes: $nodes"
    echo "GPUs per node: $gpus"
    echo "Nodelist: $nodelist"
    echo "Timeout: $timeout"
    echo "Partition: $partition"
    if [[ -n "$api_key" ]]; then
        echo "API Key: [REDACTED]"
    fi
    echo "Job name: $(generate_job_name "$model" "$nodes" "$gpus")"
    echo

    download_model "$model" "$head_node"
    
    # Generate and submit the job
    echo "Generating SBATCH script..."
    mkdir -p logs
    local sbatch_content
    sbatch_content=$(generate_sbatch_script "$model" "$nodes" "$gpus" "$nodelist" "$timeout" "$partition" "$api_key")
    
    echo "Submitting job to SLURM..."
    echo "$sbatch_content" | sbatch
    
    if [[ $? -eq 0 ]]; then
        echo "Job submitted successfully!"
        echo "Monitor with: squeue -u \$USER"
        echo "Cancel with: scancel <job_id>"
        echo "==================================="
        echo "Head node's IP: $head_node_ip"
        echo "You can connect to the server using OpenAI API Schema at http://$head_node_ip:8000/v1 after the server is up."
        echo "Check logs in the logs/vllm-<job_id>.out file to monitor the server status. (After seeing 'Starting vLLM API server on http://0.0.0.0:8000' in the logs, you can start sending requests.)"
        echo "===================================="
        echo "To view job logs, use: tail -f logs/<job_name>-<job_id>.out"
        echo "To view job errors, use: tail -f logs/<job_name>-<job_id>.err"
    else
        echo "Error: Failed to submit job" >&2
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"
