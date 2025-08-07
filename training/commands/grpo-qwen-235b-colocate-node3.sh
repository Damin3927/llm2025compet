#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3                        # ★全3ノードをすべてトレーニングに使用
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen235b-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.err

################### 環境 ###################
export WANDB_DISABLED=true
module load cuda/12.8
module load nccl/2.22.3 || true


export CUDA_HOME=/home/appli/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source /home/Competition2025/P02/P02U017/openr1/bin/activate

export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export NCCL_P2P_DISABLE=0          # P2P有効化（明示）
export NCCL_P2P_LEVEL=NVL          # NVLinkを優先的に使用
#export NCCL_IB_GID_INDEX=3         # IBネットワークの設定（Infiniband利用時）
#export NCCL_SOCKET_IFNAME=eth0     # 通信インターフェース（必要に応じて）

ulimit -v unlimited
ulimit -m unlimited

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノードリスト取得 ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_IP="${NODELIST[0]}"

################### vLLM（コロケートモード） ###################
# ※別プロセスでのvLLMサーバ起動は不要（各Trainer内でvLLMエンジンを起動）

################### GRPO Trainer（コロケートモードで実行） ###################
srun --ntasks=3 --nodelist="${NODELIST[*]}" \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" \
     bash -c "
       source /home/Competition2025/P02/P02U017/openr1/bin/activate
       echo \"[GRPO-Colo] on \$HOSTNAME  (node rank \$SLURM_NODEID, proc \$SLURM_PROCID)\"
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       export NCCL_ASYNC_ERROR_HANDLING=1
       # 3ノード・24GPUでコロケートモードによるGRPOトレーニング
       accelerate launch \\
         --config_file ../recipes/accelerate_configs/zero3.yaml \\
         --num_machines 3 \\
         --num_processes 24 \\
         --main_process_ip ${MAIN_IP} \\
         --main_process_port 29500 \\
         --rdzv_backend c10d \\
         --machine_rank \$SLURM_NODEID \\
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml \\
         --use_vllm true \\
         --vllm_mode colocate \\
         --enable_sleep_mode true

wait
echo '[Job] all processes finished.'
