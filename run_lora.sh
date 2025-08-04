#!/bin/bash
#SBATCH --partition=P02
#SBATCH --job-name=lora-r1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/Competition2025/P02/P02U006/ColossalAI/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/P02U006/ColossalAI/logs/%x-%j.err

set -exo pipefail

echo "===== ジョブ開始: $(date) ====="
echo "cwd  = $(pwd)"
echo "host = $(hostname)"
echo "JOB  = ${SLURM_JOB_ID}"
echo "NODES= ${SLURM_NODELIST}"

mkdir -p /home/Competition2025/P02/P02U006/ColossalAI/logs /home/Competition2025/P02/P02U006/ColossalAI/logs/tb

# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepseeksft310

# CUDA, NCCL, 環境変数は今まで通り
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="enp25s0np0,enp41s0np0,enp59s0np0,enp92s0np0,enp155s0np0,enp170s0np0,enp187s0np0,enp218s0np0"
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_11:1"
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-600}
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export PYTHONFAULTHANDLER=1
ulimit -c unlimited
ulimit -v unlimited
ulimit -m unlimited

# GPU監視
MON_LOG="/home/Competition2025/P02/P02U006/ColossalAI/logs/gpu_${SLURM_NODEID}.log"
(
  while true; do
    date '+[%F %T] ===== GPU util =====' >> "$MON_LOG"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> "$MON_LOG"
    sleep 60
  done
) &
MON_PID=$!

export FLASH_ATTENTION_DISABLE=1
export HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2=1

echo "=== CUDA環境 ==="
nvcc --version || true

echo "=== Pythonライブラリのバージョン ==="
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import colossalai; print('colossalai', colossalai.__version__)" || true

echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# masterアドレスとポートの決定（SLURM提供の変数を使う。複数ノードで問題ない）
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# ここからrun部分が本番
srun --ntasks=$(( SLURM_NNODES * SLURM_GPUS_ON_NODE )) \
     --ntasks-per-node=$SLURM_GPUS_ON_NODE \
     --gpus-per-node=$SLURM_GPUS_ON_NODE \
     --cpu-bind=none \
python -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /home/Competition2025/P02/P02U006/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune.py \
        --pretrained /home/Competition2025/P02/shareP02/DeepSeek-R1-0528-BF16 \
        --dataset /home/Competition2025/P02/shareP02/hci_colossalai_deepseekr10528_lorasft.jsonl \
        --plugin moe \
        --pp 3 --ep 8 \
        --batch_size 8 \
        --lr 2e-5 \
        --max_length 256 \
        --lora_rank 8 --lora_alpha 16 \
        --num_epochs 2 --warmup_steps 8 \
        --mixed_precision bf16 \
        --use_grad_checkpoint \
        --tensorboard_dir /home/Competition2025/P02/P02U006/ColossalAI/logs/tb \
        --save_dir /home/Competition2025/P02/P02U006/ColossalAI/DeepSeek-R1-0528-lora


kill "$MON_PID" || true
echo "===== ジョブ終了: $(date) ====="
