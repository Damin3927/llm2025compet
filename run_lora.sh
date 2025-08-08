#!/bin/bash
#SBATCH --partition=P02
#SBATCH --job-name=lora-r1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
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

# ログディレクトリ
BASE_LOG_DIR=/home/Competition2025/P02/P02U006/ColossalAI/logs
LOG_ROOT="${BASE_LOG_DIR}/${SLURM_JOB_ID}"
export LOG_ROOT
mkdir -p "${LOG_ROOT}" "${LOG_ROOT}/tb"

# Conda環境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepseeksft310

# CUDA toolchain
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/include:${CPATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL & PyTorch 分散
NET_IF="bond0"
export GLOO_SOCKET_IFNAME="$NET_IF"
export NCCL_SOCKET_IFNAME="$NET_IF" 
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_11:1"
export NCCL_TIMEOUT=7200
export TORCHELASTIC_TIMEOUT=7200
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_ELASTIC_STORE_TIMEOUT=7200
export TORCH_DISTRIBUTED_STORE_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ★ RoCE/IB での安定化（軽量チューニング）
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
unset NCCL_ASYNC_ERROR_HANDLING || true

export PYTHONFAULTHANDLER=1
ulimit -c unlimited
ulimit -v unlimited
ulimit -m unlimited

# SIGUSR1トラップ
trap 'echo "=== SIGUSR1 on $(hostname) ==="; pkill -USR1 -f lora_finetune.py' USR1

# GPU利用率ログ (各ノードごと)
MON_LOG="${LOG_ROOT}/gpu_${SLURM_NODEID}.log"

export FLASH_ATTENTION_DISABLE=1
export HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2=1
echo "FLASH_ATTENTION_DISABLE=$FLASH_ATTENTION_DISABLE"
echo "HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2=$HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2"

echo "=== CUDA環境 ==="
echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
nvcc --version || true

echo "=== Pythonライブラリのバージョン ==="
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import colossalai; print('colossalai', colossalai.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)" || true

echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CPATH=$CPATH"
echo "LIBRARY_PATH=$LIBRARY_PATH"

echo "[after unset NCCL_NET_PLUGIN]"; env | grep NCCL

# === MASTER を bond0 のIPに固定（★DNS揺れ回避） ===
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# マスターの bond0 IP を取得（SSH失敗時は getent にフォールバック）
MASTER_ADDR=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$MASTER_HOST" "ip -4 -o addr show $NET_IF | awk '{print \$4}' | cut -d/ -f1" 2>/dev/null || true)
if [[ -z "$MASTER_ADDR" ]]; then
  MASTER_ADDR=$(getent hosts "$MASTER_HOST" | awk '{print $1; exit}')
fi

# 分散学習のためのMASTER設定（SLURM_JOB_NODELISTから最初のノードを取得）
export MASTER_ADDR
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
echo "MASTER_HOST=$MASTER_HOST  MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  IF=$NET_IF"

echo "== [Pre-launch NCCL env] =="
env | grep NCCL


# --- 分散実行: 各ノード×8GPU（計24プロセス） ---
srun --ntasks=3 --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --output=$LOG_ROOT/slurm-%t.out \
  --error=$LOG_ROOT/slurm-%t.err \
  bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate deepseeksft310
    export GLOO_SOCKET_IFNAME=\"$GLOO_SOCKET_IFNAME\"
    export NCCL_SOCKET_IFNAME=\"$NCCL_SOCKET_IFNAME\"
    export NCCL_IB_HCA=\"$NCCL_IB_HCA\"
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    MON_LOG=\"${LOG_ROOT}/gpu_\$(hostname).log\"
    ( while true; do
        date '+[%F %T] ===== GPU util =====' >> \"\$MON_LOG\"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> \"\$MON_LOG\"
        sleep 60
      done ) &
    MON_PID=\$!

    torchrun \
      --nnodes 3 \
      --node_rank \$SLURM_PROCID \
      --nproc_per_node 8 \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      --rdzv-endpoint "$MASTER_ADDR:$MASTER_PORT" \
      --rdzv-conf join_timeout=7200,last_call_timeout=60,close_timeout=60 \
      --rdzv-backend c10d \
      --rdzv-id "lora-r1-${SLURM_JOB_ID}" \
      /home/Competition2025/P02/P02U006/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune.py \
        --pretrained /home/Competition2025/P02/shareP02/DeepSeek-R1-0528-BF16 \
        --dataset /home/Competition2025/P02/shareP02/hci_colossalai_deepseekr10528_lorasft.jsonl \
        --plugin moe \
        --pp 3 --ep 8 \
        --batch_size 8 \
        --lr 2e-5 \
        --max_length 8 \
        --lora_rank 8 --lora_alpha 16 \
        --num_epochs 2 --warmup_steps 8 \
        --mixed_precision bf16 \
        --use_grad_checkpoint \
        --tensorboard_dir $LOG_ROOT/tb \
        --save_dir $LOG_ROOT/DeepSeek-R1-0528-lora
    kill \"\$MON_PID\" || true
  "

echo "===== ジョブ終了: $(date) ====="