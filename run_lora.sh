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


set -exo pipefail                       # デバッグ

echo "===== ジョブ開始: $(date) ====="
echo "cwd  = $(pwd)"
echo "host = $(hostname)"
echo "JOB  = ${SLURM_JOB_ID}"
echo "NODES= ${SLURM_NODELIST}"

mkdir -p /home/Competition2025/P02/P02U006/ColossalAI/logs /home/Competition2025/P02/P02U006/ColossalAI/logs/tb


# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepseeksft310
which colossalai || true

# CUDA toolchain（今回は conda に入れたので CONDA_PREFIX を使う）
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/include:${CPATH:-}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # PyTorch の CUDA メモリ管理を有効化

# PyTorch 推奨の NCCL 非同期エラーハンドリング
#export NCCL_SOCKET_IFNAME=mlx5_0         # *各ノードで同じ名前を確認*
export NCCL_SOCKET_IFNAME="enp92s0np0" # 管理者の/etc/profile.d/でセットされるもに合わせる
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_11:1" # IBのポート指定も/etc/profile.d/appli.shでセットされているため
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600} # 1h まで待機
export TORCHELASTIC_TIMEOUT=3600 # 1h まで待機
export TORCH_DISTRIBUTED_TIMEOUT=3600 # 1h まで待機
export TORCH_ELASTIC_STORE_TIMEOUT=3600 # 1 hour
export TORCH_DISTRIBUTED_STORE_TIMEOUT=3600 # 1 hour
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH  # もっと欲しければ ALL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1     # 既定は3。1はコミュニケータ破棄+プロセス終了
unset NCCL_ASYNC_ERROR_HANDLING || true      # こちらは非推奨


export PYTHONFAULTHANDLER=1
ulimit -c unlimited # コアを残す
ulimit -v unlimited
ulimit -m unlimited


# ─── SIGUSR1 で全 rank に BT 取得 ────────────────────────────────
trap 'echo "=== SIGUSR1 on $(hostname) ==="; pkill -USR1 -f lora_finetune.py' USR1

# === ログ格納先を「/logs/<ジョブID>/」にする ======================
BASE_LOG_DIR=/home/Competition2025/P02/P02U006/ColossalAI/logs
LOG_ROOT="${BASE_LOG_DIR}/${SLURM_JOB_ID}"
export LOG_ROOT

# 既に存在していても OK（全ノードで race しても -p なので安全）
mkdir -p "${LOG_ROOT}"

# === GPU 利用率ログ =============================================
MON_LOG="${LOG_ROOT}/gpu_${SLURM_NODEID}.log"
(
  while true; do
    date '+[%F %T] ===== GPU util =====' >> "${MON_LOG}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> "${MON_LOG}"
    sleep 60
  done
) &
MON_PID=$!

# (conda activateの直後に書く)
export FLASH_ATTENTION_DISABLE=1
export HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2=1
echo "FLASH_ATTENTION_DISABLE=$FLASH_ATTENTION_DISABLE"
echo "HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2=$HF_TRANSFORMERS_CACHE_DISABLE_FLASH_ATTN_2"

# CUDA周辺の情報
echo "=== CUDA環境 ==="
echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
nvcc --version || true

# Pythonパッケージのバージョン確認
echo "=== Pythonライブラリのバージョン ==="
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import colossalai; print('colossalai', colossalai.__version__)"
# 必要なら transformers なども
python -c "import transformers; print('transformers', transformers.__version__)" || true

# 環境変数の確認
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "CPATH=$CPATH"
echo "LIBRARY_PATH=$LIBRARY_PATH"

# 管理者の/etc/profile.d/で必ずNCCL_NET_PLUGIN=noneがセットされるので、無理にunset不要
# unset NCCL_NET_PLUGIN
echo "[after unset NCCL_NET_PLUGIN]"; env | grep NCCL

# NCCL 環境変数の設定、お試し
#export NCCL_P2P_DISABLE=1
#export NCCL_P2P_LEVEL=NVL
#export NCCL_IB_GID_INDEX=3


# ───── ColossalAI 起動 ─────
#   * launcher 指定なし
#   * hostfile 指定あり
#   * master_port を固定
# ───────────────────────────
export MASTER_ADDR=$(head -n1 /home/Competition2025/P02/P02U006/ColossalAI/hostfile)
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

echo "== [Pre-launch NCCL env] =="
env | grep NCCL

# マスターノード1回だけ実行（他ノードへはSSHで展開）
srun -N1 -w "$MASTER_ADDR" --ntasks=1 bash -lc "
  set -e
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate deepseeksft310
  echo 'on master:' \$(hostname)
  echo '== [Pre-launch NCCL env (inside srun block)] =='
  env | grep NCCL
  which colossalai || true
  which python || true
  which torchrun || true

  TORCH_ELASTIC_STORE_TIMEOUT=3600 TORCH_DISTRIBUTED_STORE_TIMEOUT=3600 NCCL_TIMEOUT=3600 NCCL_DEBUG=INFO colossalai run \
    --hostfile /home/Competition2025/P02/P02U006/ColossalAI/hostfile \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node 8 \
    /home/Competition2025/P02/P02U006/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune.py \
    --pretrained /home/Competition2025/P02/shareP02/DeepSeek-R1-0528-BF16 \
    --dataset /home/Competition2025/P02/shareP02/hci_colossalai_deepseekr10528_lorasft.jsonl \
    --plugin moe \
    --pp 3 --ep 8 \
    --batch_size 8 \
    --lr 2e-5 \
    --max_length 32 \
    --lora_rank 8 --lora_alpha 16 \
    --num_epochs 2 --warmup_steps 8 \
    --mixed_precision bf16 \
    --use_grad_checkpoint \
    --tensorboard_dir /home/Competition2025/P02/P02U006/ColossalAI/logs/tb \
    --save_dir /home/Competition2025/P02/P02U006/ColossalAI/DeepSeek-R1-0528-lora
"

kill "$MON_PID" || true
echo "===== ジョブ終了: $(date) ====="