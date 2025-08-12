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
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1"
export NCCL_TIMEOUT=7200
export TORCHELASTIC_TIMEOUT=7200
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_ELASTIC_STORE_TIMEOUT=7200
export TORCH_DISTRIBUTED_STORE_TIMEOUT=7200
export RDZV_TIMEOUT=7200   # ←これが今 None なので追加
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_KEEPALIVE=1
export GLOO_SOCKET_KEEPALIVE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG_SUBSYS=INIT,NET

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

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  IF=$NET_IF"

echo "== [Pre-launch NCCL env] =="
env | grep NCCL

# --- 分散実行: 各ノード×8GPU（計24プロセス） ---
srun --ntasks=3 --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  --output=$LOG_ROOT/slurm-%t.out \
  --error=$LOG_ROOT/slurm-%t.err \
  bash -c "
  set -eo pipefail
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate deepseeksft310

  # ===== 既存ステージングを前提に pre-shard のみ作成 =====
  NVME_MNT=\"/nvme56\"
  LOCAL_MODEL=\"\$NVME_MNT/models/P02U006/DeepSeek-R1-0528-BF16\"
  LOCAL_SHARD=\"\$NVME_MNT/models/R1-0528-pre-sharded-pp3-ep8\"

  # 必須ファイルチェック（SentencePiece）
  REQ_FILES=(tokenizer.json tokenizer_config.json special_tokens_map.json config.json model.safetensors.index.json)
  for f in \"\${REQ_FILES[@]}\"; do
    [ -f \"\$LOCAL_MODEL/\$f\" ] || { echo \"[ERR] missing: \$LOCAL_MODEL/\$f\"; exit 1; }
  done
  # 重み分割ファイルがそれなりにあるか（最低1個だけ粗く確認）
  ls \"\$LOCAL_MODEL\"/model-*-of-*.safetensors >/dev/null 2>&1 || { echo \"[ERR] model shards not found\"; exit 1; }

  # 既にOKならスキップ（再実行に強い）
  if [ -f \"\$LOCAL_SHARD/.ok\" ]; then
    echo \"[INFO] pre-shard already exists: \$LOCAL_SHARD\"
    exit 0
  fi

  TMP_SHARD=\"\${LOCAL_SHARD}.tmp\"
  rm -rf \"\$TMP_SHARD\"

  # 分散設定を明示（外側の環境を引き継ぎつつ）
  export GLOO_SOCKET_IFNAME=\"$GLOO_SOCKET_IFNAME\"
  export NCCL_SOCKET_IFNAME=\"$NCCL_SOCKET_IFNAME\"
  export NCCL_IB_HCA=\"$NCCL_IB_HCA\"
  export MASTER_ADDR=$MASTER_ADDR
  export MASTER_PORT=$MASTER_PORT

  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=\$OMP_NUM_THREADS
  export OPENBLAS_NUM_THREADS=\$OMP_NUM_THREADS
  export NUMEXPR_NUM_THREADS=\$OMP_NUM_THREADS

  # 変換のみ（num_epochs=0, lora_rank=0）
  torchrun \
    --nnodes 3 \
    --node_rank \$SLURM_PROCID \
    --nproc_per_node 8 \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    /home/Competition2025/P02/P02U006/ColossalAI/applications/ColossalChat/examples/training_scripts/lora_finetune_preshard.py \
      --pretrained \"\$LOCAL_MODEL\" \
      --dataset /home/Competition2025/P02/shareP02/hci_colossalai_deepseekr10528_lorasft.jsonl \
      --plugin moe \
      --pp 3 --ep 8 \
      --batch_size 4 \
      --lr 2e-5 \
      --lora_rank 8 --lora_alpha 16 \
      --num_epochs 2 --warmup_steps 8 \
      --mixed_precision bf16 \
      --tensorboard_dir \"$LOG_ROOT/tb\" \
      --save_dir \"\$TMP_SHARD\"

  [ -d \"\$TMP_SHARD/modeling\" ] || { echo \"[ERR] pre-shard 作成に失敗 (modeling 不在)\"; exit 1; }
  rm -rf \"\$LOCAL_SHARD\"
  mv \"\$TMP_SHARD\" \"\$LOCAL_SHARD\"
  touch \"\$LOCAL_SHARD/.ok\"
  echo \"[OK] pre-shard created at: \$LOCAL_SHARD\"
"

echo "===== ジョブ終了: $(date) ====="