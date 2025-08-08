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

module unload cuda || true
module unload nccl || true

module purge

module load cuda/12.6
module load nccl/2.24.3

srun bash -c '
  echo "[Before cleanup]"; env | grep -E "PYTHON|LD_|CUDA"
  
  # 明示的に不要な変数をunsetしてクリーンにする
  unset PYTHONPATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH=/home/appli/cuda/12.6/lib64:/home/appli/nccl/2.24.3/lib

  # 仮想環境の activate
  source ~/openr1/bin/activate
  echo "[After cleanup and activate]"; env | grep -E "PYTHON|LD_|CUDA"

  # 実行
  python -c "import deepspeed; print(deepspeed.__version__)"
'

# Take a look at the contents of the following environment variables first.
# PATH lists the locations of the executables and LD_LIBRARY_PATH lists where to look for shared libraries.
# Earlier entries are prioritized over later ones, and : is used to separate multiple entries.
# To find a specific CUDA toolkit, insert the correct path to list first.
# In addition, you should also check that the assigned directories actually exist.
# (https://huggingface.co/docs/transformers/debugging#deepspeed-cuda-issues)

export CUDA_HOME=/home/appli/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/targets/x86_64-linux/lib:/home/appli/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

source /home/Competition2025/P02/P02U017/openr1/bin/activate

################## デバッグチェック ###################

echo -e "\n🔍 [DEBUG] CUDA/NCCL 環境確認"

# nvcc の確認
echo -n "CUDA nvcc version: "
if ! which nvcc >/dev/null 2>&1; then
  echo "❌ nvcc が見つかりません (PATHに $CUDA_HOME/bin を含めたか確認)"
else
  nvcc --version | grep release
fi
echo -e "\n🔎 nvcc 候補一覧:"; which -a nvcc

# Python 環境確認
echo -n "🧪 Python: "; which python
python -c "import sys; print(f'Venv Prefix: {sys.prefix}')"

# PyTorch
python -c "import torch; print(f'Torch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"

# CUDA_HOME チェック
if [ ! -d "$CUDA_HOME" ]; then
  echo "❌ CUDA_HOME が見つかりません: $CUDA_HOME"
  exit 1
else
  echo "✅ CUDA_HOME OK: $CUDA_HOME"
fi

# libcudart.so チェック（findのみ使用、ヒットしたパスも表示）
echo -n "🔍 libcudart.so check: "
LIBCUDART_PATHS=$(find ${LD_LIBRARY_PATH//:/ } -name "libcudart.so*" 2>/dev/null)

if [ -n "$LIBCUDART_PATHS" ]; then
  echo "✅ found"
  echo "$LIBCUDART_PATHS" | sed 's/^/   └─ /'
else
  echo "❌ not found (LD_LIBRARY_PATHを再確認)"
fi

# NCCL チェック
if [ -f "/home/appli/nccl/2.24.3/lib/libnccl.so" ]; then
  echo "✅ NCCLライブラリ OK: libnccl.so found"
else
  echo "❌ NCCLライブラリが見つかりません"
fi

# 環境変数
echo -e "\n🧾 [ENV] PATH:"
echo $PATH | tr ':' '\n' | grep -E "cuda|nccl"

echo -e "\n🧾 [ENV] LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E "cuda|nccl"

# モジュール一覧
echo -e "\n📦 [Module List]"
module list 2>&1

# Deepspeed
python -c "import deepspeed; print(f'Deepspeed Version: {deepspeed.__version__}')"

echo -e "\n✅ 環境チェック完了 (続行可能)\n"

export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

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
       accelerate launch \\
         --config_file ../recipes/accelerate_configs/zero3.yaml \\
         --num_machines 3 \\
         --num_processes 24 \\
         --main_process_ip ${MAIN_IP} \\
         --main_process_port 29500 \\
         --rdzv_backend c10d \\
         --machine_rank \$SLURM_NODEID \\
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_235b.yaml \\
         --use_vllm true \\
         --vllm_mode colocate \\
         --report_to none
      "

wait
echo '[Job] all processes finished.'
