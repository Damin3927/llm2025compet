#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3                        # ★全3ノードをすべてトレーニングに使用
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen235b-a22b-fp8-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_a22b_fp8_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_a22b_fp8_colo.err

################### 環境 ###################
#export WANDB_DISABLED=true

export REQUIRE_FULL_PREFETCH=1

module unload cuda || true
module unload nccl || true

module purge

module load cuda/12.6
module load nccl/2.24.3

# これを最上部の Python プリフェッチの前に移動
source ~/openr1/bin/activate
python -c "import hf_transfer,sys; print('hf_transfer',hf_transfer.__version__,'@',sys.prefix)"

# ここが “ダウンロード元(正)” になります
export PREFETCH_DIR="$HOME/.cache/huggingface_mydir/models/Qwen3-235B-A22B"
export HF_HUB_ENABLE_HF_TRANSFER=0      # openr1 venv に hf_transfer が入っていればOK
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE  # 事前DLはオンライン必須

# ▼▼▼ この最終版スクリプトに差し替えてください ▼▼▼
python - <<'PY'
import os, json, sys, pathlib
from huggingface_hub import snapshot_download

repo = "Qwen/Qwen3-235B-A22B"
dest = os.environ["PREFETCH_DIR"]

print(f"[Prefetch] Starting download for {repo} -> {dest}")
try:
    snapshot_download(
        repo_id=repo,
        local_dir=dest,
        allow_patterns=[
            "config.json","*.json","*.txt","*.model","*tokenizer*",
            "model-*.safetensors","safetensors.index.json","pytorch_model.bin.index.json"
        ],
        max_workers=8,
    )
except Exception as e:
    print(f"[ABORT] snapshot_download failed with an exception: {e}", file=sys.stderr)
    sys.exit(43)

# --- 堅牢なチェック機構（ファイル名を修正済み） ---
# ★★★ 修正点 ★★★
index_path = pathlib.Path(dest) / "model.safetensors.index.json"
if not index_path.is_file():
    print(f"[ABORT] Verification failed: '{index_path.name}' not found in {dest}", file=sys.stderr)
    sys.exit(44)

with open(index_path) as f:
    index_data = json.load(f)

# set() を使ってユニークなファイル名のみをカウント
expected_shards = len(set(index_data["weight_map"].values()))
actual_shards = len(list(pathlib.Path(dest).glob("model-*.safetensors")))

print(f"Shard check: Found {actual_shards} shards, Expected {expected_shards} shards.")
if actual_shards != expected_shards:
    print(f"[ABORT] Shard count mismatch! Expected {expected_shards}, but found {actual_shards}.", file=sys.stderr)
    sys.exit(45)

print(f"✅ [Prefetch OK] All {actual_shards} shards and index file are present in {dest}")
PY

################### HFキャッシュをNVMeへ固定 + 事前チェック ###################
# 全ノードで共通してマウントされている /nvme56 を優先（無い場合は他候補にフォールバック）
export NVME_BASE=/nvme56
for _p in /nvme56 /nvme12 /nvme34 /nvme78 /nvme /local "$HOME"; do
  if mountpoint -q "$_p"; then NVME_BASE="$_p"; break; fi
done

export HF_HOME="$NVME_BASE/hf-home"
export HF_HUB_CACHE="$HF_HOME/hub"
# TRANSFORMERS_CACHE は非推奨なので使わない
unset TRANSFORMERS_CACHE

# モデルID（必要なら外から上書き可能）
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B}"
export MODEL_BASENAME="${MODEL_BASENAME:-Qwen3-235B-A22B}"

# 高速DL（可能ならONに）
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
# プリフェッチ済みの2回目以降だけ、学習直前に OFFLINE をONにする（※場所は後述）

# 保存先（NVMeなど早いローカル）
export WANDB_DIR="${NVME_BASE:-/tmp}/wandb"
srun --nodes=3 --ntasks-per-node=1 --gpus=0 -l bash -lc 'mkdir -p "'"$WANDB_DIR"'"'

# 🔄 オフライン保存に切り替え
unset WANDB_DISABLED
export WANDB_MODE=offline
# 送り先（チーム/プロジェクト）
export WANDB_ENTITY="neko-llm"
export WANDB_PROJECT="qwen235b-grpo"
# 余計な出力を抑える（任意）
export WANDB_CONSOLE=off
export WANDB_SILENT=true
export WANDB_DISABLE_CODE=true
export WANDB_START_METHOD=thread 
export WANDB_GROUP="job-${SLURM_JOB_ID}"

srun -N 3 -n 3 --ntasks-per-node=1 --gpus=0 -l bash -lc '
  test -w "'"$WANDB_DIR"'" \
    && echo "[W&B dir OK] $HOSTNAME -> '"$WANDB_DIR"'" \
    || { echo "[W&B dir NG] $HOSTNAME"; exit 46; }
'

# NVMe空き確認＆作成
export REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-800}  # 目安: 235Bのfp16 shardsで~470GB、余裕を持って
srun --gpus=0 --cpus-per-task=1 -l bash -lc '
  echo "[NVMe/HF] node=$HOSTNAME  NVME_BASE='$NVME_BASE'"
  FREE_GB=$(df -BG "'$NVME_BASE'" | awk "NR==2{gsub(\"G\",\"\",\$4); print \$4}")
  if (( FREE_GB < '$REQUIRED_FREE_GB' )); then
    echo "[ABORT] $HOSTNAME: free ${FREE_GB}GB < '$REQUIRED_FREE_GB'GB on '$NVME_BASE'" >&2
    exit 42
  fi
  mkdir -p "'$HF_HOME'/hub"
  echo "HF_HOME='$HF_HOME'  HF_HUB_CACHE='$HF_HUB_CACHE'"
'

################### 早期デバッグ（環境の見える化） ###################

srun --gpus=0 --cpus-per-task=1 --mem-per-cpu=8G bash -c '
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

################### CPU/NUMA 可視化 ###################

echo -e "\n🔎 [CPU TOPOLOGY per node]"
srun --gpus=0 --cpus-per-task=1 --kill-on-bad-exit=1 -l bash -lc '
  echo "node: $HOSTNAME"
  lscpu | egrep "CPU\\(s\\):|Socket|Core|Thread|Model name"
  echo "--- NUMA ---"
  numactl -H | sed -n "1,8p"
  echo "--- SLURM vars ---"
  echo SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
  echo SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE
  echo SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE
  echo
'

################### BAR1 プリフライト & 監視ログ ###################

export MIN_BAR1_FREE_MB=${MIN_BAR1_FREE_MB:-2048}    # 1GPUあたり最低2GBのBAR1空きを要求
# -q でBAR1セクションをパース（あなたの環境は --query でbar1不可のため）

srun --nodes=3 --ntasks-per-node=1 --gpus=0 --cpus-per-task=1 -l bash -lc '
  echo "[BAR1 preflight] node=$HOSTNAME"
  MIN_FREE='"$MIN_BAR1_FREE_MB"'
  ok=1

  # 1) まず query API を試す（数字だけ / GPUごと1行）
  if frees=$(nvidia-smi --query-gpu=bar1.memory.free --format=csv,noheader,nounits 2>/dev/null); then
    :
  else
    # 2) ドライバによっては query が無いので -q をパース
    frees=$(nvidia-smi -q | awk '"'"'
      /BAR1 Memory Usage/ {b=1; next}
      b && /FB Memory Usage/ {b=0}
      b && /Free/ {print $3}
    '"'"')
  fi

  echo "[BAR1 Free list] ${frees//$'\n'/ } (MiB)"
  while read -r f; do
    [[ "$f" =~ ^[0-9]+$ ]] || continue
    if (( f < MIN_FREE )); then
      echo "[ABORT] $HOSTNAME: BAR1 free ${f}MiB < ${MIN_FREE}MiB"
      ok=0
    fi
  done <<< "$frees"

  (( ok )) && echo "[OK] BAR1 free >= ${MIN_FREE}MiB on all GPUs" \
         || { echo "[WARN] BAR1 free < ${MIN_FREE}MiB on some GPUs (続行)"; :; }
'

# 起動前ワンショット（各ノードでVRAM/BAR1の現況だけ確認）
srun --nodes=3 --ntasks-per-node=1 --gpus=0 --cpus-per-task=1 -l bash -lc '
  echo "[VRAM one-shot] $(hostname)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  echo "[BAR1 one-shot] $(hostname)"
  nvidia-smi -q | sed -n "/BAR1 Memory Usage/,/FB Memory Usage/p" | sed -n "1,6p"
'

################### 通信・並列の実行時設定 ###################

export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=ALL

export NCCL_P2P_DISABLE=0          # P2P有効化（明示）
export NCCL_P2P_LEVEL=NVL          # NVLinkを優先的に使用
#export NCCL_IB_GID_INDEX=3         # IBネットワークの設定（Infiniband利用時）

ulimit -n 65536
ulimit -v unlimited
ulimit -m unlimited

################### レポ/コンフィグの存在確認（落下防止） ###################

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノードリスト取得 ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_IP="${NODELIST[0]}"

################### vLLM（コロケートモード） ###################
# ※別プロセスでのvLLMサーバ起動は不要（各Trainer内でvLLMエンジンを起動）

echo "[Node check]"
srun --ntasks=3 --nodes=3 --ntasks-per-node=1 --nodelist="$SLURM_JOB_NODELIST" --gpus=0 --kill-on-bad-exit=1 hostname

# プリフェッチ済みなら本番ではネット遮断推奨
# export HF_HUB_OFFLINE=1

# torch.compile を“完全”オフ（Dynamo/Inductor/JIT/NVFuser）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHINDUCTOR_MAX_WORKERS=1  
export PYTORCH_JIT=0
export NVFUSER_DISABLE=1

# 念のためキャッシュ先はNVMe（無害）
export TORCHINDUCTOR_CACHE_DIR="${NVME_BASE}/torchinductor-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# ===== 各ノードへ展開（PREFETCH_DIR → NVMe）=====  # ★コメント変更
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES -l bash -lc '
  set -euo pipefail
  SRC="'"$PREFETCH_DIR"'"
  if [[ -z "$SRC" || ! -d "$SRC" ]]; then
    echo "[ABORT] invalid SRC: ${SRC:-<empty>}"; exit 41
  fi
  # ★★★ 修正点: DSTをNVMe上のジョブ固有ディレクトリに変更 ★★★
  DST="${NVME_BASE}/slurm-job-${SLURM_JOB_ID}/Qwen3-235B-A22B"

  mkdir -p "$DST"
  rsync -aL --info=progress2 "$SRC"/ "$DST"/
  echo "[copied] $HOSTNAME -> $DST"
'

# ★★★ 修正点: MODEL_PATHを新しいDSTに合わせる ★★★
export MODEL_PATH="${NVME_BASE}/slurm-job-${SLURM_JOB_ID}/Qwen3-235B-A22B"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

################### GRPO Trainer（コロケートモードで実行） ###################
srun --nodes=3 --ntasks-per-node=1 --ntasks=3 \
     --kill-on-bad-exit=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --nodelist="$SLURM_JOB_NODELIST" \
     --cpu-bind=cores \
     --gpus=8 --exclusive --chdir="$REPO_DIR" \
     bash -lc "
       source /home/Competition2025/P02/P02U017/openr1/bin/activate
       if [[ \"\$SLURM_NODEID\" != \"0\" ]]; then export WANDB_DISABLED=true; fi
       HUB=\"${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}\"; HUB=\"${HUB:-$HOME/.cache/huggingface/hub}\"
       if ls -d \"$HUB/models--${MODEL_ID//\//--}/snapshots\"/* >/dev/null 2>&1; then
         export HF_HUB_OFFLINE=1
         echo \"[HF] offline mode ON (prefetched detected)\"
       fi

       export TORCH_COMPILE_DISABLE=1
       export TORCHDYNAMO_DISABLE=1
       export TORCHINDUCTOR_DISABLE=1
       export PYTORCH_JIT=0
       export NVFUSER_DISABLE=1

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
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo_235b_a22b_fp8.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_235b-a22b-fp8.yaml \\
         --model_name_or_path "$MODEL_PATH" \
         --use_vllm true \\
         --vllm_mode colocate \\
         --report_to wandb
      "

wait
echo '[Job] all processes finished.'