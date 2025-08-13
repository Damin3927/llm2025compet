#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen235b-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.err
set -euo pipefail

################### 早期・環境サニタイズ ###################
# “ユーザーサイト”を見ない（~/.local等の異物混入＆過剰importを防ぐ）
export PYTHONNOUSERSITE=1
# Torchのコンパイル系を最初から完全OFF（初回import時に効くよう先頭で）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHINDUCTOR_MAX_WORKERS=1
export PYTORCH_JIT=0
export NVFUSER_DISABLE=1

# === dynamic topology (computed early) ===
NODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
TPN=1

################### 環境 ###################
# >> TMPDIR moved before prefetch (auto)

export NVME_BASE=/nvme12

# 親ディレクトリだけ先に用意（ここはそのままでOK）
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc \
  'mkdir -p "/nvme12/tmp/$USER" && echo "[precreate] $HOSTNAME prepared /nvme12/tmp/$USER"'

# 以降は NVMe の TMPDIR（ジョブ固有）を使う
export JOB_TMPDIR="$NVME_BASE/tmp/$USER/job-$SLURM_JOB_ID"
# 各ノードでジョブ専用ディレクトリを作成（親は既存どおり /nvme12/tmp/$USER）
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 --export=ALL,TMPDIR=/tmp -l bash -lc \
  'mkdir -m 700 -p "'"$JOB_TMPDIR"'" && echo "[TMPDIR] $HOSTNAME prepared '"$JOB_TMPDIR"'"'
export TMPDIR="$JOB_TMPDIR" 
export TMP="$JOB_TMPDIR"; export TEMP="$JOB_TMPDIR"

module unload cuda || true
module unload nccl || true
module purge
module load cuda/12.6
module load nccl/2.24.3

# venvを有効化（prefetch前に）
source ~/openr1/bin/activate
python - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec("hf_transfer")
print("hf_transfer:", "present" if spec else "absent", "@", sys.prefix)
PY

# ★追加: bitsandbytes が GPU 対応か事前チェック
python - <<'PY'
import sys, os, glob
try:
    import bitsandbytes as bnb
    import os.path as op
    pkg_dir = op.dirname(bnb.__file__)
    gpu_libs = glob.glob(op.join(pkg_dir, "libbitsandbytes_cuda*.so"))
    if not gpu_libs:
        print(f"[ABORT] bitsandbytes has no GPU CUDA libs under {pkg_dir}", file=sys.stderr)
        print("        4bit 量子化ロードは失敗します。bnbのGPU版を入れ直してください。", file=sys.stderr)
        sys.exit(47)
    else:
        print(f"[OK] bitsandbytes GPU libs: {os.path.basename(gpu_libs[0])} ...")
except Exception as e:
    print(f"[ABORT] bitsandbytes import failed: {e}", file=sys.stderr)
    sys.exit(48)
PY


# ここが “ダウンロード元”
export PREFETCH_DIR="$HOME/.cache/huggingface_mydir/models/Qwen3-235B-A22B"
# LoRA を vLLM と学習側で共有する置き場（NFS/共有FS）
export LORA_CACHE_BASE="$HOME/.cache/huggingface_mydir/loras"
export ADAPTER_NAME="qwen235b-grpo"
export ADAPTER_DIR="$LORA_CACHE_BASE/$ADAPTER_NAME"           # vLLM が参照する“公開”ディレクトリ
export ADAPTER_TMP="$LORA_CACHE_BASE/${ADAPTER_NAME}.tmp"     # 学習側が一時出力→atomic rename で公開
mkdir -p "$LORA_CACHE_BASE"


export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

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
    print(f"[ABORT] snapshot_download failed: {e}", file=sys.stderr)
    sys.exit(43)

index_path = pathlib.Path(dest) / "model.safetensors.index.json"
if not index_path.is_file():
    print(f"[ABORT] Verification failed: '{index_path.name}' not found in {dest}", file=sys.stderr)
    sys.exit(44)

with open(index_path) as f:
    index_data = json.load(f)
expected_shards = len(set(index_data["weight_map"].values()))
actual_shards = len(list(pathlib.Path(dest).glob("model-*.safetensors")))
print(f"Shard check: Found {actual_shards} shards, Expected {expected_shards} shards.")
if actual_shards != expected_shards:
    print(f"[ABORT] Shard count mismatch! Expected {expected_shards}, but found {actual_shards}.", file=sys.stderr)
    sys.exit(45)
print(f"✅ [Prefetch OK] All {actual_shards} shards and index file are present in {dest}")
PY

################### HFキャッシュをNVMeへ固定 + 事前チェック ###################
export HF_HOME="$NVME_BASE/hf-cache-$USER"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
unset TRANSFORMERS_CACHE

# モデルID
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-235B-A22B}"
export MODEL_BASENAME="${MODEL_BASENAME:-Qwen3-235B-A22B}"

# W&B
export WANDB_DIR="${NVME_BASE:-/tmp}/wandb"
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc 'mkdir -p "'"$WANDB_DIR"'"'

unset WANDB_DISABLED
export WANDB_MODE=offline
export WANDB_ENTITY="neko-llm"
export WANDB_PROJECT="qwen235b-grpo"
export WANDB_CONSOLE=off
export WANDB_SILENT=true
export WANDB_DISABLE_CODE=true
export WANDB_START_METHOD=thread
export WANDB_GROUP="job-${SLURM_JOB_ID}"

srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc '
  test -w "'"$WANDB_DIR"'" \
    && echo "[W&B dir OK] $HOSTNAME -> '"$WANDB_DIR"'" \
    || { echo "[W&B dir NG] $HOSTNAME"; exit 46; }
'

# NVMe空き確認
export REQUIRED_FREE_GB=${REQUIRED_FREE_GB:-800}
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

# 各ノードに作成
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc \
  "mkdir -p \"$JOB_TMPDIR\" \"$HF_HUB_CACHE\" && \
   echo \"[\$HOSTNAME] TMPDIR=$JOB_TMPDIR  HF_HOME=$HF_HOME\""

################### 早期デバッグ（環境の見える化：1回だけ実行） ###################
# ★ ここは重いimportがあるので“1タスクのみ”で実行
srun -N 1 -n 1 --gpus=0 --cpus-per-task=1 --mem=8G -l bash -lc '
  echo "[Before cleanup]"; env | grep -E "PYTHON|LD_|CUDA"
  unset PYTHONPATH
  unset LD_PRELOAD
  export LD_LIBRARY_PATH=/home/appli/cuda/12.6/lib64:/home/appli/nccl/2.24.3/lib
  source ~/openr1/bin/activate
  echo "[After cleanup and activate]"; env | grep -E "PYTHON|LD_|CUDA"

  # Deepspeedのバージョンは importlib.metadata で取得（torchを引き込まない）
  python - <<PY
from importlib.metadata import version, PackageNotFoundError
try:
    print("DeepSpeed Version:", version("deepspeed"))
except PackageNotFoundError:
    print("DeepSpeed Version: <not found>")
PY
'

# Take a look at the contents of the following environment variables first.
# PATH lists the locations of the executables and LD_LIBRARY_PATH lists where to look for shared libraries.
# Earlier entries are prioritized over later ones, and : is used to separate multiple entries.
# To find a specific CUDA toolkit, insert the correct path to list first.
# In addition, you should also check that the assigned directories actually exist.
# (https://huggingface.co/docs/transformers/debugging#deepspeed-cuda-issues)

# CUDA toolchain パス
export CUDA_HOME=/home/appli/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/targets/x86_64-linux/lib:/home/appli/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

source /home/Competition2025/P02/P02U017/openr1/bin/activate

################## デバッグチェック（これも1回だけ） ###################
echo -e "\n🔍 [DEBUG] CUDA/NCCL 環境確認"

echo -n "CUDA nvcc version: "
if ! which nvcc >/dev/null 2>&1; then
  echo "❌ nvcc not found"
else
  nvcc --version | grep release
fi
echo -e "\n🔎 nvcc 候補一覧:"; which -a nvcc || true

echo -n "🧪 Python: "; which python
python - <<'PY'
import sys, torch
print(f'Venv Prefix: {sys.prefix}')
print(f'Torch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')
PY

if [ ! -d "$CUDA_HOME" ]; then
  echo "❌ CUDA_HOME not found: $CUDA_HOME"; exit 1
else
  echo "✅ CUDA_HOME OK: $CUDA_HOME"
fi

echo -n "🔍 libcudart.so check: "
LIBCUDART_PATHS=$(find ${LD_LIBRARY_PATH//:/ } -name "libcudart.so*" 2>/dev/null)
if [ -n "$LIBCUDART_PATHS" ]; then
  echo "✅ found"; echo "$LIBCUDART_PATHS" | sed 's/^/   └─ /'
else
  echo "❌ not found"
fi

if [ -f "/home/appli/nccl/2.24.3/lib/libnccl.so" ]; then
  echo "✅ NCCLライブラリ OK"
else
  echo "❌ NCCLライブラリが見つかりません"
fi

echo -e "\n🧾 [ENV] PATH:"; echo $PATH | tr ':' '\n' | grep -E "cuda|nccl" || true
echo -e "\n🧾 [ENV] LD_LIBRARY_PATH:"; echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -E "cuda|nccl" || true

echo -e "\n📦 [Module List]"
module list 2>&1 || true

echo -e "\n✅ 環境チェック完了 (続行可能)\n"

################### CPU/NUMA 可視化（各ノード1回） ###################
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
export MIN_BAR1_FREE_MB=${MIN_BAR1_FREE_MB:-2048}

srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc '
  echo "[BAR1 preflight] node=$HOSTNAME"
  MIN_FREE='"${MIN_BAR1_FREE_MB:-2048}"'
  ok=1
  frees_q=$(nvidia-smi --query-gpu=bar1.memory.free --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -z "$frees_q" ] || echo "$frees_q" | tr -d "\n" | grep -Eq "[^0-9[:space:]]"; then
    frees=$(nvidia-smi -q | awk '"'"'/BAR1 Memory Usage/ {b=1; next} b && /FB Memory Usage/ {b=0} b && /Free/ {print $3}'"'"')
  else
    frees="$frees_q"
  fi
  echo "[BAR1 Free list] ${frees//$'\n'/ } (MiB)"
  while read -r f; do
    [[ "$f" =~ ^[0-9]+$ ]] || continue;
    if (( f == 0 )); then
      echo "[SKIP] $HOSTNAME: BAR1 free reported 0MiB (transient); ignoring";
      continue;
    fi;
    if (( f < MIN_FREE )); then
      echo "[WARN] $HOSTNAME: BAR1 free ${f}MiB < ${MIN_FREE}MiB"; ok=0;
    fi;
  done <<< "$frees"
  (( ok )) && echo "[OK] BAR1 free >= ${MIN_FREE}MiB on all GPUs" || echo "[WARN] BAR1 threshold not met (続行可)"
'

# 起動前ワンショット
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 --cpus-per-task=1 -l bash -lc '
  echo "[VRAM one-shot] $(hostname)"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
  echo "[BAR1 one-shot] $(hostname)"
  nvidia-smi -q | sed -n "/BAR1 Memory Usage/,/FB Memory Usage/p" | sed -n "1,6p"
'

# === NVMe persistent model cache ===
export NVME_PERSIST_DIR="${NVME_BASE}/persist-${USER}/${MODEL_BASENAME}"

srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc '
  set -euo pipefail
  SRC="'"$PREFETCH_DIR"'"
  DST="'"$NVME_PERSIST_DIR"'"
  mkdir -p "$DST"

  # 期待シャード数（SRC優先 → index解析 → フォールバック）
  exp=$( ( ls -1 "$SRC"/model-*.safetensors 2>/dev/null || true ) | wc -l )
  if [ "${exp:-0}" -eq 0 ] && [ -f "$SRC/model.safetensors.index.json" ]; then
    exp=$(python - "$SRC/model.safetensors.index.json" <<'"'PY'"'
import sys,json
j=json.load(open(sys.argv[1]))
print(len(j.get("weight_map", {})))
PY
)
  fi
  : "${exp:=118}"  # 最後の保険

  # 既存数をカウント（必ず単一整数）
  cur=$( ( ls -1 "$DST"/model-*.safetensors 2>/dev/null || true ) | wc -l )

  if [ ! -f "$DST/.ready" ] || [ "$cur" -lt "$exp" ] || [ ! -f "$DST/model.safetensors.index.json" ]; then
    echo "[persist] $HOSTNAME: populate $DST from $SRC (cur=$cur, exp=$exp)"
    numactl --interleave=all rsync -aL --delete --info=progress2 --temp-dir="$TMPDIR" "$SRC"/ "$DST"/
    # 完全性チェック → ready マーク
    new=$( ( ls -1 "$DST"/model-*.safetensors 2>/dev/null || true ) | wc -l )
    if [ "$new" -ge "$exp" ] && [ -f "$DST/model.safetensors.index.json" ]; then
      touch "$DST/.ready"
      chmod -R a-w "$DST" || true
      echo "[persist] $HOSTNAME: ready ($new/$exp)"
    else
      echo "[persist] $HOSTNAME: partial after sync ($new/$exp)"
    fi
  else
    echo "[persist] $HOSTNAME: reuse $DST ($cur/$exp)"
  fi
'

# 以降は NVMe の永続キャッシュを直接参照
export MODEL_PATH="$NVME_PERSIST_DIR"
# === wait persist cache ===
exp=$( ( ls -1 "$NVME_PERSIST_DIR"/model-*.safetensors 2>/dev/null || true ) | wc -l )
if [ "${exp:-0}" -lt 2 ] && [ -f "$NVME_PERSIST_DIR/model.safetensors.index.json" ]; then
  exp=$(python - "$NVME_PERSIST_DIR/model.safetensors.index.json" <<'PY'
import sys,json; j=json.load(open(sys.argv[1]));
print(len(set(j.get("weight_map",{}).values())))
PY
)
fi
: "${exp:=118}"  # フォールバック
while :; do
  cur=$(ls -1 "$NVME_PERSIST_DIR"/model-*.safetensors 2>/dev/null | wc -l)
  if [ "$cur" -ge "$exp" ] && [ -f "$NVME_PERSIST_DIR/model.safetensors.index.json" ]; then
    break
  fi
  echo "[WAIT] persist cache $cur/$exp"; sleep 20
done

################### 通信・並列の実行時設定 ###################
export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1"
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200

ulimit -n 65536
ulimit -v unlimited
ulimit -m unlimited

################### レポ/コンフィグの存在確認 ###################
REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノードリスト取得 ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_IP="${NODELIST[0]}"

ROLLOUT_NODE="${NODELIST[2]}"          # 例: osk-gpu91
TRAIN_NODE_A="${NODELIST[0]}"          # 例: osk-gpu54
TRAIN_NODE_B="${NODELIST[1]}"          # 例: osk-gpu56

# vLLM サーバを動かすノード（ロールアウト専用）
export VLLM_NODE="${VLLM_NODE:-$ROLLOUT_NODE}"

# vLLM の Filesystem LoRA Resolver のベースパス（name 配下を見る）
export VLLM_LORA_RESOLVER_CACHE_DIR="$LORA_CACHE_BASE"

echo "[Node check]"
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --nodelist="$SLURM_JOB_NODELIST" --gpus=0 --kill-on-bad-exit=1 hostname

# Torch compile 全OFF（再掲：保険）
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHINDUCTOR_MAX_WORKERS=1
export PYTORCH_JIT=0
export NVFUSER_DISABLE=1
export TORCHINDUCTOR_CACHE_DIR="${NVME_BASE}/torchinductor-cache"; mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# ===== 各ノードへ展開（PREFETCH_DIR → NVMe）=====
#srun --ntasks-per-node=1 -n "$NODES" -l bash -lc '
#  SRC="'"$PREFETCH_DIR"'"
#  if [[ -z "$SRC" || ! -d "$SRC" ]]; then
#    echo "[ABORT] invalid SRC: ${SRC:-<empty>}"; exit 41
#  fi
#  DST="${NVME_BASE}/slurm-job-${SLURM_JOB_ID}/Qwen3-235B-A22B"
#  mkdir -p "$DST"
#  numactl --interleave=all rsync -aL --info=progress2 "$SRC"/ "$DST"/
#  echo "[copied] $HOSTNAME -> $DST"
#'

export MODEL_PATH="$NVME_PERSIST_DIR"
# export HF_HUB_OFFLINE=1 (using local MODEL_PATH)
# export TRANSFORMERS_OFFLINE=1 (using local MODEL_PATH)

################### NUMA/バインド デバッグ関数 ###################
export DEBUG_NUMA=${DEBUG_NUMA:-1}
export NUMA_SNAPSHOT_DELAY=${NUMA_SNAPSHOT_DELAY:-20}
export NUMA_SNAPSHOT_REPEAT=${NUMA_SNAPSHOT_REPEAT:-6}
export NUMA_SNAPSHOT_INTERVAL=${NUMA_SNAPSHOT_INTERVAL:-60}

debug_header () { echo -e "\n\033[1m[NUMA-DEBUG]\033[0m $*"; }
numa_snapshot () {
  local tag="$1"
  debug_header "Snapshot($tag): CPUバインド / メモリ局在（numastat -p）"
  srun -N "$NODES" -n "$NODES" --gpus=0 -l bash -lc '
    echo ">>> NODE=$HOSTNAME"
    if command -v scontrol >/dev/null 2>&1; then
      PIDS=$(scontrol listpids "$SLURM_JOB_ID" 2>/dev/null | awk "{print \$2}" | sort -u)
    else
      PIDS=$(pgrep -u "$USER" -f "open_r1/grpo.py|accelerate|vllm|python" || true)
    fi
    if [ -z "$PIDS" ]; then echo "  (no PIDs found yet)"; exit 0; fi
    for p in $PIDS; do
      [ -d /proc/$p ] || continue
      [ -d /proc/$p ] || continue
      COMM=$(tr -d "\0" </proc/$p/comm 2>/dev/null || echo "?")
      ALLOWED=$(awk -F: '"'"'/Cpus_allowed_list/ {print $2}'"'"' /proc/$p/status 2>/dev/null | xargs)
      echo "  PID $p [$COMM]  CPUs=$ALLOWED"
      numactl -p "$p" 2>/dev/null | egrep "policy|membind|cpubind" || true
      numastat -p "$p" 2>/dev/null | sed -n "1,7p" || true
      echo
    done
  '
}

if (( DEBUG_NUMA )); then
  debug_header "Preflight: タスク配分/トポロジ/NUMA概要"
  srun -N "$NODES" -n "$NODES" --gpus=0 -l bash -lc '
    echo "=== $HOSTNAME ==="
    echo "[lscpu brief]"; lscpu | egrep "CPU\\(s\\)|Socket|Core|Thread" || true
    echo "[numactl -H head]"; numactl -H | sed -n "1,12p" || true
    echo "[nvidia-smi topo -m head]"; nvidia-smi topo -m | sed -n "1,25p" || true
  ' || true
srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 --gpus=0 -l bash -lc 'echo "$(hostname) cpuset:"; taskset -pc $$; numactl -s | egrep "policy|membind|cpubind"'
  numa_snapshot "pre-launch"
fi

################### DeepSpeed 設定を配置 ###################
DS_CFG_DIR="$REPO_DIR/../recipes/deepspeed"
DS_CFG="$DS_CFG_DIR/ds_zero3.json"
mkdir -p "$DS_CFG_DIR" || { echo "[ABORT] cannot mkdir $DS_CFG_DIR"; exit 60; }
cat > "$DS_CFG" <<'JSON'
{
  "deepspeed_multinode_launcher": "standard",
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "zero3_init_flag": true,
  "zero3_save_16bit_model": false,
  "offload_optimizer": {"device": "cpu"},
  "offload_param": {"device": "none"},
  "bf16": {"enabled": true}
}
JSON

python - "$DS_CFG" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
print('[DS JSON Check] bf16.enabled=', j.get('bf16',{}).get('enabled'), 'zero_stage=', j.get('zero_optimization',{}).get('stage'))
PY

################### デバッグラン切替 ###################
export DEBUG_RUN=${DEBUG_RUN:-1}   # 本番は 0 を渡す
echo "[RUNMODE] DEBUG_RUN=${DEBUG_RUN:-<unset>}"
# Tritonキャッシュ（必要なら）
#export TRITON_CACHE_DIR="${NVME_BASE}/triton-cache-$USER"; mkdir -p "$TRITON_CACHE_DIR"

if (( DEBUG_RUN )); then
  # デバッグでは vLLM を無効化して軽量スモーク
  export GRPO_DEBUG_FLAGS="--max_steps 12 --logging_steps 1 --save_strategy no \
                           --evaluation_strategy no --num_generations 2 \
                           --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
                           "
else
  export GRPO_DEBUG_FLAGS=""
fi

export MASTER_PORT=${MASTER_PORT:-$((12000 + SLURM_JOB_ID % 20000))}
export VLLM_TP=${VLLM_TP:-8}
#export TRAIN_DEVICE_INDEX=${TRAIN_DEVICE_INDEX:-3}

srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" --gpus=0 -l bash -lc '
  if ss -ltn "( sport = :8000 )" | grep -q LISTEN; then
    echo "[ABORT] :8000 already in use on $HOSTNAME" >&2; exit 61
  fi
'

################### vLLM Server ###################
# 初回 LoRA が既にある場合は事前ロードフラグを作る
PRELOAD_LORA_FLAG=""
if [ -d "$ADAPTER_DIR" ]; then
  PRELOAD_LORA_FLAG="--lora-modules '{\"name\":\"$ADAPTER_NAME\",\"path\":\"$ADAPTER_DIR\"}'"
fi

# 推奨: 余裕を持たせる（必要なら 0.95 まで上げてOK）
GPU_UTIL="${GPU_UTIL:-0.92}"
# enforce-eager は外す（CUDA Graph 有効化で安定＆高速化）
ENFORCE_EAGER_FLAG=""
# PRELOAD_LORA_FLAG は既存ロジックのまま
VLLM_LOG="${REPO_DIR}/../logs/vllm_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$VLLM_LOG")"

srun --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" \
     --export=ALL,TMPDIR="$JOB_TMPDIR" \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" \
     bash -lc "
       set -euo pipefail
       source ~/openr1/bin/activate
       echo '[vLLM] launching on \$HOSTNAME (TP=8, BF16, util=${GPU_UTIL})'
       CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
       python -m vllm.entrypoints.openai.api_server \
         --model \"$MODEL_PATH\" \
         --served-model-name qwen235b-base \
         --tensor-parallel-size 8 \
         --dtype bfloat16 \
         --enable-lora --max-loras 4 --lora-dtype bfloat16 \
         $PRELOAD_LORA_FLAG \
         --host 0.0.0.0 --port 8000 \
         --gpu-memory-utilization ${GPU_UTIL} \
         --max-parallel-loading-workers 1 \
         --disable-uvicorn-access-log \
         2>&1 | tee \"$VLLM_LOG\"
     " &
VLLM_SRUN_PID=$!

# ヘルス待機（落ちたら直ちにログ末尾を出して中断）
echo "[vLLM] waiting for http://$VLLM_NODE:8000/health (timeout=1800s) ..."
for i in $(seq 1 1800); do
  if ! kill -0 "$VLLM_SRUN_PID" 2>/dev/null; then
    echo "[ABORT] vLLM process died before becoming healthy. Last 80 lines:"
    tail -n 80 "$VLLM_LOG" || true
    exit 50
  fi
  if curl -fsS "http://$VLLM_NODE:8000/health" >/dev/null 2>&1; then
    echo "[vLLM] healthy"
    break
  fi
  sleep 1
  if [ "$i" -eq 1800 ]; then
    echo "[ABORT] vLLM health timeout. Last 80 lines:"
    tail -n 80 "$VLLM_LOG" || true
    exit 50
  fi
done


# LoRA エンドポイント疎通（存在すればロード→一覧で確認→アンロード）
if [ -d "$ADAPTER_DIR" ]; then
  echo "[LoRA] load attempt: $ADAPTER_NAME -> $ADAPTER_DIR"
  curl -sS -X POST "http://$VLLM_NODE:8000/v1/load_lora_adapter" \
       -H "Content-Type: application/json" \
       -d "{\"lora_name\":\"$ADAPTER_NAME\",\"lora_path\":\"$ADAPTER_DIR\"}" || true
  echo "[LoRA] models after load:"; curl -sS "http://$VLLM_NODE:8000/v1/models" | jq -r '.data[].id' || true

  echo "[LoRA] unload attempt: $ADAPTER_NAME"
  curl -sS -X POST "http://$VLLM_NODE:8000/v1/unload_lora_adapter" \
       -H "Content-Type: application/json" \
       -d "{\"lora_name\":\"$ADAPTER_NAME\"}" || true
  echo "[LoRA] models after unload:"; curl -sS "http://$VLLM_NODE:8000/v1/models" | jq -r '.data[].id' || true
fi

# 任意: LoRA スワップ健康診断（RUN_SWAP_SMOKE=1 の時だけ実行）
if [[ "${RUN_SWAP_SMOKE:-0}" == "1" ]]; then
  TEST_SH="/home/Competition2025/P02/P02U017/llm2025compet/training/commands/measure_lora_swap.sh"
  if [[ -x "$TEST_SH" ]]; then
    "$TEST_SH" || true
  else
    echo "[WARN] $TEST_SH not found or not executable; skip swap smoke"
  fi
fi

echo "[Smoke] single chat completion (base)"
curl -sS "http://$VLLM_NODE:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen235b-base",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}],
    "max_tokens": 16
  }' | jq -r '.choices[0].message.content' || true

# ローカルMODEL_PATH直読みなので確実にオフライン固定
if [ -f "$MODEL_PATH/model.safetensors.index.json" ]; then
  export HF_HUB_OFFLINE=1
fi

################### GRPO Trainer（remote vLLM 利用） ###################
# vLLM は別ノード常駐なので、学習は 2 ノード（TRAIN_NODE_A/B）のみで実行
TRAIN_NODES_LIST="${TRAIN_NODE_A},${TRAIN_NODE_B}"

TPN=1
srun -N 2 --ntasks-per-node="$TPN" --gpus-per-task=4 \
     --export=ALL,TMPDIR="$JOB_TMPDIR" \
     --kill-on-bad-exit=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --nodelist="$TRAIN_NODES_LIST" \
     --hint=nomultithread \
     --distribution=block:block \
     --cpu-bind=cores \
     --gpu-bind=closest \
     --mem-bind=local \
     --exclusive --chdir="$REPO_DIR" \
     bash -lc "
       set -euo pipefail
       source /home/Competition2025/P02/P02U017/openr1/bin/activate
       if [[ "\$SLURM_PROCID" != "0" ]]; then export WANDB_DISABLED=true; fi
       HUB=\"${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}\"; HUB=\"${HUB:-$HOME/.cache/huggingface/hub}\"
       if ls -d \"$HUB/models--${MODEL_ID//\//--}/snapshots\"/* >/dev/null 2>&1; then
         export HF_HUB_OFFLINE=1
         echo \"[HF] offline mode ON (prefetched detected)\"
       fi

       # compile系 OFF（子プロセス保険）
       export TORCH_COMPILE_DISABLE=1
       export TORCHDYNAMO_DISABLE=1
       export TORCHINDUCTOR_DISABLE=1
       export PYTORCH_JIT=0
       export NVFUSER_DISABLE=1

       echo \"[GRPO-Remote] on \$HOSTNAME  (node rank \$SLURM_NODEID, proc \$SLURM_PROCID)\"
       echo \"[train] MODEL_PATH=\$MODEL_PATH\"
       echo \"[train] $HOSTNAME node_id=\$SLURM_NODEID local_id=\$SLURM_LOCALID cpus_per_task=\$SLURM_CPUS_PER_TASK\"

       export RANK=\$SLURM_PROCID
       export WORLD_SIZE=\$SLURM_NTASKS
       export LOCAL_RANK=\$SLURM_LOCALID
       export MASTER_ADDR=${MAIN_IP}
       export MASTER_PORT=${MASTER_PORT}

       python /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_235b.yaml \
         --model_name_or_path \"$MODEL_PATH\" \
         --use_vllm true \
         --vllm_mode server \
         --vllm_server_base_url \"http://$VLLM_NODE:8000\" \
         --ds3_gather_for_generation false \
         --vllm_tensor_parallel_size ${VLLM_TP:-8} \
         ${GRPO_DEBUG_FLAGS} \
         --deepspeed \"$DS_CFG\" \
         --report_to wandb
     " &

TRAIN_SRUN_PID=$!

# 学習中スナップショット
if (( DEBUG_NUMA )); then
  sleep "$NUMA_SNAPSHOT_DELAY"
  for _i in $(seq 1 "$NUMA_SNAPSHOT_REPEAT"); do
    numa_snapshot "run-$_i"
    sleep "$NUMA_SNAPSHOT_INTERVAL"
  done
fi

# 置換後：1秒ポーリングで即終了、ログは300秒ごとに出す
(
  i=0
  while kill -0 "$VLLM_SRUN_PID" 2>/dev/null && kill -0 "$TRAIN_SRUN_PID" 2>/dev/null; do
    if (( i % 300 == 0 )); then
      ts=$(date +%H:%M:%S)
      curl -fsS "http://$VLLM_NODE:8000/health" >/dev/null && s=ok || s=ng
      echo "[HB $ts] vLLM health=$s"
    fi
    sleep 1
    i=$((i+1))
  done
) &


# 終了待ち
wait "$TRAIN_SRUN_PID"; rc=$?; if [ $rc -ne 0 ]; then echo "[ERR] training srun failed with exit code $rc" >&2; exit $rc; fi

# vLLM サーバ停止（念のため）
echo "[vLLM] stopping server..."
kill "$VLLM_SRUN_PID" 2>/dev/null || true
wait "$VLLM_SRUN_PID" 2>/dev/null || true

# 終了後サマリ
if (( DEBUG_NUMA )); then
  numa_snapshot "post"
fi

# メインノードだけで wandb 同期（任意）
if [[ "${NODELIST[0]}" == "$(hostname)" ]]; then
  echo "[W&B] syncing offline runs..."
  wandb sync --sync-all "${WANDB_DIR}" || true
fi

cleanup() {
  echo "Job finished. Cleaning up job-specific TMPDIR..."
  # 期待するジョブ固有パスかを厳格に確認してから削除
  if [[ -n "$JOB_TMPDIR" && "$JOB_TMPDIR" == "$NVME_BASE/tmp/$USER/job-$SLURM_JOB_ID" ]]; then
    srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 bash -lc 'rm -rf -- '"$JOB_TMPDIR"
    echo "Cleanup complete: $JOB_TMPDIR"
  else
    echo "[WARN] Skip cleanup: JOB_TMPDIR not job-specific -> '$JOB_TMPDIR'"
  fi
}
trap cleanup EXIT

wait
echo '[Job] all processes finished.'
