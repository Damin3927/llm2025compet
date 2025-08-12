#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=8
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
TPN="${SLURM_NTASKS_PER_NODE:-8}"

################### 環境 ###################
# >> TMPDIR moved before prefetch (auto)

export NVME_BASE=/nvme12

# まず各ノードで TMPDIR の親を作成（このステップは TMPDIR=/tmp で実行して警告回避）

srun -N "$NODES" -n "$NODES" --gpus=0 --export=ALL,TMPDIR=/tmp -l bash -lc 'mkdir -p "/nvme12/tmp/$USER" && echo "[precreate] $HOSTNAME prepared /nvme12/tmp/$USER"'

# 以降の処理では NVMe の TMPDIR を使う

export TMPDIR="$NVME_BASE/tmp/$USER"; mkdir -p "$TMPDIR"

export TMP="$TMPDIR"; export TEMP="$TMPDIR"
export REQUIRE_FULL_PREFETCH=1

module unload cuda || true
module unload nccl || true
module purge
module load cuda/12.6
module load nccl/2.24.3

# venvを有効化（prefetch前に）
source ~/openr1/bin/activate
python - "$DS_CFG" <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec("hf_transfer")
print("hf_transfer:", "present" if spec else "absent", "@", sys.prefix)
PY

# ここが “ダウンロード元”
export PREFETCH_DIR="$HOME/.cache/huggingface_mydir/models/Qwen3-235B-A22B"
export HF_HUB_ENABLE_HF_TRANSFER=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

python - "$DS_CFG" <<'PY'
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
export NVME_BASE=/nvme12
export TMPDIR="$NVME_BASE/tmp/$USER"; mkdir -p "$TMPDIR"
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
srun -N "$NODES" -n "$NODES" --gpus=0 -l bash -lc "
  mkdir -p \"$TMPDIR\" \"$HF_HUB_CACHE\" && \
  echo \"[$HOSTNAME] TMPDIR=$TMPDIR  HF_HOME=$HF_HOME\"
"

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
python - "$DS_CFG" <<'PY'
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
  "zero3_save_16bit_model": true,
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
export TRITON_CACHE_DIR="${NVME_BASE}/triton-cache-$USER"; mkdir -p "$TRITON_CACHE_DIR"

if (( DEBUG_RUN )); then
  # デバッグでは vLLM を無効化して軽量スモーク
  export GRPO_DEBUG_FLAGS="--max_steps 12 --logging_steps 1 --save_strategy no \
                           --evaluation_strategy no --num_generations 2 \
                           --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
                           --use_vllm false"
else
  export GRPO_DEBUG_FLAGS=""
fi

export MASTER_PORT=${MASTER_PORT:-$((12000 + SLURM_JOB_ID % 20000))}

################### GRPO Trainer（コロケート） ###################
srun -N "$NODES" --ntasks-per-node="$TPN" --gpus-per-task=1 \
     --kill-on-bad-exit=1 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     --nodelist="$SLURM_JOB_NODELIST" \
     --hint=nomultithread \
     --distribution=block:block \
     --cpu-bind=cores \
     --gpu-bind=closest \
     --mem-bind=local \
     --exclusive --chdir="$REPO_DIR" \
     bash -lc "
       source /home/Competition2025/P02/P02U017/openr1/bin/activate
       if [[ \"\$SLURM_NODEID\" != \"0\" ]]; then export WANDB_DISABLED=true; fi
       HUB=\"${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}\"; HUB=\"${HUB:-$HOME/.cache/huggingface/hub}\"
       if ls -d \"$HUB/models--${MODEL_ID//\//--}/snapshots\"/* >/dev/null 2>&1; then
         export HF_HUB_OFFLINE=1
         echo \"[HF] offline mode ON (prefetched detected)\"
       fi

       # 再度、compile系をOFF（子プロセス保険）
       export TORCH_COMPILE_DISABLE=1
       export TORCHDYNAMO_DISABLE=1
       export TORCHINDUCTOR_DISABLE=1
       export PYTORCH_JIT=0
       export NVFUSER_DISABLE=1

       echo \"[GRPO-Colo] on \$HOSTNAME  (node rank \$SLURM_NODEID, proc \$SLURM_PROCID)\"
       echo "[train] MODEL_PATH=\$MODEL_PATH"
       echo "[train] $HOSTNAME node_id=$SLURM_NODEID local_id=$SLURM_LOCALID cpus_per_task=$SLURM_CPUS_PER_TASK"
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4

       export RANK=\$SLURM_PROCID
       export WORLD_SIZE=\$SLURM_NTASKS
       export LOCAL_RANK=\$SLURM_LOCALID
       export MASTER_ADDR=${MAIN_IP}
       export MASTER_PORT=${MASTER_PORT}

       echo "[model-check] MODEL_PATH=$MODEL_PATH"
       shopt -s nullglob
       shards=("$MODEL_PATH"/model-*.safetensors)
       echo "[model-check] $HOSTNAME shards=${#shards[@]} index=$([[ -f "$MODEL_PATH/model.safetensors.index.json" ]] && echo yes || echo no)"

       echo "[train-check] host=$HOSTNAME node_rank=$SLURM_NODEID proc=$SLURM_PROCID local=$SLURM_LOCALID"
       taskset -pc $$ | awk -F': ' '{print "[train-check] taskset="$2}'
       numactl -s | egrep 'cpubind|membind'

       echo "[model-check] MODEL_PATH=$MODEL_PATH";
       shopt -s nullglob;
       shards=("$MODEL_PATH"/model-*.safetensors);
       echo "[model-check] $HOSTNAME shards=${#shards[@]} index=$([[ -f "$MODEL_PATH/model.safetensors.index.json" ]] && echo yes || echo no)"
       echo "[train] RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID NODE_RANK=$SLURM_NODEID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

       python /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_235b.yaml \
         --model_name_or_path \"$MODEL_PATH\" \
         --use_vllm true \
         --vllm_mode colocate \
         --ds3_gather_for_generation true \
         ${GRPO_DEBUG_FLAGS} \
         --deepspeed "$DS_CFG" \
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

# 終了待ち
wait "$TRAIN_SRUN_PID"; rc=$?; if [ $rc -ne 0 ]; then echo "[ERR] training srun failed with exit code $rc" >&2; exit $rc; fi

# 終了後サマリ
if (( DEBUG_NUMA )); then
  numa_snapshot "post"
fi

# 終了クリーンアップ
function cleanup() {
    echo "Job finished. Cleaning up NVMe directory..."
    srun -N "$NODES" -n "$NODES" --ntasks-per-node=1 bash -c "rm -rf ${NVME_BASE}/slurm-job-${SLURM_JOB_ID}"
    echo "Cleanup complete."
}
trap cleanup EXIT

wait
echo '[Job] all processes finished.'


#######################################################################
# How to use (デバッグ/本番ランの使い分け)
#
# ▼前提
# - モデルは事前に $PREFETCH_DIR に snapshot_download され、
#   各ノードの NVMe 永続キャッシュ $NVME_PERSIST_DIR に同期されます。
#   ".ready" があると再同期はスキップ。やり直す場合は各ノードで ".ready" を削除。
#
# ▼よく使う環境変数（sbatch の前に付けて渡せます）
#   DEBUG_RUN=1|0            # 1=デバッグ（既定）/ 0=本番
#   MIN_BAR1_FREE_MB=2048    # BAR1閾値。低めに流したい時は 1024 などに下げる
#   MASTER_PORT=xxxxx        # 通信ポート固定したい時
#   REQUIRED_FREE_GB=800     # NVMeの最低空き容量チェック
#   DEBUG_NUMA=0|1           # NUMAスナップショット有効/無効（既定 1）
#   NUMA_SNAPSHOT_DELAY=20   # 開始までの待ち秒
#   NUMA_SNAPSHOT_REPEAT=6   # 取得回数
#   NUMA_SNAPSHOT_INTERVAL=60# 取得間隔秒
#
# 投げるコマンド（お試し→本番）：
#
# 例: このファイルのパスを変数に入れてから投げる
# FILE=/home/Competition2025/P02/P02U017/llm2025compet/training/commands/grpo-qwen-235b-colocate-node3.sh
#
# スモーク（DEBUG）
# sbatch --export=ALL,DEBUG_RUN=1 \
#   -N 3 --gpus-per-node=8 --ntasks-per-node=8 \
#   --nodelist=osk-gpu[54,56,91] --time=0:30:00 \
#   --job-name=grpo-qwen235b-smoke \
#   "$FILE"
#
# 本番
# sbatch --export=ALL,DEBUG_RUN=0 \
#   -N 3 --gpus-per-node=8 --ntasks-per-node=8 \
#   --nodelist=osk-gpu[54,56,91] --time=4:00:00 \
#   --job-name=grpo-qwen235b-colo \
#   "$FILE"
# ▼ログ/監視
#   ファイル: 
#     stdout: /home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.out
#     stderr: /home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen235b_colo.err
#   コマンド:
#     tail -f <上記out/err>
#     squeue -j <JOBID>
#     sacct -j <JOBID> --format=JobID,State,Elapsed,NodeList
#     scancel <JOBID>
#
# ▼モデルキャッシュの状態確認（任意）
#   各ノード:
#     ls -1 $NVME_PERSIST_DIR/model-*.safetensors | wc -l   # → 118 を期待
#     test -f $NVME_PERSIST_DIR/model.safetensors.index.json && echo OK
#   不整合があるノードは ".ready" を削除すると次回再同期します:
#     rm -f $NVME_PERSIST_DIR/.ready
#
# ▼HFオフライン挙動
#   事前に $HF_HUB_CACHE にスナップショットがあれば自動で HF_HUB_OFFLINE=1 を有効化。
#   完全にローカル固定で動かしたい場合は、下の2行のコメントを外してもOK:
#     # export HF_HUB_OFFLINE=1
#     # export TRANSFORMERS_OFFLINE=1
#
# ▼NVMeの掃除
#   このスクリプトは終了時に /nvme12/slurm-job-<JOB_ID> をクリーンアップします。
#   永続キャッシュ /nvme12/persist-<USER>/... は残します。
#   溜まった一時ジョブディレクトリを手動で掃除する例（各ノードで実行）:
#     find /nvme12 -maxdepth 1 -type d -name "slurm-job-*" -mtime +1 -exec rm -rf {} +
#
# ▼トラブルシュート
#   - 「GRPO requires at least 2 generations per prompt…」:
#       num_generations が 2 以上かを確認（デバッグ既定は2。本番は設定YAML側）
#   - NUMAスナップショットに "no PIDs found yet":
#       学習が短すぎる可能性。NUMA_SNAPSHOT_DELAY を小さくするか、max_steps を増やす
#   - BAR1 の一時的 0MiB:
#       スクリプトで自動的に [SKIP] 扱い。持続して閾値未満なら MIN_BAR1_FREE_MB を調整
#
# ▼再現用の最小確認（モデルパス/NUMA/ランク）
#   ログに以下が出力されればOK:
#     [persist] ... ready (118/118) or reuse (118/118)
#     [model-check] ... shards=118 index=yes
#     [train] RANK=.. LOCAL_RANK=.. NODE_RANK=.. CUDA_VISIBLE_DEVICES=...
#######################################################################
