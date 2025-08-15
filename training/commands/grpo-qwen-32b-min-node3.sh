#!/bin/bash
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --distribution=block:block
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen32b
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen32b_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen32b_colo.err
set -euo pipefail

export PYTHONNOUSERSITE=1
source /home/Competition2025/P02/P02U017/openr1/bin/activate


module unload cuda || true
module unload nccl || true
module purge
module load cuda/12.6
module load nccl/2.24.3

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

NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
srun -N2 -n2 -w ${NODELIST[0]},${NODELIST[1]} bash -lc 'source ~/openr1/bin/activate; python -c "import sys; print(sys.prefix)"'

TRAIN_NODES="${NODELIST[0]},${NODELIST[1]}"  # 学習2ノード使うなら :0:2
VLLM_NODE="${NODELIST[2]}"        # vLLM ノード

MASTER_HOST="${NODELIST[0]}"
MASTER_ADDR=$(getent ahostsv4 "$MASTER_HOST" | awk 'NR==1{print $1}')
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT

# (任意) IPv4 の NIC を PyTorch/NCCL に合わせると安定
IFACE=$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
[ -n "$IFACE" ] && export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"


srun --nodes=1 --ntasks=1 --nodelist="${VLLM_NODE}" \
  bash -lc '
    source /home/Competition2025/P02/P02U017/openr1/bin/activate
    ulimit -n 65536; ulimit -v unlimited; ulimit -m unlimited
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-32B \
      --served-model-name qwen32b-base \
      --tensor-parallel-size 8 \
      --dtype bfloat16 \
      --host 0.0.0.0 --port 8000 \
      --disable-uvicorn-access-log
  ' &
VLLM_SRUN_PID=$!

# --- 2) ヘルスゲート（落ちたら即中断） ---
echo "[vLLM] waiting for http://${VLLM_NODE}:8000/health ..."
for i in $(seq 1 600); do
  if ! kill -0 "$VLLM_SRUN_PID" 2>/dev/null; then
    echo "[ABORT] vLLM died before healthy"; exit 50
  fi
  if curl -fsS "http://${VLLM_NODE}:8000/health" >/dev/null 2>&1; then
    echo "[vLLM] healthy"; break
  fi
  sleep 1
  [[ $i -eq 600 ]] && { echo "[ABORT] vLLM health timeout"; exit 51; }
done

# === DEBUG WRAPPER (drop-in) ===
export DEBUG_DIR="/home/Competition2025/P02/P02U017/llm2025compet/training/logs/debug_${SLURM_JOB_ID}"
mkdir -p "$DEBUG_DIR"

cat > "$DEBUG_DIR/train_wrapper.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
# 1) ログを逐次出す
exec > >(stdbuf -oL -eL awk '{print strftime("[%-m/%-d %H:%M:%S]"), $0; fflush()}') 2>&1
set -x

# 2) 典型的な“黙って落ちる”系に効く環境
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO             # うるさければ WARN に戻す
export RDMAV_FORK_SAFE=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 3) ソフト＆ハードの見える化
echo "[ENV] HOSTNAME=$HOSTNAME SLURM_NODEID=${SLURM_NODEID:-?} RANK=${RANK:-?}"
echo "[PATH] python=$(command -v python)  pip=$(command -v pip)"
ulimit -a
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
free -h || true
df -h / /home /tmp || true

# 4) venv を必ず合わせる
source /home/Competition2025/P02/P02U017/openr1/bin/activate
python - <<'PY'
import sys, pkgutil
print("[PY] prefix:", sys.prefix)
for m in ("peft","datasets","transformers","torch","numpy","pandas"):
    print(f"[import] {m}:", "OK" if pkgutil.find_loader(m) else "MISSING")
PY

# 5) torchrun を実行（引数はこのスクリプトの引数をそのまま）
echo "[RUN] torchrun $*"
torchrun "$@"
BASH
chmod +x "$DEBUG_DIR/train_wrapper.sh"

# --- 3) 学習起動（torchrun/ZeRO-3） ---
srun --nodes=2 --ntasks=2 --nodelist="${NODELIST[0]},${NODELIST[1]}" --export=ALL \
     --hint=nomultithread --distribution=block:block \
     --cpu-bind=cores --gpu-bind=closest --mem-bind=local \
  bash -lc '
  source /home/Competition2025/P02/P02U017/openr1/bin/activate
  # IFACE を各ノードで推定 → NCCL/GLOO 揃える（既に入れてたら重複OK）
  IFACE=$(ip -4 route get 1.1.1.1 | awk "{for(i=1;i<=NF;i++) if(\$i==\"dev\"){print \$(i+1); exit}}")
  [ -n "$IFACE" ] && export NCCL_SOCKET_IFNAME="$IFACE" GLOO_SOCKET_IFNAME="$IFACE"
  echo "[IFACE] $HOSTNAME -> $IFACE"

  # MASTER_ADDR は IP を使う（上で export 済み想定）
  "$DEBUG_DIR"/train_wrapper.sh \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=${SLURM_NODEID} \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \
      --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_32b_mini.yaml \
      --deepspeed /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/recipes/deepspeed/ds_zero3.json \
      --use_vllm true \
      --vllm_server_base_url "http://'"$VLLM_NODE"':8000" \
      --vllm_tensor_parallel_size 8
'

trap '
  rc=$?
  echo "[EXIT] rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "=== vLLM last 200 lines ==="
    journalctl -t uvicorn -n 200 2>/dev/null || true
    # もしくは自前で vLLM の出力を tee していたログを tail
    for f in /home/.../logs/train_*.out /home/.../logs/train_*.err; do
      [ -f "$f" ] && { echo "=== tail $f ==="; tail -n 200 "$f"; }
    done
  fi
' EXIT

