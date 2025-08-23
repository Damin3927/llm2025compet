#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --hint=nomultithread
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen8b-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen8b_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen8b_colo.err

set -euo pipefail
set -x

################### 早期・環境サニタイズ ###################
export PYTHONNOUSERSITE=1
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TORCHINDUCTOR_MAX_WORKERS=1
export PYTORCH_JIT=0
export NVFUSER_DISABLE=1

# Triton を強制無効化
export DEEPSPEED_DISABLE_TRITON=1
export DS_DISABLE_TRITON=1

################### 環境 ###################
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

# CUDA toolchain
export CUDA_HOME=/home/appli/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/targets/x86_64-linux/lib:/home/appli/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

source ~/openr1/bin/activate

# 一時ディレクトリ
export TMPDIR="/nvme12/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# HF キャッシュ
export HF_HOME=/home/Competition2025/P02/P02U017/.cache/huggingface_mydir
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="/home/Competition2025/P02/P02U017/hf-datasets-cache"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
# export NCCL_ASYNC_ERROR_HANDLING=1   # ← accelerate 側で扱うのでここは未設定でOK

################### ランタイム系 ###################
export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_HCA="mlx5_0:1,mlx5_1:1"
# export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=7200

export GLOO_LOG_LEVEL=TRACE


# 推定メモリ計算用（任意）：現在 8B, vLLM-TP=24 を想定
export MODEL_PARAMS_B=${MODEL_PARAMS_B:-8}
export VLLM_TP=${VLLM_TP:-24}

ulimit -n 65536
ulimit -v unlimited
ulimit -m unlimited

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノード情報 ###################
MAIN_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$MAIN_NODE"
export MAIN_IP=$(getent ahostsv4 "$MAIN_NODE" | awk 'NR==1{print $1}')
export MASTER_ADDR="${MAIN_IP}"
export MASTER_PORT=${MASTER_PORT:-29501}
#export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO

################### DeepSpeed NVMe ###################
DEEPSPEED_NVME_PATH="/nvme12/deepspeed_offload/${SLURM_JOB_ID}"
echo "[INFO] Creating DeepSpeed NVMe offload directory: $DEEPSPEED_NVME_PATH"
srun --nodes=3 --ntasks-per-node=1 --nodelist="$SLURM_JOB_NODELIST" --export=ALL \
  bash -lc "mkdir -p '$DEEPSPEED_NVME_PATH'"

export DS_CONF_TMP_SHARED="$REPO_DIR/../recipes/accelerate_configs/zero3.$SLURM_JOB_ID.materialized.yaml"
export SLURM_JOB_ID  # envsubst 用
envsubst < ../recipes/accelerate_configs/zero3.yaml > "$DS_CONF_TMP_SHARED"
ls -l "$DS_CONF_TMP_SHARED"

################### （任意）メモリ予算チェッカー ###################
# vLLM の重みシャードサイズや GPU 総メモリ, NVMe 空き容量を簡易表示
srun -N3 -n3 --nodelist="$SLURM_JOB_NODELIST" bash -lc 'python - <<PY
import os, torch, shutil
params_b = float(os.environ.get("MODEL_PARAMS_B", "8"))
tp = int(os.environ.get("VLLM_TP","24"))
dtype_bytes = 2  # bf16
vllm_weight_shard = params_b*1e9*dtype_bytes/tp
print(f"[MemEst] vLLM weight per-rank ≈ {vllm_weight_shard/1024**3:.2f} GiB (params={params_b}B, tp={tp}, bf16)")
tot = torch.cuda.get_device_properties(0).total_memory/1024**3
print(f"[MemEst] GPU total mem ≈ {tot:.1f} GiB")
nvme = os.environ.get("DEEPSPEED_NVME_PATH")
if nvme:
    usage = shutil.disk_usage(nvme)
    print(f"[MemEst] NVMe at {nvme}: free ≈ {usage.free/1024**3:.0f} GiB / total ≈ {usage.total/1024**3:.0f} GiB")
PY'

################### vLLM（コロケートモード）環境検査 ###################
srun -N3 -n3 --nodelist="${SLURM_JOB_NODELIST}" bash -lc 'python - << "PY"
import os, socket
print("[EnvCheck]", socket.gethostname(),
      "RANK", os.environ.get("RANK"),
      "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
      "WORLD_SIZE", os.environ.get("WORLD_SIZE"),
      "MASTER_ADDR", os.environ.get("MASTER_ADDR"),
      "MASTER_PORT", os.environ.get("MASTER_PORT"))
PY'

################### GRPO Trainer（コロケート） ###################
srun --nodes=3 --ntasks-per-node=1 --nodelist="$SLURM_JOB_NODELIST" \
     --kill-on-bad-exit=1 \
     --hint=nomultithread --mem-bind=local --gpu-bind=closest \
     --cpus-per-task=60 --cpu-bind=cores --distribution=block:block \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" --export=ALL,DS_CONF_TMP_SHARED="$DS_CONF_TMP_SHARED" \
     bash -c "
       source ~/openr1/bin/activate
       echo \"[GRPO-Colo] host=\$(hostname -s) nodeid=\$SLURM_NODEID procid=\$SLURM_PROCID\"
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       unset NCCL_ASYNC_ERROR_HANDLING
       export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

       # ★ 正しい書式
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

       export TORCH_COMPILE_DISABLE=1
       export TORCHDYNAMO_DISABLE=1
       export TORCHINDUCTOR_DISABLE=1
       export PYTORCH_JIT=0
       export NVFUSER_DISABLE=1

       unset WORLD_SIZE RANK LOCAL_RANK

       echo \"[DEBUG] MAIN_IP=\${MAIN_IP}  DS_CONF_TMP_SHARED=$DS_CONF_TMP_SHARED\"
       echo \"[DEBUG] SLURM_NODEID=\$SLURM_NODEID  SLURM_PROCID=\$SLURM_PROCID\"
       echo \"[DEBUG] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
       echo \"[DEBUG] deepspeed config (materialized):\"
       cat \"$DS_CONF_TMP_SHARED\" || true

       # NVMe パスを DS_CONF から読み出して存在確認（os.environ 経由）
       if grep -q nvme_path \"$DS_CONF_TMP_SHARED\"; then
         NVME_DIR=\$(python - <<'PY'
import os, sys, yaml
p = os.environ.get('DS_CONF_TMP_SHARED','')
if not p: sys.exit(0)
with open(p,'r') as f:
    y = yaml.safe_load(f)
try:
    print(y['deepspeed_config']['zero_optimization']['offload_param']['nvme_path'])
except Exception:
    pass
PY
)
         if [ -n \"\$NVME_DIR\" ]; then
           echo \"[DEBUG] nvme_path=\$NVME_DIR exists?\"; ls -ld \"\$NVME_DIR\" || true
         fi
       fi

       echo \"[DEBUG] torch/deepspeed import sanity:\"
       python - << 'PY'
import torch
print('[pycheck] torch.cuda.is_available =', torch.cuda.is_available())
try:
    import deepspeed
    print('[pycheck] deepspeed import OK')
except Exception as e:
    print('[pycheck] deepspeed import FAILED:', e)
PY

       # 3ノード・24GPUでコロケートモードによるGRPOトレーニング
       accelerate launch \\
         --config_file \"$DS_CONF_TMP_SHARED\" \\
         --num_machines 3 \\
         --main_process_ip ${MAIN_IP} \\
         --main_process_port 29500 \\
         --rdzv_backend c10d \\
         --machine_rank \$SLURM_PROCID \\
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo1.5b.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml \\
         --use_vllm true \\
         --vllm_mode colocate \\
         --vllm_tensor_parallel_size 1 \\
         --vllm_gpu_memory_utilization 0.30 \\
         --bf16 true \\
         --report_to none
     "

trap 'echo "[CLEANUP] removing $TMPDIR $DEEPSPEED_NVME_PATH"; \
      srun -N3 -n3 --nodelist="${SLURM_JOB_NODELIST}" bash -lc "rm -rf $DEEPSPEED_NVME_PATH"; \
      rm -rf "$TMPDIR"' EXIT

wait
echo '[Job] all processes finished.'
