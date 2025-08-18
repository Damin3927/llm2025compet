#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=3                        # ★全3ノードをすべてトレーニングに使用
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --hint=nomultithread
#SBATCH --nodelist=osk-gpu[54,56,91]
#SBATCH --job-name=grpo-qwen1_5b-colo
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b_colo.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b_colo.err


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

# CUDA toolchain パス
export CUDA_HOME=/home/appli/cuda/12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$CUDA_HOME/targets/x86_64-linux/lib:/home/appli/nccl/2.24.3/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

source ~/openr1/bin/activate

# 一時ディレクトリを実行可能なNVMeに変更
export TMPDIR="/nvme12/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

# 既存の実態に合わせる
export HF_HOME=/home/Competition2025/P02/P02U017/.cache/huggingface_mydir
export HF_HUB_CACHE="$HF_HOME/hub"                    # モデル類のキャッシュ
export HF_DATASETS_CACHE="/home/Competition2025/P02/P02U017/hf-datasets-cache"  # データセット用

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export NCCL_ASYNC_ERROR_HANDLING=1

# ランタイム環境系
export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
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

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### ノードリスト取得 ###################
NODELIST=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MAIN_IP="${NODELIST[0]}"

DEEPSPEED_NVME_PATH="/nvme12/deepspeed_offload/${SLURM_JOB_ID}"
echo "[INFO] Creating DeepSpeed NVMe offload directory: $DEEPSPEED_NVME_PATH"
srun --nodes=3 --ntasks-per-node=1 --nodelist="$SLURM_JOB_NODELIST" --export=ALL \
  bash -lc "mkdir -p '$DEEPSPEED_NVME_PATH'"

DS_CONF_TMP_SHARED="$REPO_DIR/../recipes/accelerate_configs/zero3.$SLURM_JOB_ID.materialized.yaml"
export SLURM_JOB_ID  # envsubst のため
envsubst < ../recipes/accelerate_configs/zero3.yaml > "$DS_CONF_TMP_SHARED"
ls -l "$DS_CONF_TMP_SHARED"

################### vLLM（コロケートモード） ###################
# ※別プロセスでのvLLMサーバ起動は不要（各Trainer内でvLLMエンジンを起動）

################### GRPO Trainer（コロケートモードで実行） ###################
srun --nodes=3 --ntasks=3 --nodelist="${NODELIST[*]}" \
     --kill-on-bad-exit=1 \
     --hint=nomultithread --mem-bind=local --gpu-bind=closest \
     --cpus-per-task=60 --cpu-bind=cores --distribution=block:block \
     --gres=gpu:8 --exclusive --chdir="$REPO_DIR" --export=ALL \
     bash -c "
       source ~/openr1/bin/activate
       echo \"[GRPO-Colo] on \$HOSTNAME  (rank \$SLURM_PROCID)\"
       export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4
       unset NCCL_ASYNC_ERROR_HANDLING
       export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

       export TORCH_COMPILE_DISABLE=1
       export TORCHDYNAMO_DISABLE=1
       export TORCHINDUCTOR_DISABLE=1
       export PYTORCH_JIT=0
       export NVFUSER_DISABLE=1

       # 3ノード・24GPUでコロケートモードによるGRPOトレーニング
       accelerate launch \\
         --config_file "$DS_CONF_TMP_SHARED" \\
         --num_machines 3 \\
         --num_processes 24 \\
         --main_process_ip ${MAIN_IP} \\
         --main_process_port 29500 \\
         --rdzv_backend c10d \\
         --machine_rank \$SLURM_PROCID \\
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo1.5b.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml \\
         --use_vllm true \\
         --vllm_gpu_memory_utilization 0.7 \\
         --vllm_mode colocate \\
         --report_to none
     "

trap 'echo "[CLEANUP] removing $TMPDIR $DEEPSPEED_NVME_PATH"; \
      srun -N3 -n3 --nodelist="${SLURM_JOB_NODELIST}" bash -lc "rm -rf $DEEPSPEED_NVME_PATH"; \
      rm -rf "$TMPDIR"' EXIT

wait
echo '[Job] all processes finished.'
