#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=1              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[91] # 利用するノードのリスト
#SBATCH --job-name ms-235b     # ジョブの名前
#SBATCH --time 3:00:00         # ジョブの最大実行時間
#SBATCH --output ms-235b.out   # 標準出力ファイル
#SBATCH --error ms-235b.err    # 標準エラーファイル
#SBATCH --mem=0            # 各ノードのメモリサイズ
#SBATCH --cpus-per-task=160         # number of cores per tasks

# export WANDB_DISABLED="true"   # WANDBを一旦無効化

# Slurmで確保したノードリストの先頭をマスターノードのアドレスとして設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

# 使用されていない適当なポート番号を設定 (例: 29500)
export MASTER_PORT=29500
echo "MASTER_PORT: $MASTER_PORT"

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=NVL
# export NCCL_IB_GID_INDEX=3
# export DEEPSPEED_TIMEOUT=7200
# export GLOO_SOCKET_IFNAME=enp25s0np0
# export NCCL_SOCKET_IFNAME=enp25s0np0
# export NCCL_TIMEOUT=5400
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=^lo,docker,virbr
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

# https://github.com/QwenLM/Qwen3/issues/1278

module load cuda/12.8           # nvccを使うためにCUDAをロード

source ms-swift/bin/activate      # venvを有効化

ulimit -v unlimited
ulimit -m unlimited

cd llm2025compet/training/ms-swift || exit 1

export SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && echo $SITE_PACKAGES && \
export CUDNN_PATH=$SITE_PACKAGES/nvidia/cudnn CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include && \
export LD_LIBRARY_PATH=/home/appli/cuda/12.8/lib64 && \
export CUDA_HOME=/home/appli/cuda/12.8 && \
export CUDNN_LIBRARY=/home/appli/cuda/12.8/lib64 && \
export CPLUS_INCLUDE_PATH=$SITE_PACKAGES/nvidia/cudnn/include && \
export TORCH_CUDA_ARCH_LIST="9.0"

# nvidia-smi --query-gpu=compute_cap --format=csv,noheader # GPUのCompute Capabilityを調べる

if [ ":$PATH:" != *":/home/appli/cuda/12.8/bin:"* ]; then
    export PATH="/home/appli/cuda/12.8/bin:$PATH"
fi
echo "PATH ... $PATH"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --load Qwen/Qwen3-30B-A3B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 300 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 30 \
    --min_lr 1e-6 \
    --save ~/llm2025compet/training/ms-swift/data \
    --save_interval 100 \
    --max_length 32768 \
    --num_workers 1 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 64 \
    --packing true \
    --streaming false \
    --dataset_shuffle true \
    --use_flash_attn true \
    --wandb_project qwen3 \
    --wandb_exp_name 30b-a3b-sft \
    --log_interval 1 \
    --use_chat_template true
    
# 実行コマンド
# sbatch ./llm2025compet/training/commands/ms-swift/sft-qwen-30b-ms-node1.sh