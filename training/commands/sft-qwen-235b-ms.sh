#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=3              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[54,56,91] # 利用するノードのリスト
#SBATCH --job-name sft-235b     # ジョブの名前
#SBATCH --time 3:00:00         # ジョブの最大実行時間
#SBATCH --output sft-235b16.out   # 標準出力ファイル
#SBATCH --error sft-235b16.err    # 標準エラーファイル
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

echo ${WORLD_SIZE}
echo ${RANK}

# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# Please ensure that the weight saving paths are the same for both nodes.

# NNODES=$WORLD_SIZE \
# NODE_RANK=$RANK \
# megatron sft \
#     --load Qwen3-30B-A3B-Base-mcore \
#     --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
#     --tensor_model_parallel_size 3 \
#     --expert_model_parallel_size 8 \
#     --moe_grouped_gemm true \
#     --moe_shared_expert_overlap true \
#     --moe_aux_loss_coeff 0.01 \
#     --micro_batch_size 1 \
#     --global_batch_size 24 \
#     --packing true \
#     --recompute_granularity full \
#     --recompute_method uniform \
#     --recompute_num_layers 1 \
#     --train_iters 2000 \
#     --eval_iters 50 \
#     --finetune true \
#     --cross_entropy_loss_fusion true \
#     --lr 1e-5 \
#     --lr_warmup_iters 100 \
#     --min_lr 1e-6 \
#     --save megatron_output/Qwen3-30B-A3B-Base \
#     --eval_interval 200 \
#     --save_interval 200 \
#     --max_length 8192 \
#     --num_workers 8 \
#     --dataset_num_proc 8 \
#     --no_save_optim true \
#     --no_save_rng true \
#     --sequence_parallel true \
#     --use_flash_attn true

# 実行コマンド
# sbatch ./llm2025compet/training/commands/sft-qwen-235b-ms.sh

srun --jobid $SLURM_JOB_ID --mem=0 bash -c \
    "
    NNODES=$WORLD_SIZE \
    NODE_RANK=$RANK \
    megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --tensor_model_parallel_size 3 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 24 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 100 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true"