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

# 8 * 80GiB, 3.2s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --load Qwen3-235B-A22B-Instruct-2507-mcore \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-235B-A22B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
    
# 実行コマンド
# sbatch ./llm2025compet/training/commands/sft-qwen-235b-ms-node1.sh