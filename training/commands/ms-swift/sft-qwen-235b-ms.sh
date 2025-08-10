#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=3              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[54,56,91] # 利用するノードのリスト
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

nodes=2
nproc_per_node=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=$nnodes \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
MASTER_PORT=29500 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen3-235B-A22B \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#500' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 32 / $nproc_per_node / $nnodes) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
    
# 実行コマンド
# sbatch ./llm2025compet/training/commands/ms-swift/sft-qwen-30b-ms-node1.sh