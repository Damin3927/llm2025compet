#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=3              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[54,56,91] # 利用するノードのリスト
#SBATCH --job-name sft-235b     # ジョブの名前
#SBATCH --time 1:00:00         # ジョブの最大実行時間
#SBATCH --output sft-235b.out   # 標準出力ファイル
#SBATCH --error sft-235b.err    # 標準エラーファイル
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

export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=^lo,docker,virbr
#export TORCH_DISTRIBUTED_DEBUG=DETAIL

module load cuda/12.8           # nvccを使うためにCUDAをロード

source openr1/bin/activate      # venvを有効化

ulimit -v unlimited
ulimit -m unlimited

cd llm2025compet/training/open-r1/src || exit 1

srun --jobid $SLURM_JOB_ID --mem=0 bash -c \
    "accelerate launch \
        --config_file ../recipes/accelerate_configs/zero3.yaml \
        --num_machines 3 \
        --num_processes 24 \
        --main_process_ip \"$MASTER_ADDR\" \
        --main_process_port \"$MASTER_PORT\" \
        --rdzv_backend c10d \
        open_r1/sft.py \
        --config ../../configs/Qwen3-235b/sft/config_distill.yaml"

# 実行方法
# HOMEで以下を実行する。自動でopen-r1のソースコードディレクトリに移動することに注意

# 以下のコマンドでダミージョブをキャンセルする必要がある。
# /home/Competition2025/P02/shareP02/scripts/scancel.sh 287614

# 実行コマンド
# sbatch ./llm2025compet/training/commands/sft-qwen-32b-node3.sh