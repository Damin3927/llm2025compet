#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=1              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[91] # 利用するノードのリスト
#SBATCH --job-name dl-235b     # ジョブの名前
#SBATCH --time 2:00:00         # ジョブの最大実行時間
#SBATCH --output dl-235b.out # 標準出力ファイル
#SBATCH --error dl-235b.err  # 標準エラーファイル
#SBATCH --mem=0                # 各ノードのメモリサイズ 0は無制限
#SBATCH --cpus-per-task=160     # number of cores per tasks

# export WANDB_DISABLED="true"   # WANDBを一旦無効化

# Slurmで確保したノードリストの先頭をマスターノードのアドレスとして設定
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

# 使用されていない適当なポート番号を設定 (例: 29500)
export MASTER_PORT=29500
echo "MASTER_PORT: $MASTER_PORT"

export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_GID_INDEX=3
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NVME_BASE=/nvme78
export HF_HOME="$NVME_BASE/P02U024/hf-cache"
export HF_HUB_CACHE="$NVME_BASE/P02U024/hub"
export SENTENCE_TRANSFORMERS_HOME="$NVME_BASE/P02U024/sentence/"
mkdir -p ${HF_HOME}

module load cuda/12.8           # nvccを使うためにCUDAをロード

source openr1/bin/activate      # venvを有効化

ulimit -v unlimited
ulimit -m unlimited

cd llm2025compet/training/open-r1/src || exit 1

if [ "$SLURM_PROCID" == "0" ]; then
    echo "Downloading model to NVMe on the first node..."
    python3 open_r1/download_model.py --name Qwen/Qwen3-235B-A22B
fi