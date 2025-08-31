#!/bin/bash
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1  # GPUを1つだけ要求
#SBATCH --job-name=unsloth-warmup
#SBATCH --time=0:05:00     # 5分あれば十分です
#SBATCH --output=warmup_unsloth.out
#SBATCH --error=warmup_unsloth.err

# --- 環境設定 ---
module load cuda/12.8
export CUDA_HOME=/home/appli/cuda/12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

source ~/openr1/bin/activate
export UNSLOTH_CACHE=~/unsloth_cache

# --- キャッシュ生成スクリプトの実行 ---
echo "GPUノードでUnslothのキャッシュを生成します..."
python /home/Competition2025/P02/P02U025/llm2025compet/training/open-r1/src/open_r1/warmup_unsloth.py
echo "キャッシュの生成が完了しました。"