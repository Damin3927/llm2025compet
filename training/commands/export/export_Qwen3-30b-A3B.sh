#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=1              # 利用するノード数
#SBATCH --gpus-per-node=1      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[91] # 利用するノードのリスト
#SBATCH --job-name ms-export     # ジョブの名前
#SBATCH --time 3:00:00         # ジョブの最大実行時間
#SBATCH --output ms-export.out   # 標準出力ファイル
#SBATCH --error ms-export.err    # 標準エラーファイル
#SBATCH --mem=0            # 各ノードのメモリサイズ
#SBATCH --cpus-per-task=160         # number of cores per tasks

module load cuda/12.8           # nvccを使うためにCUDAをロード

source ms-swift/bin/activate      # venvを有効化

ulimit -v unlimited
ulimit -m unlimited

cd llm2025compet/training/ms-swift || exit 1

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3-30B-A3B-Base \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir ~/.cache/ms-swift/Qwen3-30B-A3B-Base \