#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --nodes=1              # 利用するノード数
#SBATCH --gpus-per-node=1      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[56] # 利用するノードのリスト
#SBATCH --job-name sft-0.5b    # ジョブの名前
#SBATCH --time 1:00:00         # ジョブの最大実行時間
#SBATCH --output sft-0.5b.out  # 標準出力ファイル
#SBATCH --error sft-0.5b.err   # 標準エラーファイル

export WANDB_DISABLED="true"   # WANDBを一旦無効化

# --- 環境設定 ---
module load cuda/12.8           # nvccを使うためにCUDAをロード

source openr1/bin/activate      # venvを有効化

cd llm2025compet/training/open-r1/src || exit 1

accelerate launch \
    --config_file ../recipes/accelerate_configs/zero2.yaml \
    --num_machines 1 \
    --num_processes 1 \
    open_r1/sft.py \
    --config ../../configs/Qwen2.5-Distill-0.5b-test/sft/config_distill.yaml

# 実行方法
# HOMEで以下を実行する。自動でopen-r1のソースコードディレクトリに移動することに注意

# 以下のコマンドでダミージョブをキャンセルする必要がある。
# /home/Competition2025/P02/shareP02/scripts/scancel.sh 287614

# 実行コマンド
# sbatch ./llm2025compet/training/commands/sft-qwen-0.5b.sh

# bashでの実行コマンド
# srun --partition P02 --nodes=1 --nodelist osk-gpu[56] --gpus-per-node=1 --time 1:00:00 --pty bash -i
# . ./llm2025compet/training/commands/sft-qwen-0.5b.sh