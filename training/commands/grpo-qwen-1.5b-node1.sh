#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition P02               # 利用するパーティション
#SBATCH --nodes=1                     # ノード数
#SBATCH --gpus-per-node=1             # 1ノードあたりGPU数
#SBATCH --nodelist osk-gpu[91]          # 利用ノードを固定
#SBATCH --job-name=grpo-qwen1.5b      # ジョブ名
#SBATCH --time=1:00:00                # 最大実行時間
#SBATCH --output=grpo-qwen1.5b.%j.out # 標準出力
#SBATCH --error=grpo-qwen1.5b.%j.err  # 標準エラー
#SBATCH --mem=0                       # 必要なら固定値に変更

################### 環境変数 / モジュール ###################
export WANDB_DISABLED="true"          # 使うなら false に
module load cuda/12.8                 # CUDA 環境
source openr1/bin/activate            # venv を有効化

################### 実行ディレクトリ ###################
# open-r1 のソースフォルダへ移動
cd /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src || exit 1

################### GRPO 学習 ###################
accelerate launch \
    --config_file ../recipes/accelerate_configs/zero2.yaml \
    --num_machines 1 \
    --num_processes 1 \
    open_r1/grpo.py \
    --config ../recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_math_gated.yaml

################### 参考 – 実行方法 ###################
# --- ジョブ投入例 ---
# sbatch llm2025compet/training/commands/grpo-qwen-1.5b.sh
#
# --- 対話実行例 (デバッグ時) ---
# srun --partition P02 --nodes=1 --nodelist=osk-gpu91 \
#      --gpus-per-node=1 --time=4:00:00 --pty bash -i
# . llm2025compet/training/commands/grpo-qwen-1.5b.sh
