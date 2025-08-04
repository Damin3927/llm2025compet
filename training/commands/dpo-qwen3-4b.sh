#!/bin/bash
#SBATCH --partition P02        # 利用するパーティション（キュー）
#SBATCH --ntasks-per-node=1    # 1ノードあたりのタスク数
#SBATCH --nodes=3              # 利用するノード数
#SBATCH --gpus-per-node=8      # 1ノードあたりのGPU数
#SBATCH --nodelist osk-gpu[54,56,91] # 利用するノードのリスト
#SBATCH --job-name sft-32b     # ジョブの名前
#SBATCH --time 0:30:00         # ジョブの最大実行時間
#SBATCH --output sft-32b.out   # 標準出力ファイル
#SBATCH --error sft-32b.err    # 標準エラーファイル
#SBATCH --mem=512G             # メモリの割り当て  --mem=0 # 無制限にする場合は0を指定


export WANDB_DISABLED="true"   # WANDBを一旦無効化

# --- 環境設定 ---
module load cuda/12.8           # nvccを使うためにCUDAをロード

source openr1/bin/activate      # venvを有効化

cd llm2025compet/training/open-r1/src || exit 1


accelerate launch \
    --config_file ../recipes/accelerate_configs/zero3.yaml \
    --num_machines 3 \
    --num_processes 24 \
    open_r1/dpo.py \
    --config ../../configs/Qwen3-4B/DPO/config_dpo.yaml

# 複数GPUならzero3.yamlを使う
# GPUが1つならzero2.yamlを使う
# srun --partition P02 --nodes=3 --nodelist osk-gpu[54,56,91] --gpus-per-node=8 --time 1:00:00 --pty bash -i