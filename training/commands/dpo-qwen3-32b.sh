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
    --config_file ../recipes/accelerate_configs/zero3.yaml \
    --num_machines 1 \
    --num_processes 8 \
    open_r1/dpo.py \
    --config ../../configs/Qwen3-32b/DPO/config_dpo.yaml

# 複数GPUならzero3.yamlを使う
# GPUが1つならzero2.yamlを使う