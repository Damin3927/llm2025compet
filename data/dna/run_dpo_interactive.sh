#!/bin/bash

# さくらDCサーバ対話型実行用スクリプト
# 本田大明チーム（P02）向け

echo "=== DPO Data Generation - Interactive Mode ==="
echo "Team: 本田大明 (P02)"
echo "Date: $(date)"
echo ""

# 対話型セッション開始
echo "Starting interactive session on compute node..."
echo "This will allocate resources for DPO data generation."
echo ""

# srunで対話型セッションを開始
srun --partition=P02 \
     --nodes=1 \
     --nodelist=osk-gpu54 \
     --gres=gpu:8 \
     --time=24:00:00 \
     --pty bash -c "
echo '=== Interactive Session Started ==='
echo 'Node: \$(hostname)'
echo 'Date: \$(date)'
echo ''

# CUDA環境設定
module load cuda/12.4
echo 'CUDA module loaded'

# 環境変数設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME='/home/Competition2025/P02/shareP02/.cache/huggingface'
export TRANSFORMERS_CACHE='/home/Competition2025/P02/shareP02/.cache/transformers'

# GPU状態確認
echo '=== GPU Status ==='
nvidia-smi
echo ''

# 作業ディレクトリに移動
cd \$SLURM_SUBMIT_DIR
echo 'Working directory: \$(pwd)'
echo ''

# パッケージインストール（必要に応じて）
echo '=== Installing Dependencies ==='
pip install -r requirements_node.txt --user
echo ''

# Hugging Face ログイン確認
echo '=== Hugging Face Hub Check ==='
if huggingface-cli whoami > /dev/null 2>&1; then
    echo 'OK Logged in as: \$(huggingface-cli whoami)'
else
    echo 'WARNING Not logged in to Hugging Face Hub'
    echo '   Run: huggingface-cli login'
fi
echo ''

echo '=== Ready for DPO Data Generation ==='
echo 'To start generation, run:'
echo '  python gen_dpo_node.py'
echo ''
echo 'Or use the automatic script:'
echo '  bash run_dpo_generation.sh'
echo ''
echo 'To monitor GPU usage:'
echo '  watch -n 1 nvidia-smi'
echo ''

# 対話型シェルを開始
bash -i
"