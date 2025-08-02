#!/bin/bash

# DPO データ生成実行スクリプト
# H100 x 8 環境での実行を想定

set -e  # エラー時に停止

# ログファイル設定
LOG_FILE="dpo_generation_$(date +%Y%m%d_%H%M%S).log"

echo "=== DPO Data Generation Starting ===" | tee $LOG_FILE
echo "Date: $(date)" | tee -a $LOG_FILE
echo "Hostname: $(hostname)" | tee -a $LOG_FILE
echo "GPU Count: $(nvidia-smi -L | wc -l)" | tee -a $LOG_FILE
echo "Available Memory: $(free -h | grep Mem | awk '{print $7}')" | tee -a $LOG_FILE
echo "Python Version: $(python --version)" | tee -a $LOG_FILE
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# GPU状態確認
echo "=== GPU Status ===" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 必要なファイルの存在確認
echo "=== File Check ===" | tee -a $LOG_FILE
if [[ -f "data/dna/questions.json" ]]; then
    echo "OK questions.json found ($(wc -l < data/dna/questions.json) lines)" | tee -a $LOG_FILE
else
    echo "ERROR questions.json not found!" | tee -a $LOG_FILE
    exit 1
fi

if [[ -f "data/dna/gen_dpo_node.py" ]]; then
    echo "OK gen_dpo_node.py found" | tee -a $LOG_FILE
else
    echo "ERROR gen_dpo_node.py not found!" | tee -a $LOG_FILE
    exit 1
fi

if [[ -f ".venv/bin/activate" ]]; then
    echo "OK Virtual environment found" | tee -a $LOG_FILE
else
    echo "ERROR Virtual environment not found! Run 'uv sync' first." | tee -a $LOG_FILE
    exit 1
fi
echo "" | tee -a $LOG_FILE

# 仮想環境をアクティベート
echo "=== Activating Virtual Environment ===" | tee -a $LOG_FILE
source .venv/bin/activate
echo "Virtual environment activated: $(which python)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 依存関係の確認（uv環境では事前にインストール済み）
echo "=== Checking Dependencies ===" | tee -a $LOG_FILE
python -c "import torch, transformers, accelerate, bitsandbytes, datasets, tqdm; print('OK All dependencies available')" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# Hugging Face Hubへのログイン確認
echo "=== Hugging Face Hub Check ===" | tee -a $LOG_FILE
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "OK Logged in to Hugging Face Hub as: $(huggingface-cli whoami)" | tee -a $LOG_FILE
else
    echo "WARNING Not logged in to Hugging Face Hub. Run 'huggingface-cli login' if needed." | tee -a $LOG_FILE
fi
echo "" | tee -a $LOG_FILE

# 既存の出力ファイル確認
if [[ -f "dpo_dataset_405b.jsonl" ]]; then
    EXISTING_LINES=$(wc -l < dpo_dataset_405b.jsonl)
    echo "INFO Found existing output file with $EXISTING_LINES processed items" | tee -a $LOG_FILE
    echo "   Will resume from where it left off..." | tee -a $LOG_FILE
else
    echo "INFO No existing output file found. Starting fresh..." | tee -a $LOG_FILE
fi
echo "" | tee -a $LOG_FILE

# DPO データ生成を実行
echo "=== Starting DPO Data Generation ===" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 作業ディレクトリをdata/dnaに変更
cd data/dna

# Python実行とログ記録
python gen_dpo_node.py 2>&1 | tee -a ../../$LOG_FILE

# 実行結果の確認
EXIT_CODE=${PIPESTATUS[0]}
echo "" | tee -a $LOG_FILE
echo "=== Execution Results ===" | tee -a $LOG_FILE
echo "End time: $(date)" | tee -a $LOG_FILE
echo "Exit code: $EXIT_CODE" | tee -a $LOG_FILE

if [[ -f "dpo_dataset_405b.jsonl" ]]; then
    FINAL_LINES=$(wc -l < dpo_dataset_405b.jsonl)
    echo "STATS Final output: $FINAL_LINES processed items" | tee -a ../../$LOG_FILE
    echo "   Output file: data/dna/dpo_dataset_405b.jsonl" | tee -a ../../$LOG_FILE
fi

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS DPO Data Generation completed successfully!" | tee -a $LOG_FILE
else
    echo "ERROR DPO Data Generation failed with exit code $EXIT_CODE" | tee -a $LOG_FILE
fi

echo "LOG Full log saved to: $LOG_FILE" | tee -a $LOG_FILE
echo "=== Script Finished ===" | tee -a $LOG_FILE