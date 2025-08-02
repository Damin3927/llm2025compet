#!/bin/bash

# SLURM対応 DPO データ生成スクリプト
# さくらDCサーバ H100 x 8 環境での実行用

#SBATCH --job-name=dpo_gen_405b
#SBATCH --partition=P02  # 本田大明チーム
#SBATCH --nodes=1
#SBATCH --nodelist=osk-gpu54  # P02チーム用ノード
#SBATCH --gres=gpu:8      # H100 x 8基を使用
#SBATCH --time=168:00:00  # 最大7日間（必要に応じて調整）
#SBATCH --output=/home/Competition2025/P02/shareP02/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/shareP02/logs/%x-%j.err

# ジョブ開始ログ
echo "=== SLURM DPO Data Generation Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo ""

# CUDA環境設定
module load cuda/12.4
echo "CUDA module loaded"

# Conda環境の設定（必要に応じて）
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv
# echo "Conda environment: $CONDA_DEFAULT_ENV"

# 環境変数設定
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export HF_HOME="/home/Competition2025/P02/shareP02/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/Competition2025/P02/shareP02/.cache/transformers"

# 環境情報表示
echo "=== Environment Information ==="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Python: $(which python)"
echo "Python Version: $(python --version)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Available GPUs: $(nvidia-smi -L | wc -l)"
echo "Total GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024 " GB"}')"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# GPUステータス確認
echo "=== Initial GPU Status ==="
nvidia-smi
echo ""

# 作業ディレクトリに移動
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# 必要なファイルの確認
echo "=== File Check ==="
for file in "data/dna/questions.json" "data/dna/gen_dpo_node.py"; do
    if [[ -f "$file" ]]; then
        echo "OK $file found"
    else
        echo "ERROR $file not found!"
        exit 1
    fi
done
echo ""

# Python仮想環境のセットアップ
echo "=== Setting up Python Virtual Environment ==="
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "Virtual environment activated: $(which python)"
else
    echo " Virtual environment not found!"
    echo "Please run 'uv sync' to create the virtual environment"
    exit 1
fi
echo ""

# 依存関係の確認
echo "=== Checking Dependencies ==="
python -c "import torch, transformers, accelerate, bitsandbytes, datasets, tqdm; print('OK All dependencies available')"
echo "Dependencies check completed"
echo ""

# Hugging Face Hub ログイン確認
echo "=== Hugging Face Hub Check ==="
if huggingface-cli whoami > /dev/null 2>&1; then
    echo "OK Logged in as: $(huggingface-cli whoami)"
else
    echo "WARNING Not logged in to Hugging Face Hub"
    echo "   To login, run: huggingface-cli login"
    echo "   Or set token: export HUGGINGFACE_HUB_TOKEN=your_token_here"
    # 環境変数からトークンを読み込む場合
    if [[ -n "$HUGGINGFACE_HUB_TOKEN" ]]; then
        echo "   Using token from environment variable"
    fi
fi
echo ""

# 既存の出力確認
if [[ -f "dpo_dataset_405b.jsonl" ]]; then
    EXISTING_COUNT=$(wc -l < dpo_dataset_405b.jsonl)
    echo "INFO Found existing output: $EXISTING_COUNT processed items"
    echo "   Will resume processing..."
else
    echo "INFO Starting fresh generation..."
fi
echo ""

# メモリとディスク使用量チェック
echo "=== Resource Check ==="
echo "Memory usage: $(free -h | grep Mem)"
echo "Disk usage: $(df -h . | tail -1)"
echo ""

# DPO データ生成実行
echo "=== Starting DPO Data Generation ==="
echo "Execution start time: $(date)"
echo ""

# GPU使用率監視を背景で開始
nvidia-smi dmon -s pucvmet -d 60 > /home/Competition2025/P02/shareP02/logs/gpu_monitor_${SLURM_JOB_ID}.log &
MONITOR_PID=$!

# 作業ディレクトリをdata/dnaに変更
cd data/dna

# メイン処理実行
python gen_dpo_node.py

# 実行結果確認
EXIT_CODE=$?
echo ""
echo "=== Execution Completed ==="
echo "Execution end time: $(date)"
echo "Exit code: $EXIT_CODE"

# GPU監視停止
kill $MONITOR_PID 2>/dev/null || true

# 最終結果統計
if [[ -f "dpo_dataset_405b.jsonl" ]]; then
    FINAL_COUNT=$(wc -l < dpo_dataset_405b.jsonl)
    FILE_SIZE=$(du -h dpo_dataset_405b.jsonl | cut -f1)
    echo "STATS Final statistics:"
    echo "   - Processed items: $FINAL_COUNT"
    echo "   - Output file size: $FILE_SIZE"
    echo "   - Output file: dpo_dataset_405b.jsonl"
fi

# 最終GPU状態
echo ""
echo "=== Final GPU Status ==="
nvidia-smi
echo ""

# ジョブ終了ログ
echo "=== Job Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "End Time: $(date)"
echo "Duration: $SECONDS seconds"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS DPO Data Generation completed successfully!"
else
    echo "ERROR DPO Data Generation failed with exit code $EXIT_CODE"
    echo "Check error log: dpo_gen_${SLURM_JOB_ID}.err"
fi

echo "LOG Output log: /home/Competition2025/P02/shareP02/logs/dpo_gen_405b-${SLURM_JOB_ID}.out"
echo "LOG Error log: /home/Competition2025/P02/shareP02/logs/dpo_gen_405b-${SLURM_JOB_ID}.err"
echo "LOG GPU monitoring log: /home/Competition2025/P02/shareP02/logs/gpu_monitor_${SLURM_JOB_ID}.log"

# キャンセル用スクリプトの情報
echo ""
echo "=== Job Management Information ==="
echo "To cancel this job: bash /home/Competition2025/P02/shareP02/scripts/scancel.sh $SLURM_JOB_ID"
echo "To check job status: squeue -j $SLURM_JOB_ID"
echo "To check all jobs: squeue -u $(whoami)"

echo "=== SLURM Job Finished ==="

exit $EXIT_CODE