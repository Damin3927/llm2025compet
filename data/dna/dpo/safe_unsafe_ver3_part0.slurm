#!/bin/bash
#SBATCH --job-name=qwen3_vllm0
#SBATCH --partition=P02
#SBATCH --nodelist=osk-gpu54       # ← ノードを固定！
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # GPU 8枚
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=00:50:00
#SBATCH --output=logs/qwen3_vllm_%j.out
#SBATCH --error=logs/qwen3_vllm_%j.err

echo "🟢 ノード: $(hostname)"
nvidia-smi

python safe_unsafe_ver3_part0.py
