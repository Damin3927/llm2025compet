#!/bin/bash
#SBATCH --job-name=qwen3_8gpu
#SBATCH --partition=P01
#SBATCH --nodelist=osk-gpu51
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/home/Competition2025/adm/X006/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/adm/X006/logs/%x-%j.err
#SBATCH --export=OPENAI_API_KEY='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'
#--- モジュール & Conda --------------------------------------------
module purge
module load cuda/12.6 miniconda/24.7.1-py312
module load cudnn/9.6.0  
module load nccl/2.24.3 
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmbench

# Hugging Face 認証
export HF_TOKEN= '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'
export HF_HOME=${SLURM_TMPDIR:-$HOME}/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
mkdir -p "$HF_HOME"
echo "HF cache dir : $HF_HOME"                   # デバッグ用

#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4,5,6,7 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（8GPU）----------------------------------------------
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 8 \
  --enable-reasoning \
  --reasoning-parser qwen3 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  > vllm.log 2>&1 &
pid_vllm=$!

#--- ヘルスチェック -------------------------------------------------
until curl -s http://127.0.0.1:8000/health >/dev/null; do
  echo "$(date +%T) vLLM starting …"
  sleep 10
done
echo "vLLM READY"

#--- 推論 -----------------------------------------------------------
python llm-compe-eval/evaluate_huggingface_models.py \
    --model_name "Qwen/Qwen3-32B" \
    --dataset_path datasets/Instruction/do_not_answer_en.csv \
    --output_dir evaluation_results \
    --use_vllm \
    --max_questions 100 \
    --vllm_base_url http://localhost:8000/v1 > predict.log 2>&1

#--- 後片付け -------------------------------------------------------
kill $pid_vllm
kill $pid_nvsmi
wait