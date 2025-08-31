#!/bin/bash
################### Slurm 基本設定 ###################
#SBATCH --partition=P02
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8           # 1ノード内の全 GPU を使用
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=osk-gpu[56]
#SBATCH --job-name=grpo-qwen1_5b-colo1n
#SBATCH --time=4:00:00
#SBATCH --mem=0
#SBATCH --output=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b_colo1n.out
#SBATCH --error=/home/Competition2025/P02/P02U017/llm2025compet/training/logs/grpo-qwen1_5b_colo1n.err

################### 環境 ###################
export WANDB_DISABLED=true
module load cuda/12.8
source ~/openr1/bin/activate

# vLLM／NCCL の安定用フラグ
export TRL_UPDATE_NAMED_PARAM_CONCURRENCY=4   # 重み同期の並列度
export NCCL_ASYNC_ERROR_HANDLING=1            # NCCL ハング対策

ulimit -v unlimited
ulimit -m unlimited

REPO_DIR=/home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src
cd "$REPO_DIR" || exit 1

################### GRPO Trainer（colocate・1ノード） ###################
srun --nodes=1 --ntasks=1 --gres=gpu:8 --exclusive --chdir="$REPO_DIR" \
     bash -c "
       source ~/openr1/bin/activate
       echo '[GRPO-Colo1N] on \$HOSTNAME'
       accelerate launch \\
         --config_file ../recipes/accelerate_configs/zero3.yaml \\
         --num_machines 1 \\
         --num_processes 8 \\
         --main_process_port 29500 \\
         /home/Competition2025/P02/P02U017/llm2025compet/training/open-r1/src/open_r1/grpo.py \\
         --config /home/Competition2025/P02/P02U017/llm2025compet/training/configs/Qwen3-32b/grpo/config_grpo_1.5b.yaml \\
         --use_vllm true \\
         --vllm_mode colocate
     "

wait
echo '[Job] colocated 1-node training finished.'
