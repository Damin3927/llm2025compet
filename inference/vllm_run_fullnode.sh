#!/bin/bash
#SBATCH --job-name=vllm-multinode
#SBATCH --nodes=3
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=P02

srun --ntasks=3 --gres=gpu:8 --exclusive ./vllm_start.sh
