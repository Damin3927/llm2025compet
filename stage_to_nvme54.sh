#!/bin/bash
#SBATCH --partition=P02
#SBATCH --job-name=stage-nvme54
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/Competition2025/P02/P02U006/ColossalAI/logs/%x-%j.out
#SBATCH --error=/home/Competition2025/P02/P02U006/ColossalAI/logs/%x-%j.err
#SBATCH --nodelist=osk-gpu54

set -euo pipefail

SRC_MODEL="/home/Competition2025/P02/shareP02/DeepSeek-R1-0528-BF16"
DST_ROOT="/nvme54/models/$USER"
DST_MODEL="$DST_ROOT/DeepSeek-R1-0528-BF16"

echo "[INFO] $(hostname): staging to $DST_MODEL"

[ -d /nvme54 ] && [ -w /nvme54 ] || { echo "[ERROR] /nvme54 not present or not writable"; exit 1; }

mkdir -p "$DST_MODEL"
rsync -aH --info=progress2 "$SRC_MODEL"/ "$DST_MODEL"/

# 簡易チェック
MISS=0
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json vocab.json merges.txt; do
  [ -f "$DST_MODEL/$f" ] || { echo "[WARN] missing $f"; MISS=1; }
done
[ "$MISS" -eq 0 ] && echo "[OK] tokenizer files present"

du -sh "$DST_MODEL"
