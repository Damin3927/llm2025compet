# LoRA Merge Tool

```bash
sbatch run.sh \
    --base-model Qwen/Qwen3-235B-A22B \
    --adapters neko-llm/adapter-repo \
    --out-dir merged/qwen3-235b-a22b \
    --offload-dir offload/qwen3-235b-a22b \
    --upload-to-hub \
    --hub-repo-id neko-llm/Qwen3-235B-A22B-merged
```
