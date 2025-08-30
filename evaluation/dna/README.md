# LLM Competition Evaluation with Do-Not-Answer

以下ログインノード:
```bash
cd llm2025compet
uv pip install -e .

# サーバーを立ち上げる (複数サーバー立ち上げたい場合は、複数回実行)
pushd inference
./vllm_sbatch.sh --model "qwen/Qwen3-235B-A22B" --nodes 3 --gpus 8 --nodelist osk-gpu[54,56,91] --timeout 03:00:00
popd

# クライアントを立ち上げる
pushd evaluation/dna
python infer.py --help

## 単一サーバーの場合
python infer.py \
    --model "qwen/Qwen3-235B-A22B" \
    --dataset_path ./datasets/Instruction/do_not_answer_en.csv \
    --base_url "http://osk-gpu54:8000" \
    --num_workers 40 \
    --flush_every 40 \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub

## 複数サーバーの場合
# --base_urls に各ノードのベースURLを指定し、--num_workers は「各ノードあたり」のワーカー数になります。
# 例: 3ノード × 40ワーカー = 合計120並列
python infer.py \
    --model "qwen/Qwen3-235B-A22B" \
    --dataset_path ./datasets/Instruction/do_not_answer_en.csv \
    --base_urls http://osk-gpu54:8000,http://osk-gpu56:8000,http://osk-gpu91:8000 \
    --num_nodes 3 \
    --num_workers 40 \
    --flush_every 60 \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub

# 指定した vLLM サーバー(単一/複数)が立ち上がるまで待ち、立ち上がったら推論が始まる

# 既存の JSON アップロードのみする場合
# (推論は行わず、predictions/hle_*.json を Datasets として Hub に公開)
python infer.py \
    --model "qwen/Qwen3-235B-A22B" \
    --dataset_path ./datasets/Instruction/do_not_answer_en.csv \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub \
    --upload_only

# JSON は eval.py と互換な形式なので、このまま判定にかけられる
python eval.py \
    --dataset_path ./datasets/Instruction/do_not_answer_en.csv \
    --predictions_dataset "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --wandb_project neko-llm-test \
    --num_workers 10
```


