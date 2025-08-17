# 推論スクリプトの利用方法


以下ログインノード:
```bash
cd llm2025compet
uv pip install -e .

# サーバーを立ち上げる (複数サーバー立ち上げたい場合は、複数回実行)
pushd inference
./vllm_sbatch.sh \
    --model "qwen/Qwen3-235B-A22B" \
    --nodes 3 \
    --gpus 8 \
    --nodelist osk-gpu[54,56,91] \
    --timeout 03:00:00 \
    --expert-parallel
popd

# Dense + LoRA の例（Qwen3-32B）
pushd inference
./vllm_sbatch.sh \
    --model "Qwen/Qwen3-32B" \
    --nodes 3 \
    --gpus 8 \
    --nodelist osk-gpu[54,56,91] \
    --timeout 03:00:00 \
    --lora 'neko=neko-llm/Qwen3-32B-test-lora'
popd

# 指定した vLLM サーバー(単一/複数)が立ち上がるまで待ち、立ち上がったら推論が始まる
pushd evaluation/hle
python infer.py --help

## 単一サーバーの場合
python infer.py \
    --model "qwen/Qwen3-235B-A22B" \
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
    --base_urls http://osk-gpu54:8000,http://osk-gpu56:8000,http://osk-gpu91:8000 \
    --num_nodes 3 \
    --num_workers 40 \
    --flush_every 60 \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub

# 既存の JSON アップロードのみする場合
# (推論は行わず、predictions/hle_*.json を Datasets として Hub に公開)
python infer.py \
    --model "Qwen/Qwen3-235B-A22B" \
    --upload_only \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --input_json predictions/hle_Qwen-Qwen3-235B-A22B.json

# JSON は eval.py と互換な形式なので、このまま判定にかけられる
python eval.py \
    --predictions_dataset "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --num_workers 100 \
    --judge o3-mini-2025-01-31

# 判定結果 (judged_*.json) を Hugging Face にアップロード
python upload_eval_result.py \
    --input_json judged_hle_Qwen-Qwen3-235B-A22B.json \
    --dataset_name neko-llm/eval-judged-Qwen-Qwen3-235B-A22B
```
