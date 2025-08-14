# 推論スクリプトの利用方法


以下ログインノード:
```bash
cd llm2025compet
uv pip install -e .

# サーバーを立ち上げる
pushd inference
./vllm_sbatch.sh --model "qwen/Qwen3-235B-A22B" --nodes 3 --gpus 8 --nodelist osk-gpu[54,56,91] --timeout 03:00:00
popd

# クライアントを立ち上げる
# 上記のノードを指定した場合は、osk-gpu54 が base_url になる
pushd evaluation/hle
python infer.py --help

python infer.py \
    --model "qwen/Qwen3-235B-A22B" \
    --base_url "http://osk-gpu54:8000" \
    --num_workers 40 \
    --flush_every 40 \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub

# これで vLLM サーバーが立ち上がるまで待ちながら、立ち上がったら推論が始まる
# 推論結果は predictions/hle_<モデル名ベース>.json に JSON 辞書形式 (id -> {model, response, usage}) で保存される
# --flush_every で新規結果を書き出す頻度を制御 (デフォルト 20)。--push_to_hub と --dataset_name を指定した場合は、最後に HF Datasets にアップロードされる

# 既存の JSON アップロードのみする場合
# (推論は行わず、predictions/hle_*.json を Datasets として Hub に公開)
python infer.py \
    --model "Qwen/Qwen3-235B-A22B" \
    --upload_only \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --input_json predictions/hle_Qwen-Qwen3-235B-A22B.json

# JSON は eval.py と互換な形式なので、このまま判定にかけられる
python eval.py \
    --prediction_dataset "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --num_workers 100 \
    --judge o3-mini-2025-01-31

# 判定結果 (judged_*.json) を Hugging Face にアップロード
python upload_eval_result.py \
    --input_json judged_hle_Qwen-Qwen3-235B-A22B.json \
    --dataset_name neko-llm/eval-judged-Qwen-Qwen3-235B-A22B
```
