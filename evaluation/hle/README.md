# 推論スクリプトの利用方法


以下ログインノード:
```bash
cd llm2025compet
uv pip install -e .

# サーバーを立ち上げる
pushd inference
./vllm_sbatch.sh --model "qwen/Qwen3-235B-A22B" --nodes 2 --gpus 8 --nodelist osk-gpu[54,56] --timeout 03:00:00
popd

# クライアントを立ち上げる
# 上記のノードを指定した場合は、osk-gpu54 が base_url になる
pushd evaluation/hle
python infer.py --help

python infer.py \
    --model "Qwen/Qwen3-235B-A22B" \
    --base_url "http://osk-gpu54:8000" \
    --num_workers 2 \
    --dataset_name "neko-llm/eval-Qwen-Qwen3-235B-A22B" \
    --push_to_hub

# これで vLLM サーバーが立ち上がるまで待ちながら、立ち上がったら推論が始まる
# 推論結果は、batch_size (default: 50) で指定した長さ分ごとに HF Datasets にアップロードされる
```

