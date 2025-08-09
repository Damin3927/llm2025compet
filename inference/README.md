# vLLM 推論サーバー

SLURM 上で vLLM 推論サーバーを起動するためのスクリプト

## Usage

以下をログインノードで実行：

```bash
# ノードを2つ、GPU をそれぞれ 8 つずつ使い、DeepSeek-R1-0528 の推論サーバーを立ち上げる
# 立ち上がるのに、最大約 10 分ほどかかると思われます

# Qwen3-235B-A22B
./vllm_sbatch.sh --model "qwen/Qwen3-235B-A22B" --nodes 2 --gpus 8 --nodelist osk-gpu[54,56] --timeout 01:00:00

# DeepSeek-R1-0528
./vllm_sbatch.sh --model "deepseek-ai/DeepSeek-R1-0528" --nodes 2 --gpus 8 --nodelist osk-gpu[54,56] --timeout 01:00:00
```

上記の後、計算ノードでサーバーが立ち上がり始めますので、`logs` に生成されたログを見ながら気長に待ってください。もしくは、スクリプトで API が通常通り動き始めるまでリクエストを送ってみて待つ処理を入れてください。

### Python で待つ場合

```python
import requests
import time
from openai import OpenAI

# ./vllm_batch.sh で最後に表示された IP に変える
SERVER_URL = "http://10.255.255.xx:8000"
# もしくは、ヘッドノード名（e.g. osk-gpu54）を用いても良い
SERVER_URL = "http://osk-gpuxx:8000"

def wait_until_vllm_up():
    ping_url = f"{SERVER_URL}/ping"
    while True:
        try:
            response = requests.get(ping_url, timeout=60)
            if response.status_code == 200:
                # The PONG response from vllm is just the integer 200
                break
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            time.sleep(10)

    print("vLLM Server is now up")


if __name__ == "__main__":
    wait_until_vllm_up()

    client = OpenAI(
        base_url=f"{SERVER_URL}/v1",
        api_key="token-abc123",
    )
    ...
```

### ログを見て待つ場合
`Starting vLLM API server on http://0.0.0.0:8000` の表示が確認されると、API リクエストを送れるようになります。
