# vLLM 推論サーバー

このディレクトリには、SLURMクラスター上でvLLM推論サーバーを起動するためのスクリプトが含まれています。

## ファイル構成

- `vllm_sbatch.sh` - SLURMジョブを投入するメインスクリプト
- `vllm_start.sh` - vLLMサーバーを起動する実行スクリプト (直接実行はしません)
- `client_example.py` - APIクライアントのサンプル

## vllm_sbatch.sh の使用方法

### 基本的な使用方法

```bash
# ノードを2つ、GPU をそれぞれ 4 つずつ使い、Qwen3-235B-A22B の推論サーバーを立ち上げる
./vllm_sbatch.sh --model "Qwen/Qwen3-235B-A22B" --nodes 2 --gpus 4 --nodelist osk-gpu[54,91] --timeout 01:00:00

# ヘルプを表示
./vllm_sbatch.sh --help
```

### ジョブ名の自動生成

スクリプトは以下の形式でジョブ名を自動生成します：

```
vllm-{モデル名}-n{ノード数}-g{GPU数}
```

例：
- `vllm-Qwen3-32B-n1-g2`

### 出力ファイル

ジョブの出力は以下のファイルに保存されます：

- `{ジョブ名}-{ジョブID}.out` - 標準出力
- `{ジョブ名}-{ジョブID}.err` - エラー出力

### ジョブの監視と管理

```bash
# 実行中のジョブを確認
squeue -u $USER

# ジョブの詳細を確認
scontrol show job <ジョブID>

# ジョブをキャンセル
scancel <ジョブID>

# ジョブの出力をリアルタイムで確認
tail -f <ジョブ名>-<ジョブID>.out
```
