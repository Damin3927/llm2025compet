# 物理問題CoT生成パイプライン

## 概要

HLE競技で高得点を獲得するため、公開データセットを高品質な学習データに変換するパイプラインです。

PHYBenchやPhysReasonなどの優れた公開データセットには問題・解法・答えが含まれています。本システムでは、これらの貴重なデータを活用し、元の解法を基に段階的な思考過程（Chain of Thought、CoT）を生成することで、LLMが推論プロセスを学習できる形式に拡張します。

3つのステップで処理：公開データセット取り込み → CoT生成

## 処理の流れ

### 1. 公開データセットの配置

まず、ダウンロードした公開データセットを所定の場所に配置します。

```bash
# PHYBenchの場合
data/phybench/original/PHYBench-fullques_v1.json
```

### 2. 公開データセットのインポート

配置した公開データセットをインポートし、処理しやすい形式に整えます。  
この段階で、問題・解法・答えが整理された形式になります。

```bash
cd physics/
python 1_import_phybench.py
# → data/phybench/preprocessed/dataset.jsonl が生成される
```

### 3. 思考過程（CoT）の生成

インポートしたデータの解法を基に、段階的な思考過程を生成します。

```bash
python 2_generate_cot.py phybench
# → data/phybench/generated/generated_cot_vX.X_YYYYMMDD_HHMM.jsonl が生成される
```

変換例：
```
[変換前] KE = 1/2 * m * v^2 = 1/2 * 5 * 10^2 = 250J
[変換後] 運動エネルギーを求める問題ですね。質量が5kg、速度が10m/s...
```

最終的な学習データ形式：
```json
{
    "id": 1,
    "question": "問題文",
    "output": "<think>思考過程...</think>最終的な答え",
    "answer": "正解"
}
```

## クイックスタート

### 環境準備

```bash
# リポジトリのルートディレクトリで実行
cd /path/to/llm2025compet

# uvをインストール
curl -LsSf https://astral.sh/uv/0.8.0/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc  # または新しいターミナルを開く

# 依存関係をインストール  
uv sync
```

### PHYBenchの処理例

```bash
# 物理問題データのディレクトリへ移動
cd data/hle/sft/physics

# APIキーの設定
echo "OPENROUTER_API_KEY=your-api-key-here" > .env

# 1. データ配置
mkdir -p data/phybench/original
# JSONファイルをdata/phybench/original/に配置（ファイル名は任意）

# 2. インポート
python 1_import_phybench.py

# 3. CoT生成
python 2_generate_cot.py phybench
```

※ `uv sync`実行後は、uvが自動的にプロジェクトの仮想環境を管理します。各pythonコマンドは自動的に正しい環境で実行されます。

