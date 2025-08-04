# 物理問題CoT生成パイプライン

## 概要

公開データセット（PHYBench、PhysReason等）の問題・解法・答えから、段階的な思考過程（Chain of Thought、CoT）を生成し、LLMの学習データに変換するパイプライン。

5段階の処理パイプライン：
1. **インポート**：公開データセットを統一形式に変換
2. **CoT生成**：思考過程を含む学習データを生成
3. **CoT品質評価**：生成された思考過程の学習価値を評価
4. **CoT再生成**：改善提案に基づき思考過程を再生成
5. **アップロード**：生成データをHugging Faceに公開

## ファイル構成

```
physics/
├── 1_import_phybench.py       # PhyBench用データインポート
├── 1_import_physreason.py     # PhysReason用インポート
├── 2_generate_cot.py          # CoT生成
├── 3_evaluate_cot.py          # CoT品質評価
├── 4_regenerate_cot.py        # CoT再生成
├── 5_upload_data.py           # HFアップロード
├── generate_and_evaluate.sh   # 生成と評価の一括実行
├── regenerate_and_evaluate.sh # 再生成と評価の一括実行
├── prompts/
│   ├── registry.yaml          # プロンプトバージョン管理
│   └── versions/
│       ├── generate_v3.0.yaml
│       └── regenerate_v3.0.yaml
└── data/
    └── phybench/
        ├── original/          # 元データ
        ├── preprocessed/      # 前処理済み
        └── generated/         # 生成結果
```

## 環境準備

### 必要なツールのインストール

```bash
# リポジトリのルートディレクトリで実行
cd /path/to/llm2025compet

# uvをインストール（Python依存関係管理ツール）
curl -LsSf https://astral.sh/uv/0.8.0/install.sh | sh
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
source ~/.bashrc  # または新しいターミナルを開く

# 依存関係をインストール
uv sync
```

### APIキーの設定

```bash
# 物理問題データのディレクトリへ移動
cd data/hle/sft/physics

# OpenRouter APIキーを設定
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

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

### 3. CoTの生成

インポートしたデータの解法を基に、CoTを生成します。  

```bash
python 2_generate_cot.py phybench --prompt-version 3.0
# → data/phybench/generated/generated_cot_v3.0_YYYYMMDD_HHMM.jsonl が生成される
```

変換例：
```
[変換前] KE = 1/2 * m * v^2 = 1/2 * 5 * 10^2 = 250J
[変換後] As a physics researcher, I find this problem fascinating because...
         運動エネルギーを求める問題ですね。質量が5kg、速度が10m/s...
```

### 4. CoTの品質評価

生成されたCoTの学習価値を評価し、グレード（A-D）を付与します。

```bash
python 3_evaluate_cot.py phybench
# → 各データにevaluationフィールドが追加される
```

評価基準：
- **必須要件**：独立性、論理的完全性、正確性、解答到達
- **学習価値**：6観点（方法選択、段階的導出、検証、誤り対処、物理的洞察、メタ認知）
- **グレード**：A（8.0点以上）、B（6.0点以上）、C（4.0点以上）、D（不適格）

### 5. CoTの再生成

改善提案に基づいてCoTを再生成します。

```bash
python 4_regenerate_cot.py phybench --prompt-version 3.0
# → 各データのcot_historyに改善された思考過程が追加される
```

### 6. Hugging Faceへのアップロード

生成データをHugging Face Hubに公開します。

```bash
python 5_upload_data.py phybench
# → https://huggingface.co/datasets/neko-llm/HLE_SFT_PHYBench に公開
```

最終的な学習データ形式：
```json
{
    "id": 1,
    "question": "問題文",
    "output": "<think>思考過程...</think>最終的な答え",
    "answer": "正解",
    "metadata": {
        "original_solution": "元の解答",
        "cot_history": [{
            "timestamp": "2025-08-02T04:47:00",
            "output": "生成されたCoT",
            "evaluation": {
                "grade": "A",
                "score": 8.5,
                "passed_requirements": {...},
                "learning_value_scores": {...},
                "strengths": [...],
                "weaknesses": [...],
                "improvement_suggestions": [...]
            }
        }]
    }
}
```
