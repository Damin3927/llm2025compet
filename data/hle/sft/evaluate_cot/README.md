# CoT評価・再生成ツール

## 概要

任意のデータセットに含まれるCoT（Chain of Thought）を評価し、低品質なものを改善・再生成するツールセット。

## ツール構成

### 1. 評価ツール（1_evaluate_cot.py）

任意のデータセットに含まれるCoTを評価し、学習データとしての品質をグレード（A〜D）で判定する。

**必須要件**（一つでも満たさない場合はグレードD）：
- 独立性：外部参照なしで問題を解いているか
- 論理的完全性：論理の飛躍なく結論まで到達しているか  
- 正確性：原理や手法が正しく適用されているか
- 解答到達：正しい答えに到達しているか

**学習価値スコア**（各10点満点の6観点）：
- 方法選択の説明
- 段階的な導出
- 検証と確認
- よくある誤りへの対処
- 問題領域の洞察
- メタ認知的要素

**グレード判定**：
- A: 優秀な学習データ（平均8.0点以上）
- B: 良好な学習データ（平均6.0点以上）
- C: 使用可能な学習データ（平均4.0点以上）
- D: 学習データとして不適切（必須要件未達成または平均4.0点未満）

### 2. 再生成ツール（2_regenerate_cot.py）

低品質なCoTを評価結果に基づいて改善・再生成する。評価で特定された強みを維持しながら弱点を改善することで、より高品質な学習データへの変換を行う。デフォルトではグレードC以下を対象とするが、グレード指定やID指定により処理対象を制御できる。

## 必要な環境

### Python環境
uv環境（Python 3.12）で実行

### APIキー設定
プロジェクトルートの`.env`ファイルに設定：
```
OPENROUTER_API_KEY=your-api-key-here
```

## 使用方法

### ステップ1: 評価

```bash
# データセット全体を評価
uv run python 1_evaluate_cot.py --dataset dataset.jsonl

# 特定のIDのみ評価
uv run python 1_evaluate_cot.py --dataset dataset.jsonl --ids 1,5,17
```

### ステップ2: 再生成

```bash
# グレードC以下を再生成（デフォルト）
uv run python 2_regenerate_cot.py --dataset dataset.jsonl

# グレードB以下を再生成
uv run python 2_regenerate_cot.py --dataset dataset.jsonl --grade B

# 特定IDのみ再生成
uv run python 2_regenerate_cot.py --dataset dataset.jsonl --ids 1,5,17
```

### ステップ3: 再評価

```bash
# 再生成されたCoTを評価
uv run python 1_evaluate_cot.py --dataset dataset.jsonl
```

## データ形式

### 入力データの形式（最初に必要なデータ）

```json
{
  "id": 1,
  "question": "問題文",
  "output": "CoT（思考過程）",
  "answer": "正解"
}
```

### 評価後のデータ形式（1_evaluate_cot.py実行後）

元のJSONLファイルにmetadataフィールドが追記される：

```json
{
  "id": 1,
  "question": "...",
  "output": "...",
  "answer": "...",
  "metadata": {
    "cot_history": [{
      "timestamp": "2025-01-01T10:00:00",
      "output": "評価時のCoT",
      "evaluation": {
        "grade": "B",
        "score": 6.5,
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

### 再生成後のデータ形式（2_regenerate_cot.py実行後）

再生成後、新しいエントリがcot_historyに追加され、outputフィールドが更新される：

```json
{
  "output": "新しいCoT",  // 更新される
  "metadata": {
    "cot_history": [
      {既存の評価済みエントリ},
      {
        "timestamp": "2025-01-01T11:00:00",
        "output": "新しいCoT",
        "evaluation": null  // 未評価
      }
    ]
  }
}
```

## 典型的なワークフロー

1. **初回評価**: 全データのCoTを評価
2. **改善**: グレードC/D のCoTを再生成
3. **再評価**: 再生成されたCoTを評価
4. **繰り返し**: 必要に応じて2-3を繰り返す

## 注意事項

- 元のJSONLファイルを直接更新するため、バックアップを推奨
- 評価・再生成にはOpenRouter APIキーが必要