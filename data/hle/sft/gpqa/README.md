# HLE SFT GPQA Diamond Dataset

## 概要

このデータセットは、GPQA (Graduate-level Google-proof Q&A) Diamond データセットを基に、Chain of Thought (CoT) 推論を追加して生成されたSupervised Fine-Tuning (SFT) 用のデータセットです。

専門的な科学分野（物理学、化学、生物学）における高度な質問に対して、段階的な推論プロセスを含む回答を提供します。

## データセット統計

- **総問題数**: 198 問
- **成功生成数**: 61 問
- **成功率**: 30.8%

## ファイル形式

このデータセットは以下の3つの形式で提供されています：

### 1. CSV形式 (`gpqa_diamond_cot_dataset.csv`)
- 一般的な表形式データ
- Excel やスプレッドシートソフトで開けます
- Pandas で簡単に読み込み可能

### 2. Parquet形式 (`gpqa_diamond_cot_dataset.parquet`)
- 高効率なカラム型データ形式
- 大規模データの高速読み込みに最適
- Apache Arrow エコシステムで推奨

### 3. JSONL形式 (`gpqa_diamond_cot_dataset.jsonl`)
- JSON Lines 形式（1行1レコード）
- ストリーミング処理に適している
- 機械学習フレームワークで広く対応

## データ構造

各レコードには以下のフィールドが含まれています：

| フィールド名 | 型 | 説明 |
|-------------|----|----|
| `id` | int | 問題の一意識別子 |
| `question` | str | 元の質問文 |
| `output` | str | CoT推論過程を含む完全な回答（&lt;think&gt;...&lt;/think&gt;形式） |
| `answer` | str | 最終的な正解 |
| `generated_answer` | str | generated_answer |
| `judgment_status` | str | judgment_status |

## サンプルデータ

```json
{
  "id": 2,
  "question": "trans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.\n\n1 was treated with pyridinium chlorochromate, forming product 2.\n\n3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.\n\nhow many carbon atoms are there in product 3?",
  "output": "<think>trans-Cinnamaldehyde (C₉H₈O) reacts with CH₃MgBr to form (E)-4-phenylbut-3-en-2-ol (product 1, C₁₀H₁₂O), adding one carbon. Product 1 is oxidized with PCC to form (E)-4-phenylbut-3-en-2-one (product 2, C₁₀H₁₀O), with no change in carbon count. Product 2 undergoes cyclopropanation with dimethyl sulfoxonium methylide, adding one carbon to form 1-(2-phenylcyclopropyl)ethan-1-one (product 3, C₁₁H₁₂O). Thus, product 3 has 11 carbon atoms.</think>11",
  "answer": "11",
  "generated_answer": "11",
  "judgment_status": "correct"
}
```

## 使用方法

### Python (Pandas)

```python
import pandas as pd

# CSV形式で読み込み
df = pd.read_csv('gpqa_diamond_cot_dataset.csv')

# Parquet形式で読み込み（推奨）
df = pd.read_parquet('gpqa_diamond_cot_dataset.parquet')

# JSONL形式で読み込み
df = pd.read_json('gpqa_diamond_cot_dataset.jsonl', lines=True)

print(f"データセットサイズ: {len(df)} 問")
print(df.head())
```

### Hugging Face Datasets

```python
from datasets import load_dataset

# このリポジトリから直接読み込み
dataset = load_dataset("neko-llm/HLE_SFT_GPQA_Diamond")

# 特定のファイル形式を指定
dataset = load_dataset("neko-llm/HLE_SFT_GPQA_Diamond", data_files="gpqa_diamond_cot_dataset.parquet")
```

### PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GPQADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'question': row['question'],
            'output': row['output'],
            'answer': row['answer']
        }

# データ読み込み
df = pd.read_parquet('gpqa_diamond_cot_dataset.parquet')
dataset = GPQADataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 生成方法

このデータセットは以下の手順で生成されました：

1. **元データ**: [GPQA Diamond Dataset](https://huggingface.co/datasets/Idavidrein/gpqa) の train split を使用
2. **CoT生成**: DeepSeek-R1-0528:free モデルを使用して推論過程を生成
3. **フォーマット**: `<think>推論過程</think>最終回答` の形式で構造化
4. **品質管理**: API呼び出しの成功/失敗を記録し、品質を担保

### 生成に使用したモデル
- **モデル**: `deepseek/deepseek-r1-0528:free`
- **API**: OpenRouter API
- **生成方式**: JSON構造化プロンプトによる推論生成
- **出力形式**: `{"reasoning": "推論過程", "answer": "回答"}`のJSON形式

## ライセンス

このデータセットは元のGPQAデータセットのライセンスに従います。学術研究目的での使用を推奨します。

## 引用

このデータセットを使用する場合は、以下を引用してください：

```bibtex
@dataset{hle_sft_gpqa_diamond,
  title={HLE SFT GPQA Diamond Dataset with Chain of Thought},
  author={neko-llm},
  year={2024},
  url={https://huggingface.co/datasets/neko-llm/HLE_SFT_GPQA_Diamond}
}
```

元のGPQAデータセットの引用：
```bibtex
@article{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q&A Benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  journal={arXiv preprint arXiv:2311.12022},
  year={2023}
}
```

## 問い合わせ

データセットに関する質問や問題がある場合は、[Issues](https://huggingface.co/datasets/neko-llm/HLE_SFT_GPQA_Diamond/discussions) でお知らせください。
