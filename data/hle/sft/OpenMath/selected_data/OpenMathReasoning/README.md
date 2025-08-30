---
configs:
- config_name: genselect
  data_files: "data/genselect_OpenMathReasoning_genselect_5000_samples.parquet"
- config_name: cot
  data_files: "data/cot_OpenMathReasoning_cot_5000_samples.parquet"
---

# SFT_OpenMathReasoning

## データセットの説明

このデータセットは、以下の分割（split）ごとに整理された処理済みデータを含みます。

- **genselect**: 1 JSON files, 1 Parquet files
- **cot**: 1 JSON files, 1 Parquet files

## データセット構成

各 split は JSON 形式と Parquet 形式の両方で利用可能です:
- **JSONファイル**: 各 split 用サブフォルダ内の元データ（`cot/`）
- **Parquetファイル**: split名をプレフィックスとした最適化データ（`data/cot_*.parquet`）

各 JSON ファイルには、同名の split プレフィックス付き Parquet ファイルが対応しており、大規模データセットの効率的な処理が可能です。

## 使い方

```python
from datasets import load_dataset

# 特定の split を読み込む
genselect_data = load_dataset("neko-llm/SFT_OpenMathReasoning", "genselect")
cot_data = load_dataset("neko-llm/SFT_OpenMathReasoning", "cot")

# または、data_files を手動で指定して読み込む
dataset = load_dataset(
    "parquet",
    data_files={
        "genselect": "data/genselect_*.parquet",
        "cot": "data/cot_*.parquet",
    }
)

# 個別のファイルを読み込む
import pandas as pd
df = pd.read_parquet("data/genselect_filename.parquet")

# Load all files for a specific split
from pathlib import Path
split_files = list(Path("data").glob("genselect_*.parquet"))
for file in split_files:
    df = pd.read_parquet(file)
    # Process df...
```

## ファイル構成

```
SFT_OpenMathReasoning/
├── genselect/
│   ├── file1.json
│   ├── file2.json
│   └── ...
├── data/
│   └── genselect/
│       ├── file1.parquet
│       ├── file2.parquet
│       └── ...
└── README.md
```
