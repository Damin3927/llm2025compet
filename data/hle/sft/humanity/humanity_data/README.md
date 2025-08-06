---
configs:
- config_name: law
  data_files: "data/law_law.parquet"
- config_name: history
  data_files: "data/history_history_reasoning.parquet"
---

# HLE_SFT_humanity

## データセットの説明

このデータセットは、以下の分割（split）ごとに整理された処理済みデータを含みます。

- **law**: 1 JSON files, 1 Parquet files
- **history**: 1 JSON files, 1 Parquet files

## データセット構成

各 split は JSON 形式と Parquet 形式の両方で利用可能です:
- **JSONファイル**: 各 split 用サブフォルダ内の元データ（`history/`）
- **Parquetファイル**: split名をプレフィックスとした最適化データ（`data/history_*.parquet`）

各 JSON ファイルには、同名の split プレフィックス付き Parquet ファイルが対応しており、大規模データセットの効率的な処理が可能です。

## 使い方

```python
from datasets import load_dataset

# 特定の split を読み込む
law_data = load_dataset("neko-llm/HLE_SFT_humanity", "law")
history_data = load_dataset("neko-llm/HLE_SFT_humanity", "history")

# または、data_files を手動で指定して読み込む
dataset = load_dataset(
    "parquet",
    data_files={
        "law": "data/law_*.parquet",
        "history": "data/history_*.parquet",
    }
)

# 個別のファイルを読み込む
import pandas as pd
df = pd.read_parquet("data/law_filename.parquet")

# Load all files for a specific split
from pathlib import Path
split_files = list(Path("data").glob("law_*.parquet"))
for file in split_files:
    df = pd.read_parquet(file)
    # Process df...
```

## ファイル構成

```
HLE_SFT_humanity/
├── law/
│   ├── file1.json
│   ├── file2.json
│   └── ...
├── data/
│   └── law/
│       ├── file1.parquet
│       ├── file2.parquet
│       └── ...
└── README.md
```
