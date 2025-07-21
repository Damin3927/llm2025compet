# 評価 Evaluation

## 環境構築

評価用の環境構築は不要とする方針。

プロジェクトルートに .env を作成し、以下を記入：
```dotenv
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 実行方法

### Humanity's Last Exam (HLE)


### Do-Not-Answer (DNA)

評価するためのテストデータ生成（一時的に使用）
```bash
python3 evaluation/dna/inference.py
```

DNA評価スクリプトの実行
```bash
python3 evaluation.py \
  --dataset do_not_answer_response \
  --judge gpt-4.1-mini-2025-04-14
```

フォルダ構成
```plaintext
dna/
├── inference.py # DNA評価スクリプトで評価するためのテストデータ生成（一時的に使用）
├── eval.py      # DNA評価スクリプト
└── datasets/    # Huggin Faceでデータセットをやり取りするまで一時的に使用
```

