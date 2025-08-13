# classfiy_scripts_sft
Hugging Face のデータセットを読み込み、question だけで有害性カテゴリを三段階（Risk / Type / Specific）に階層分類。  
その場でコンソールに統計レポートを出しつつ、PNG グラフも保存します。  
モデルは OpenRouter 経由（デフォルト: meta-llama/llama-3-8b-instruct）  
出力ミスを減らすため、選択肢を番号化し、半角数字のみで回答させます  
失敗時は OTHER に安全フォールバック  
dna_hierarchy.json は同ディレクトリに置く前提  

1 ) リポジトリ構造

    .
    └── stats/
        ├── classfiy_scripts_sft 
        ├── dna_hierarchy.json 
        ├── requirements.txt
        ├── .env
        └── figs/

2 ) 依存関係のインストール

```bash:requirements.txt
pip install -r requirements.txt
```

3 ) OpenRouter API キー

あらかじめtokenを取得して.envファイルに設定
```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
```

### 使い方

HuggingFaceに先にログインしてください！ 

```
huggingface-cli login
```

```
# dna_hierarchy.json はスクリプトと同じディレクトリにある前提
python classfiy_scripts_sft.py 
  --hf_repo_id neko-llm/wj-Adversarial_harmful
  --hf_split train 
  --hierarchy_file ./dna_hierarchy.json 
  --start_index 0 --end_index 1145
  --outdir ./figs 
  --shorten_specific 
  --output_file ./classified_wj_ah_train.head.jsonl
  --report_txt ./reports/wj_ah_head_stats.txt
```


|オプション	|説明| 既定|
|:-----------|------------:|:------------:|
|--hf_repo_id | HF の repo id（例: neko-llm/wj-Adversarial_harmful） |	必須|
|--hf_split	| train / validation / test etc. | train|
|--hf_name	config |名があるデータセット用 | なし|
|--hf_field_id	|ID フィールド名 | id|
|--hf_field_question | question フィールド名 | question|
|--start_index / --end_index | 処理範囲（半開区間） 最後のid+1にしておくと良い	| 0 / 1e12|
|--model_name |OpenRouter のモデル名 | meta-llama/llama-3-8b-instruct|
|--api_key|	直渡し用 API キー | .env 参照|
|--output_file	|JSONL にも書き出す	 |なし|
|--outdir | 図の保存先 | figs|
|--top_n_specific | Specific Harm 上位 N を図示	|10|
|--shorten_specific	| Specific のラベルを「コロン前」だけに短縮|オフ|
|--output_file	| Specific のラベルを「コロン前」だけに短縮|	任意.jsonl|
|--report_txt	| Specific のラベルを「コロン前」だけに短縮|	任意.txt|




コンソール

```:bash
=== 集計結果 ===
総件数: 100
参考: エラー/スキップ含む行(概算): 3 (3.00%)

--- Risk Area 分布 ---
Misinformation Harms: 28 (28.00%)
Malicious Uses: 17 (17.00%)
...

--- Type of Harm 分布 ---
Disseminating false or misleading information: 11 (11.00%)
Assisting illegal activities: 5 (5.00%)
...

--- Specific Harm 上位10 ---
Misinterpretation or Wrong Context: 7 (7.00%)
Illegal Digital Activities: 5 (5.00%)
...

```


上記のような結果が出ますので、notionに記載をお願いします。