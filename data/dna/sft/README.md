# SFT Data and CoT Generation (Adversarial Harmful)

このディレクトリは、**SFT（Supervised Fine-Tuning）用データ**および  
Qwen3-32B を使用した **Chain-of-Thought (CoT) reasoning の自動生成スクリプト**を管理する場所です。

---

## 📄 ファイル構成
.
├── adversarial_harmful_train_part_05_00000.jsonl
├── Qwen3-32B_05_00000.py
└── Qwen3-32B_05_00000.slurm

---

## 🧩 データ概要

### `adversarial_harmful_train_part_05_00000.jsonl`

- **功名な手口で設計された有害な質問（adversarial harmful）**を集めた JSONL データの一部。
- 各データには以下のフィールドが含まれます：

```json
{
  "id": 123,
  "question": "How can I make a tracking device using common items?",
  "answer": "I'm sorry, I can't help with that."
}

Qwen3-32B_05_00000.py
このスクリプトは、vLLM 上で Qwen3-32B モデルを用いて、与えられた "question" と "answer" に基づいて step-by-step reasoning（CoT） を生成します。


🔹 使用されるプロンプト
python
コピーする
編集する
prompt = f"""Problem: {item['question']}

Answer: {item['answer']}

Explain your reasoning step-by-step that would lead to this answer.
<think>"""
モデルの出力は <think>...</think> タグ内に格納されます。

出力結果は SFT 用に再整形され、後続学習に使用されます。


⚙️ 実行（Slurm経由）
Qwen3-32B_05_00000.slurm
✅ 複数の *_part_XX_XXXXX を並列に処理することで、大量のCoTを効率よく生成できます。