from datasets import load_dataset, Features, Value
import json

# === カラムの型をすべて明示的に文字列型にする ===
features = Features({
    "vanilla": Value("string"),
    "adversarial": Value("string"),
    "completion": Value("string"),
    "data_type": Value("string")
})

# === データセットを読み込む（trainのみ）===
train_ds = load_dataset(
    "allenai/wildjailbreak",
    name="train",
    split="train",
    features=features
)


'''
vanilla_harmful ストレートな有害な質問
vanilla_benign 一見有害だが、実は無害な質問
adversarial_harmful 功名な手口を使った有害質問
adversatial_benign 功名な手口に見えるが、実は無害な質問
'''

type = 'vanilla_benign'

# ===  vanilla_benignだけを抽出 ===
vnl_ds = train_ds.filter(lambda x: x["data_type"] == type)
# === "sorry" が含まれるものを除外 ===
# === 小文字化してSorryにも対応 ===
vnl_ds = vnl_ds.filter(
    lambda x: "sorry" not in x.get("completion", "").lower()
)


# === 出力ファイル ===
output_path = type+"_train.jsonl"


# === JSONL 形式で保存 ===
with open(output_path, "w", encoding="utf-8") as f:
    for idx, example in enumerate(vnl_ds):

        #直接的な質問はvanilla
        problem = example["vanilla"].strip()
        
        
        answer = example["completion"].strip()
        output = f"<think></think> {answer}"

        json.dump({
            "id": idx + 1,
            "problem": problem,
            "output": output,
            "answer": answer
        }, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ {len(vnl_ds)} 件のデータを書き出しました: {output_path}")