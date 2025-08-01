import numpy as np
import pandas as pd
from collections import Counter
from datasets import get_dataset_config_names, get_dataset_split_names,load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the WildJailbreak training set
train_ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

# Load the WildJailbreak evaluation set
test_ds = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)

df_all = train_ds["train"].to_pandas()
df = df_all[df_all["data_type"] == "vanilla_harmful"]
df_all_test = test_ds["train"].to_pandas()
df_test = df_all_test[df_all_test["data_type"] == "vanilla_harmful"]



import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def mmr_with_query(embeddings: torch.Tensor, k: int, lam: float = 0.5) -> list[int]:
    # embeddings: (N, d), 正規化済みを前提（cosine を内積で使う）
    N = embeddings.size(0)
    device = embeddings.device

    # クエリを全体平均として作る（正規化）
    query_vec = embeddings.mean(dim=0, keepdim=True)
    query_vec = torch.nn.functional.normalize(query_vec, dim=1)  # (1, d)

    # 代表性スコア：各点と query のコサイン類似度
    rep_score = torch.matmul(embeddings, query_vec.T).squeeze(1)  # (N,)

    selected = []
    # 初回は代表性最大を選ぶ
    first = int(rep_score.argmax().item())
    selected.append(first)

    # 各候補について既選との最大類似度（冗長性）を保持
    # 初期化：一度選ばれたものとの similarity
    redundancy = torch.zeros(N, device=device)  # max similarity to S
    redundancy = torch.maximum(
        redundancy,
        torch.matmul(embeddings, embeddings[first:first+1].T).squeeze(1)
    )
    mask = torch.ones(N, dtype=torch.bool, device=device)
    mask[first] = False

    for _ in range(k - 1):
        # MMR スコア計算（masked）
        scores = (1 - lam) * rep_score - lam * redundancy  # (N,)
        scores = scores.masked_fill(~mask, float("-inf"))
        next_idx = int(scores.argmax().item())
        selected.append(next_idx)

        # 冗長性アップデート：既選との最大類似度を保持
        sim_to_new = torch.matmul(embeddings, embeddings[next_idx: next_idx+1].T).squeeze(1)
        redundancy = torch.maximum(redundancy, sim_to_new)

        mask[next_idx] = False

    return selected




model = SentenceTransformer("all-MiniLM-L6-v2")

texts = df["vanilla"].tolist()
embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
selected_idx = mmr_with_query(embs, k=1200, lam=0.5)
df_selected = df.iloc[selected_idx].reset_index(drop=True)

df_selected["completion"] =  "<think></think>" + df_selected["completion"].astype(str) 


# ==========================================================
# Qwen3 32B を用いて <think> ... </think> に COT を挿入する処理
# ==========================================================

tok = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-32B",
    trust_remote_code=True,
)

# --- add <think> and </think> if missing ---
special_tokens = {"additional_special_tokens": []}
for tag in ["<think>", "</think>"]:
    if tok.convert_tokens_to_ids(tag) == tok.unk_token_id:
        special_tokens["additional_special_tokens"].append(tag)
if special_tokens["additional_special_tokens"]:
    tok.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-32B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
).eval()

# tokenizer に新トークンを追加した場合はembed数をリサイズ
model.resize_token_embeddings(len(tok))

end_tag = "</think>"
eos_id = tok.convert_tokens_to_ids(end_tag)

def build_prompt(prompt, answer):
    return (
        "### User prompt:\n"
        f"{prompt}\n\n"
        "### Assistant answer (ground-truth):\n"
        f"{answer}\n\n"
        "Now write the step-by-step reasoning that leads to this answer.\n"
        "<think>"
    )

def gen_cot(batch_rows, max_new=256):
    prompts = [build_prompt(r.vanilla, r.completion.replace("<think></think>", "")) 
               for _, r in batch_rows.iterrows()]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=eos_id,
        )

    texts = tok.batch_decode(outs, skip_special_tokens=True)
    cots  = [t.split("<think>")[1].split(end_tag)[0].strip() for t in texts]
    return cots

batch = 8
for i in range(0, len(df_selected), batch): 
    rows = df_selected.iloc[i:i+batch]
    cots = gen_cot(rows)
    df_selected.loc[rows.index, "completion"] = [
        f"<think>{c}</think>{rows.iloc[j].completion.replace('<think></think>', '')}"
        for j, c in enumerate(cots)
    ]

df_selected.to_json("vanilla_with_cot.jsonl", orient="records", force_ascii=False)
print("✅ saved vanilla_with_cot.jsonl")

