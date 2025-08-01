import numpy as np
import pandas as pd
from collections import Counter
from datasets import get_dataset_config_names, get_dataset_split_names,load_dataset
from sentence_transformers import SentenceTransformer
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


