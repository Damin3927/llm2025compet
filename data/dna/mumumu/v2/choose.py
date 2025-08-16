import json
import pandas as pd
import torch
import logging
import sys
from sentence_transformers import SentenceTransformer
from pathlib import Path

def main():
    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # ファイルパス
    input_file = "../v1/vanilla_with_cot_vllm_cot_extracted.jsonl"
    output_file = "selected_cot_1200.jsonl"
    
    try:
        # データの読み込み
        logger.info(f"Loading data from {input_file}...")
        data = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples")
        
        # データの基本情報を表示
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Sample data keys: {data[0].keys() if data else 'No data'}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # MMR選択関数
    def mmr_with_query(embeddings: torch.Tensor, k: int, lam: float = 0.5) -> list[int]:
        """
        MMR (Maximal Marginal Relevance) アルゴリズムによるサンプル選択
        
        Args:
            embeddings: 正規化済み埋め込みベクトル (N, d)
            k: 選択するサンプル数
            lam: 多様性の重み (0.0-1.0)
        
        Returns:
            選択されたサンプルのインデックスリスト
        """
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

    try:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # テキストデータの準備（questionフィールドを使用）
        logger.info("Preparing text data...")
        texts = df["question"].tolist()
        if not texts:
            raise ValueError("No texts found to encode")

        logger.info("Encoding texts...")
        embs = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        logger.info(f"Encoded {len(texts)} texts to embeddings shape: {embs.shape}")

        # k が利用可能なサンプル数を超えていないかチェック
        max_k = len(texts)
        k = min(1200, max_k)
        if k < 1200:
            logger.warning(f"Requested k=1200 but only {max_k} samples available. Using k={k}")

        logger.info(f"Selecting {k} samples using MMR...")
        selected_idx = mmr_with_query(embs, k=k, lam=0.5)
        df_selected = df.iloc[selected_idx].reset_index(drop=True)
        logger.info(f"Selected {len(df_selected)} samples")

        # 選択されたデータを保存
        logger.info(f"Saving selected data to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for _, row in df_selected.iterrows():
                json.dump(row.to_dict(), f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"Successfully saved {len(df_selected)} selected samples to {output_file}")
        
        # 統計情報の表示
        logger.info("Selection statistics:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Selected samples: {len(df_selected)}")
        logger.info(f"  Selection ratio: {len(df_selected)/len(df)*100:.2f}%")

    except Exception as e:
        logger.error(f"Failed during text encoding and selection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
