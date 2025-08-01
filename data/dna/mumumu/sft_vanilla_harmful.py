import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import sys
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Load the WildJailbreak training set
    logger.info("Loading WildJailbreak training dataset...")
    train_ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    
    # Load the WildJailbreak evaluation set
    logger.info("Loading WildJailbreak evaluation dataset...")
    test_ds = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
    
    df_all = train_ds["train"].to_pandas()
    df = df_all[df_all["data_type"] == "vanilla_harmful"]
    logger.info(f"Loaded {len(df)} vanilla_harmful training samples")
    
    if len(df) == 0:
        raise ValueError("No vanilla_harmful samples found in training data")
    
    df_all_test = test_ds["train"].to_pandas()
    df_test = df_all_test[df_all_test["data_type"] == "vanilla_harmful"]
    logger.info(f"Loaded {len(df_test)} vanilla_harmful test samples")
    
except Exception as e:
    logger.error(f"Failed to load datasets: {e}")
    sys.exit(1)

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




try:
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    logger.info("Encoding texts...")
    texts = df["vanilla"].tolist()
    if not texts:
        raise ValueError("No texts found to encode")
    
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
    
    # completion列の存在確認
    if "completion" not in df_selected.columns:
        raise ValueError("'completion' column not found in selected data")
    
    df_selected["completion"] = "<think></think>" + df_selected["completion"].astype(str)
    
except Exception as e:
    logger.error(f"Failed during text encoding and selection: {e}")
    sys.exit(1) 


# ==========================================================
# Qwen3 32B を用いて <think> ... </think> に COT を挿入する処理
# ==========================================================

try:
    logger.info("Loading Qwen3-32B tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-32B",
        trust_remote_code=True,
    )
    tok.padding_side = "left"
    
    # pad_token がなければ eos_token を流用（左パディングのために必要）
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id  # 念のため明示
        logger.info("Set pad_token to eos_token")
        
    additional_special_tokens = []
    for tag in ["<think>", "</think>"]:
        if tok.convert_tokens_to_ids(tag) == tok.unk_token_id:
            additional_special_tokens.append(tag)
    if additional_special_tokens:
        tok.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        logger.info(f"Added special tokens: {additional_special_tokens}")
    
    logger.info("Loading Qwen3-32B model... (This may take several minutes)")
    # GPU メモリチェック
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Available GPU memory: {gpu_memory:.1f} GB")
        if gpu_memory < 60:
            logger.warning("GPU memory may be insufficient for Qwen3-32B. Consider using smaller model or CPU.")
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-32B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    
    # tokenizer に新トークンを追加した場合はembed数をリサイズ
    original_vocab_size = model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(len(tok))
    new_vocab_size = model.get_input_embeddings().weight.size(0)
    if original_vocab_size != new_vocab_size:
        logger.info(f"Resized token embeddings from {original_vocab_size} to {new_vocab_size}")
    
    logger.info("Model loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error("Consider using a smaller model or ensure sufficient GPU memory")
    sys.exit(1)

try:
    end_tag = "</think>"
    eos_id = tok.convert_tokens_to_ids(end_tag)
    if eos_id == tok.unk_token_id:
        logger.warning("</think> token not found in vocabulary, using eos_token_id instead")
        eos_id = tok.eos_token_id
    
    logger.info(f"Using eos_id: {eos_id} for {end_tag}")
except NameError:
    logger.error("Tokenizer not initialized. Cannot proceed with generation.")
    sys.exit(1)

def build_prompt(prompt, answer):
    """プロンプトを構築する関数"""
    if not prompt or not answer:
        raise ValueError("Both prompt and answer must be non-empty")
    
    return (
        "### User prompt:\n"
        f"{prompt}\n\n"
        "### Assistant answer (ground-truth):\n"
        f"{answer}\n\n"
        "Now write the step-by-step reasoning that leads to this answer.\n"
        "<think>"
    )

def gen_cot(batch_rows, max_new=256):
    """Chain-of-Thoughtを生成する関数"""
    try:
        if len(batch_rows) == 0:
            logger.warning("Empty batch provided to gen_cot")
            return []
        
        prompts = []
        for _, r in batch_rows.iterrows():
            try:
                prompt = build_prompt(r.vanilla, r.completion.replace("<think></think>", ""))
                prompts.append(prompt)
            except Exception as e:
                logger.error(f"Failed to build prompt for row: {e}")
                prompts.append("")  # 空のプロンプトを追加して継続
        
        # 空のプロンプトをフィルタリング
        valid_prompts = [p for p in prompts if p]
        if len(valid_prompts) == 0:
            logger.error("No valid prompts generated")
            return [""] * len(batch_rows)
        
        logger.debug(f"Generating CoT for {len(valid_prompts)} prompts")
        inputs = tok(valid_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outs = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=eos_id,
                    pad_token_id=tok.pad_token_id,
                )
            except torch.cuda.OutOfMemoryError:
                logger.error("GPU out of memory during generation. Try reducing batch size.")
                return [""] * len(batch_rows)
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                return [""] * len(batch_rows)

        texts = tok.batch_decode(outs, skip_special_tokens=False)
        cots = []
        for t in texts:
            try:
                if "<think>" in t and end_tag in t:
                    cot = t.split("<think>")[1].split(end_tag)[0].strip()
                else:
                    logger.warning("Generated text does not contain expected <think> tags")
                    cot = ""
                cots.append(cot)
            except Exception as e:
                logger.error(f"Error extracting CoT from generated text: {e}")
                cots.append("")
        
        return cots
        
    except Exception as e:
        logger.error(f"Unexpected error in gen_cot: {e}")
        return [""] * len(batch_rows)

try:
    batch_size = 8
    total_batches = (len(df_selected) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(df_selected)} samples in {total_batches} batches of size {batch_size}")
    
    processed_count = 0
    
    for i in range(0, len(df_selected), batch_size): 
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{total_batches}")
        
        try:
            rows = df_selected.iloc[i:i+batch_size]
            cots = gen_cot(rows)
            
            if len(cots) != len(rows):
                logger.error(f"Mismatch: got {len(cots)} CoTs for {len(rows)} rows")
                continue
            
            # 各行の completion を更新
            for j, (idx, row) in enumerate(rows.iterrows()):
                try:
                    original_completion = row.completion.replace('<think></think>', '')
                    new_completion = f"<think>{cots[j]}</think>{original_completion}"
                    df_selected.loc[idx, "completion"] = new_completion
                    processed_count += 1
                    
                    # 100件ごとにログ出力
                    if processed_count % 100 == 0:
                        logger.info(f"🎯 {processed_count}件のCoT保存しました！ (進捗: {processed_count}/{len(df_selected)})")
                        
                except Exception as e:
                    logger.error(f"Failed to update completion for row {idx}: {e}")
            
            logger.info(f"Completed batch {batch_num}/{total_batches} (総処理数: {processed_count})")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_num}: {e}")
            continue
    
    # 最終的な処理数をログ出力
    if processed_count % 100 != 0:  # 最後の100件未満の分もログ出力
        logger.info(f"🎯 最終: {processed_count}件のCoT保存しました！")
    
    # 結果の保存
    output_file = "vanilla_with_cot.jsonl"
    logger.info(f"Saving results to {output_file}")
    
    try:
        df_selected.to_json(output_file, orient="records", force_ascii=False, lines=True)
        logger.info(f"✅ Successfully saved {len(df_selected)} samples to {output_file}")
        
        # ファイルサイズとサンプル確認
        output_path = Path(output_file)
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            logger.info(f"Output file size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to save output file: {e}")
        # バックアップとして pickle で保存を試行
        try:
            backup_file = "vanilla_with_cot_backup.pkl"
            df_selected.to_pickle(backup_file)
            logger.info(f"Saved backup as {backup_file}")
        except Exception as backup_e:
            logger.error(f"Failed to save backup: {backup_e}")
    
except Exception as e:
    logger.error(f"Fatal error during processing: {e}")
    sys.exit(1)

logger.info("Script completed successfully")

