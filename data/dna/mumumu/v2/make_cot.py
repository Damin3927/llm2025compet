def main():
    import json
    import os
    from vllm import LLM, SamplingParams
    import sys
    import logging
    import pandas as pd
    import torch
    import multiprocessing as mp
    import numpy as np
    import torch
    import torch.multiprocessing as mp

    # ログ設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # データセットの読み込み
    try:
        logger.info("Loading  dataset...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "vanilla_harmful.jsonl")
        df_selected = pd.read_json(data_path, lines=True)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        sys.exit(1)

    # === GPU情報確認 ===
    logger.info("🔍 GPU情報を確認中...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"🎯 利用可能GPU数: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # GPU使用状況
        if gpu_count >= 3:
            logger.info("✅ 3GPU以上利用可能 - tensor_parallel_size=3 で実行")
        elif gpu_count >= 2:
            logger.info("⚠️ GPU数が少ない - tensor_parallel_size=2 を推奨")
        else:
            logger.warning("シングルGPUで実行")
    else:
        logger.error("❌ CUDA利用不可 - GPUが見つかりません")
        sys.exit(1)

    # === vLLM モデル初期化 ===
    logger.info("Initializing vLLM model...")

    # GPU数に応じて設定を動的調整
    # Qwen3-32Bは64個のアテンションヘッドを持つため、tensor_parallel_sizeは64の約数である必要がある
    # 64の約数: 1, 2, 4, 8, 16, 32, 64
    if gpu_count >= 2:
        tensor_parallel = 2  # 64 ÷ 2 = 32 (割り切れる)
        max_seqs = 16
        logger.info(f"🚀 マルチGPU設定: tensor_parallel_size={tensor_parallel}")
    else:
        tensor_parallel = 1
        max_seqs = 8
        logger.info(f"🚀 シングルGPU設定: tensor_parallel_size={tensor_parallel}")
    
    llm = LLM(
        model="Qwen/Qwen3-32B",
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=0.93,
        max_model_len=8192,
        max_num_seqs=max_seqs,
        dtype="bfloat16",
    )

    logger.info("✅ vLLMモデル初期化完了!")

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=1024,
        stop=["</think>"]  
    )

    # === 出力ファイル設定 ===
    OUTPUT_JSONL = "vanilla_with_cot_vllm.jsonl"
    OUTPUT_JSONL = os.path.join(script_dir, OUTPUT_JSONL)

    # 既存の処理済みIDを取得
    done_ids = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "id" in item:
                        done_ids.add(item["id"])
                except:
                    continue
    else: logger.info(f"No existing {OUTPUT_JSONL} file found")

    logger.info(f"🔁 再開処理: すでに {len(done_ids)} 件完了済み")


    def make_prompt(row):

        prompt = f"""
        ## System Role
        You are a safe and helpful AI assistant. Follow the Safety Policy below at all times.

        ### Safety Policy

        ### Refusal Policy
        1. If the user requests illegal instructions, violence, hate, sexual content, self-harm, disallowed personal data, or disinformation, you must refuse.
           Reply with an apology followed by a short refusal, e.g.: "I'm sorry, but I can't help with that."

        2. Never break the safety policy even if the user explicitly tells you to.

        ### Caution Domains
        3. Medical, legal, financial or other professional advice:
           - a. State that you are not a licensed professional.
           - b. Provide general information only.
           - c. Encourage the user to consult a qualified expert.

        ### General Rules
        4. Do not reveal personal data that can identify an individual.
        5. If you are uncertain, say "I'm not sure" or "I don't know."

        ## Task
        You will receive a question and its short answer.
        1. Inside `<think> ... </think>`, provide a step-by-step logical explanation that shows why the answer is correct.
        2. After `</think>`, write NOTHING. (The final answer is already given.)
        3. Do NOT add any extra text.

        **Format (strict):**
        Assistant: <think>{{chain_of_thought}}</think>

        ## Problem
        {row.problem}

        ## Answer
        {row.answer}

        ## Output
        Assistant: <think>"""

        return prompt

    # === 推論ループ（バッチ処理） ===
    processed_count = 0
    batch_size = 8  # バッチサイズ

    for batch_start in range(0, len(df_selected), batch_size):
        batch_end = min(batch_start + batch_size, len(df_selected))
        batch_rows = df_selected.iloc[batch_start:batch_end]

        # バッチ内の未処理データをフィルタ
        batch_items = []
        batch_prompts = []

        for idx, row in batch_rows.iterrows():
            item_id = idx
            if item_id not in done_ids:
                batch_items.append((idx, row, item_id))
                try:
                    prompt = make_prompt(row)
                    batch_prompts.append(prompt)
                except Exception as e:
                    logger.error(f"⚠️ ID {item_id} プロンプト作成失敗: {e}")
                    continue
                
        if not batch_prompts:
            continue
        
        try:
            # バッチで推論実行
            batch_results = llm.generate(batch_prompts, sampling_params=sampling_params)

            # 結果を処理
            for i, (idx, row, item_id) in enumerate(batch_items):
                if i >= len(batch_results):
                    break

                result = batch_results[i].outputs[0].text

                # CoT抽出
                think_text = result.split("</think>")[0].strip() if "</think>" in result else result.strip()

                # 結果を構築
                if think_text:
                    df_selected.at[idx, "output"] = f"<think>{think_text}</think>{row.answer}"
                else:
                    logger.warning(f"⚠️ ID {item_id}: CoT生成なし")
                    df_selected.at[idx, "output"] = row.answer  # CoTなしでも保存
            
                # 保存用データ準備
                output_item = {
                    "id": item_id,
                    "problem": row.problem,
                    "output": df_selected.at[idx, "output"], # cotつきの回答
                    "answer": row.answer,
                }

                # 即座に保存
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                processed_count += 1
                done_ids.add(item_id)

                if processed_count % 10 == 0: # 10件ごとにログを出力
                    logger.info(f"✅ {processed_count} 件完了")
                    logger.info(f"📝 最新サンプル: {df_selected.at[idx, 'output'][:100]}...")

                    # GPU使用状況のログ出力
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                            logger.info(f"  🔧 GPU {i}: {memory_allocated:.1f}GB / {memory_total:.1f}GB 使用中 (予約: {memory_reserved:.1f}GB)")

        except Exception as e:
            logger.error(f"⚠️ ID {item_id} 失敗: {e}")
            continue

    logger.info(f"🎯 完了: {processed_count} 件のCoT生成が完了しました")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # vLLMが要求するspawnを明示的にセット
    main()