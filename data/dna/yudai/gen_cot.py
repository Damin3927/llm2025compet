#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_think_qwen3_both_from_answer.py
-------------------------------------------
- 単一JSON(dict or list) / JSONL の両対応
- preferred_output / non_preferred_output の両方に、解答テキストから導出したCoT(<think>…</think>)を埋める
- 既に非空の <think>…</think> がある場合は上書きしない（空のみ埋める or 無い場合は先頭付与）
- Qwen/Qwen3-32B（ベース）を vLLM で使用
-------------------------------------------
"""

import os, json, re, time, hashlib, traceback
from pathlib import Path
from typing import Dict, Optional, List, Any, Set

from vllm import LLM, SamplingParams

# =========================
# ハードコーディング設定（ここだけ編集）
# =========================
CONFIG = {
    "INPUT_PATH": "/home/Competition2025/P02/P02U011/test/inputs/out_00000.jsonl",
    "OUTPUT_PATH": "/home/Competition2025/P02/P02U011/test/outputs/train_part_00000_filled.jsonl",

    # vLLM / モデル
    "MODEL": "Qwen/Qwen3-32B",
    "TP": 8,
    "MAX_MODEL_LEN": 8192,
    "MAX_NUM_SEQS": 4,
    "GPU_UTIL": 0.90,

    # 生成パラメータ
    "MAX_TOKENS": 512,       # CoT用なので短めでOK
    "TEMPERATURE": 0.2,
    "TOP_P": 0.9,
    "SEED": 42,

    # どのフィールドを処理するか（順番厳守）
    "FIELDS": ["preferred_output", "non_preferred_output"],

    # リトライ設定
    "RETRY_MAX": 3,
    "RETRY_SLEEP": 2.0,  # 秒（指数バックオフ）
}

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# =========================
# ヘルパ
# =========================
THINK_OPEN = r"<think>"
THINK_CLOSE = r"</think>"

def has_nonempty_think(s: str) -> bool:
    if not isinstance(s, str):
        return False
    m = re.search(r"<think>(.*?)</think>", s, flags=re.DOTALL|re.IGNORECASE)
    return bool(m and m.group(1).strip())

def has_empty_think(s: str) -> bool:
    if not isinstance(s, str):
        return False
    m = re.search(r"<think>(.*?)</think>", s, flags=re.DOTALL|re.IGNORECASE)
    return bool(m and not m.group(1).strip())

def strip_all_think_blocks(s: str) -> str:
    """すべての <think>…</think> を除去して最終出力のみを返す。"""
    if not isinstance(s, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL|re.IGNORECASE).strip()

def insert_or_replace_think(answer: str, think_text: str) -> str:
    """
    - 既に非空の <think> がある: そのまま返す（上書きしない）
    - 空の <think></think> がある: 最初の空 think を埋める
    - think がない: 先頭に <think>…</think> を付与
    """
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)

    if has_nonempty_think(answer):
        return answer

    if has_empty_think(answer):
        # 最初の空thinkだけ置換
        def repl_once(m):
            inner = m.group(1)
            if inner.strip() == "":
                return f"<think>{think_text}</think>"
            return m.group(0)
        return re.sub(r"<think>(.*?)</think>", repl_once, answer, count=1, flags=re.DOTALL|re.IGNORECASE)

    # 無い場合は先頭に付与
    return f"<think>{think_text}</think>" + (answer if answer.startswith("\n") else ("\n" + answer if answer else ""))

def valid_reasoning(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip()
    if len(t) < 15:
        return False
    # タグ混入はNG（こちらで <think> を付加する）
    if re.search(r"</?think>|</?final>", t, flags=re.IGNORECASE):
        return False
    return True

def build_prompt_for_answer(answer_text: str) -> str:
    """
    ベースモデル向け（プレーンプロンプト）。
    - 回答文の内容に基づく “手順・根拠の短いステップ要約” を作らせる
    - 新しい事実を追加しないことを強調
    - タグ禁止
    """
    return (
        "Write ONLY the hidden step-by-step reasoning that led to the following answer.\n"
        "- Use ONLY information present in the answer; do NOT introduce new facts.\n"
        "- Focus on brief numbered steps or concise plan (setup -> settings -> test -> safety/validation).\n"
        "- Output ONLY the reasoning text (no XML/HTML tags, no final answer, no headings).\n\n"
        "Answer:\n"
        f"{answer_text}\n\n"
        "Reasoning:"
    )

def batched_generate(llm: LLM, sp: SamplingParams, prompts: List[str]) -> List[str]:
    outs = llm.generate(prompts, sampling_params=sp)
    texts = []
    for o in outs:
        t = o.outputs[0].text.strip() if o.outputs else ""
        texts.append(t)
    return texts

def gen_reasoning_with_retry(llm: LLM, sp: SamplingParams, answer_text: str,
                             retry_max: int, retry_sleep: float) -> str:
    prompt = build_prompt_for_answer(strip_all_think_blocks(answer_text))
    last = ""
    for i in range(retry_max):
        out = batched_generate(llm, sp, [prompt])[0]
        last = out
        if valid_reasoning(out):
            return out
        time.sleep(retry_sleep * (2 ** i))
    return last  # 最後の出力を返す（下流でvalidityチェック済み）

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def atomic_append(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_existing_ids(output_path: Path) -> Set[Any]:
    ids: Set[Any] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        ids.add(obj["id"])
                except Exception:
                    continue
    return ids

def derive_output_from_input(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_filled_qwen3-32B.jsonl")

# =========================
# 本体
# =========================
def main():
    in_path = Path(CONFIG["INPUT_PATH"])
    out_path = Path(CONFIG["OUTPUT_PATH"]) if CONFIG["OUTPUT_PATH"] else derive_output_from_input(in_path)

    print(f"[INFO] INPUT : {in_path}", flush=True)
    print(f"[INFO] OUTPUT: {out_path}", flush=True)

    # vLLM 初期化
    llm = LLM(
        model=CONFIG["MODEL"],
        trust_remote_code=True,
        tensor_parallel_size=CONFIG["TP"],
        gpu_memory_utilization=CONFIG["GPU_UTIL"],
        max_model_len=CONFIG["MAX_MODEL_LEN"],
        max_num_seqs=CONFIG["MAX_NUM_SEQS"],
    )
    sp = SamplingParams(
        temperature=CONFIG["TEMPERATURE"],
        top_p=CONFIG["TOP_P"],
        max_tokens=CONFIG["MAX_TOKENS"],
        # タグを出させない・思考だけ短めに切る
        stop=["</think>", "<think>", "<final>", "</final>"],
        seed=CONFIG["SEED"],
    )

    existing_ids = load_existing_ids(out_path)
    print(f"[INFO] already processed: {len(existing_ids)} (skip)", flush=True)

    is_jsonl = in_path.suffix.lower() == ".jsonl"
    n_in = n_ok = n_skip = n_fail = 0

    def handle_one(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        changed_any = False
        for key in CONFIG["FIELDS"]:
            if key not in rec or not isinstance(rec[key], str):
                continue

            text = rec[key]
            # すでに中身ありの think は触らない
            if has_nonempty_think(text):
                continue

            # CoT生成（回答から導出）
            reasoning = gen_reasoning_with_retry(
                llm, sp, text, CONFIG["RETRY_MAX"], CONFIG["RETRY_SLEEP"]
            )
            if not valid_reasoning(reasoning):
                # 埋め失敗（次のフィールドへ）
                continue

            rec[key] = insert_or_replace_think(text, reasoning)
            changed_any = True

        # 少なくとも preferred_output は非空thinkになっていることをOK条件に
        ok_pref = ("preferred_output" in rec) and has_nonempty_think(rec["preferred_output"])
        ok_nonp = True
        if "non_preferred_output" in CONFIG["FIELDS"]:
            ok_nonp = ("non_preferred_output" in rec) and has_nonempty_think(rec["non_preferred_output"])

        return rec if (ok_pref and ok_nonp) else (rec if changed_any else None)

    try:
        if is_jsonl:
            with in_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    n_in += 1
                    try:
                        rec = json.loads(line)
                        rec_id = rec.get("id", sha1(line))
                        if rec_id in existing_ids:
                            n_skip += 1
                            continue

                        out = handle_one(rec)
                        if out is None:
                            n_fail += 1
                            continue

                        atomic_append(out_path, json.dumps(out, ensure_ascii=False))
                        existing_ids.add(rec_id)
                        n_ok += 1
                        if n_ok % 10 == 0:
                            print(f"[PROGRESS] ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
                    except Exception:
                        traceback.print_exc()
                        n_fail += 1
        else:
            # 単一JSON（dict or list） → 1行ずつJSONLに追記
            with in_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, dict):
                n_in = 1
                rec = obj
                rec_id = rec.get("id", sha1(json.dumps(rec, ensure_ascii=False)))
                if rec_id in existing_ids:
                    n_skip += 1
                else:
                    try:
                        out = handle_one(rec)
                        if out is None:
                            n_fail += 1
                        else:
                            atomic_append(out_path, json.dumps(out, ensure_ascii=False))
                            existing_ids.add(rec_id)
                            n_ok += 1
                    except Exception:
                        traceback.print_exc()
                        n_fail += 1
            elif isinstance(obj, list):
                for rec in obj:
                    if not isinstance(rec, dict):
                        continue
                    n_in += 1
                    rec_id = rec.get("id", sha1(json.dumps(rec, ensure_ascii=False)))
                    if rec_id in existing_ids:
                        n_skip += 1
                        continue
                    try:
                        out = handle_one(rec)
                        if out is None:
                            n_fail += 1
                        else:
                            atomic_append(out_path, json.dumps(out, ensure_ascii=False))
                            existing_ids.add(rec_id)
                            n_ok += 1
                            if n_ok % 10 == 0:
                                print(f"[PROGRESS] ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
                    except Exception:
                        traceback.print_exc()
                        n_fail += 1
            else:
                print("Unsupported JSON root type (dict/list expected).", flush=True)
    finally:
        print(f"✅ Done: in={n_in} ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
        print(f"📝 Output: {out_path}", flush=True)

if __name__ == "__main__":
    main()
