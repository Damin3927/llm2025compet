import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
# Ensure project root is on sys.path when run as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset

from scripts.utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    build_messages,
    call_openrouter,
    categorize_prompt,
    detect_language,
    ensure_dirs,
    init_env,
    load_style_config,
    quality_check,
    sanitize_text,
    sha256_hex16,
    summarize_for_context,
)


SYSTEM_BASE = (
    "You are a highly knowledgeable AI assistant. Prioritize safety and provide clear, concise, and practical answers. Always respond in English."
)


def build_user_prompt(prompt_text: str, category: str, cfg: Dict[str, Any]) -> str:
    # Force English templates regardless of input language
    lang = "en"
    templates = cfg.get("templates", {})
    disclaimers = templates.get("disclaimers", {})
    refusals = templates.get("refusals", {})

    if category == "MEDICAL_ADVICE":
        pre = disclaimers.get("medical", {}).get(lang)
    elif category == "LEGAL_ADVICE":
        pre = disclaimers.get("legal", {}).get(lang)
    elif category == "FINANCIAL_ADVICE":
        pre = disclaimers.get("financial", {}).get(lang)
    elif category == "PRIVACY_VIOLATION":
        pre = refusals.get("privacy", {}).get(lang)
    else:
        pre = None

    header = (pre + "\n\n") if pre else ""
    return header + prompt_text


def generate_ideal_response(prompt_text: str, category: str, cfg: Dict[str, Any], api_key: Optional[str] = None) -> str:
    # Force English output and checks
    lang = "en"

    # Context handling / summarization if needed
    max_chars = 8000  # conservative guard regardless of tokens
    prompt_for_model = summarize_for_context(prompt_text, max_chars=max_chars)

    user_prompt = build_user_prompt(prompt_for_model, category, cfg)

    messages = build_messages(system_prompt=SYSTEM_BASE, user_prompt=user_prompt)
    text = call_openrouter(
        messages=messages,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_tokens=DEFAULT_MAX_TOKENS,
        seed=DEFAULT_SEED,
        api_key=api_key,
    )

    # Apply PII scrubbing to preferred_output only per spec
    text = sanitize_text(text)

    # Guardrail checks; attempt one repair if needed
    qr = quality_check(text, category, cfg, lang)
    if not qr.passed:
        # Add explicit instruction for repair
        repair_hint = (
            "\n\nPlease repair the response to satisfy required phrases and remove forbidden phrases."
        )
        messages = build_messages(system_prompt=SYSTEM_BASE, user_prompt=user_prompt + repair_hint)
        text = call_openrouter(
            messages=messages,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS,
            seed=DEFAULT_SEED,
            api_key=api_key,
        )
        text = sanitize_text(text)

    return text


def main() -> None:
    init_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default="/home/argo/llm2025compet/data/processed/dna_dpo_v2.jsonl")
    parser.add_argument("--style", type=str, default="/home/argo/llm2025compet/config/style_templates.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default="/home/argo/llm2025compet/logs/create_dataset_v2.ckpt")
    parser.add_argument("--errlog", type=str, default="/home/argo/llm2025compet/logs/dataset_v2_errors.jsonl")
    parser.add_argument("--progress", type=str, default="/home/argo/llm2025compet/logs/dataset_v2_progress.jsonl")
    parser.add_argument("--safety_csv", type=str, default="/home/argo/llm2025compet/data/cards/safety_review.csv")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs([
        "/home/argo/llm2025compet/data/processed",
        "/home/argo/llm2025compet/logs",
        "/home/argo/llm2025compet/reports",
    ])

    cfg = load_style_config(args.style)

    ds = load_dataset("neko-llm/dna_dpo_hh-rlhf", split=args.split)

    # Detect field names from dataset metadata
    fields = set(getattr(ds, "column_names", []))
    # Heuristic mapping for question and preferred_output
    question_key = "question" if "question" in fields else (
        "prompt" if "prompt" in fields else "input"
    )
    preferred_key = "preferred_output" if "preferred_output" in fields else (
        "chosen" if "chosen" in fields else None
    )

    if question_key is None or preferred_key is None:
        raise RuntimeError(f"Required fields not found. fields={fields}")

    # Resume support: load processed ids
    processed_ids: set[str] = set()
    if args.resume and os.path.exists(args.ckpt):
        with open(args.ckpt, "r", encoding="utf-8") as f:
            for line in f:
                pid = line.strip()
                if pid:
                    processed_ids.add(pid)

    ensure_dirs([
        os.path.dirname(args.out),
        os.path.dirname(args.errlog),
        os.path.dirname(args.progress),
        os.path.dirname(args.safety_csv),
    ])

    # Prepare writers with thread locks
    out_lock = threading.Lock()
    err_lock = threading.Lock()
    ckpt_lock = threading.Lock()
    safety_lock = threading.Lock()
    prog_lock = threading.Lock()

    # Ensure safety CSV has header
    if not os.path.exists(args.safety_csv):
        ensure_dirs([os.path.dirname(args.safety_csv)])
        with open(args.safety_csv, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["id", "category", "question", "timestamp"])

    # Open output and logs
    out_f = open(args.out, "a", encoding="utf-8")
    err_f = open(args.errlog, "a", encoding="utf-8")
    ckpt_f = open(args.ckpt, "a", encoding="utf-8")
    prog_f = open(args.progress, "a", encoding="utf-8")

    def write_out(line: str) -> None:
        with out_lock:
            out_f.write(line + "\n")
            out_f.flush()

    def write_err(obj: Dict[str, Any]) -> None:
        with err_lock:
            err_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            err_f.flush()

    def write_ckpt(ex_id: str) -> None:
        with ckpt_lock:
            ckpt_f.write(ex_id + "\n")
            ckpt_f.flush()

    def write_prog(obj: Dict[str, Any]) -> None:
        with prog_lock:
            prog_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            prog_f.flush()

    def append_safety(id_: str, category: str, question: str) -> None:
        with safety_lock:
            with open(args.safety_csv, "a", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow([id_, category, question, datetime.utcnow().isoformat() + "Z"])

    # Processing function
    def process_example(idx: int, ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        try:
            question = ex.get(question_key, "")
            old_pref = ex.get(preferred_key, "")
            ex_id = ex.get("id") or sha256_hex16(question)

            if args.resume and ex_id in processed_ids:
                return ex_id, None, "skipped"

            category = categorize_prompt(question)

            if category == "OTHER":
                lang = "en"
                qr = quality_check(old_pref, category, cfg, lang)
                if qr.passed and old_pref:
                    new_pref = sanitize_text(old_pref)
                else:
                    new_pref = generate_ideal_response(question, category, cfg, api_key=args.api_key)
            else:
                new_pref = generate_ideal_response(question, category, cfg, api_key=args.api_key)

            out_ex: Dict[str, Any] = {}
            out_ex.update(ex)
            out_ex["question"] = question
            out_ex["preferred_output"] = new_pref
            out_ex["non_preferred_output"] = old_pref
            out_ex["category"] = category
            out_ex["id"] = ex_id
            out_ex["_meta"] = {
                "model": DEFAULT_MODEL,
                "decoding": {
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "seed": DEFAULT_SEED,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            # Safety review logging
            if category == "PRIVACY_VIOLATION":
                append_safety(ex_id, category, question)

            return ex_id, out_ex, None
        except Exception as e:  # noqa: BLE001
            return None, None, f"{type(e).__name__}: {e}"

    written = 0
    futures: List[cf.Future[Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        submitted = 0
        for i, ex in enumerate(ds):
            if args.limit is not None and submitted >= args.limit:
                break
            futures.append(executor.submit(process_example, i, ex))
            submitted += 1

        for fut in cf.as_completed(futures):
            ex_id, out_obj, err = fut.result()
            if err is not None:
                write_err({
                    "id": ex_id,
                    "error": err,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
                continue

            if out_obj is None or ex_id is None:
                # skipped or invalid
                continue

            write_out(json.dumps(out_obj, ensure_ascii=False))
            write_ckpt(ex_id)
            written += 1
            write_prog({"written": written, "last_id": ex_id, "ts": datetime.utcnow().isoformat() + "Z"})

    out_f.close()
    err_f.close()
    ckpt_f.close()
    prog_f.close()

    print(f"Wrote {written} examples to {args.out}")


if __name__ == "__main__":
    main()


