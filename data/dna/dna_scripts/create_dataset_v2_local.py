import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Local project paths only under data/dna (dynamic, repo位置に依存しない)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE)

from datasets import load_dataset  # type: ignore

from dna_scripts.utils import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    call_openrouter,
    ensure_dirs,
    init_env,
    quality_check,
    sanitize_text,
    sha256_hex16,
)


SYSTEM_BASE = (
    "You are a highly knowledgeable AI assistant. Prioritize safety and provide clear, concise, and practical answers. Always respond in English."
)


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def generate_ideal_response(prompt_text: str, category: str, cfg: Dict[str, Any], api_key: Optional[str] = None) -> str:
    # English-only path
    # prepend disclaimer/refusal
    pre_map = {
        "MEDICAL_ADVICE": "I am not a medical professional. This is general information and not medical advice. Please consult a qualified professional for specific guidance.",
        "LEGAL_ADVICE": "I am not a lawyer. This is general information and not legal advice. Please consult a qualified attorney for specific guidance.",
        "FINANCIAL_ADVICE": "I am not a financial advisor. This is general information and not investment advice. Please consult a qualified professional for important decisions.",
        "PRIVACY_VIOLATION": "I can’t help with that request as it involves private or personal information. Protecting users’ privacy and confidentiality is a top priority.",
    }
    header = pre_map.get(category)
    user_prompt = (header + "\n\n" if header else "") + prompt_text

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
    # Preserve reasoning by default for the reasoning model dataset
    text = sanitize_text(text, preserve_think=True, scrub_cot=True)

    # Guardrails
    req_map = {
        "MEDICAL_ADVICE": ["I am not a medical professional", "general information"],
        "LEGAL_ADVICE": ["I am not a lawyer", "general information"],
        "FINANCIAL_ADVICE": ["I am not a financial advisor", "general information"],
        "PRIVACY_VIOLATION": ["can’t help", "privacy"],
    }
    min_len = int(cfg.get("min_len", 60))
    if len(text) < min_len or any(p not in text for p in req_map.get(category, [])):
        # one repair attempt
        repair = user_prompt + "\n\nPlease repair the response to satisfy required phrases (English) and remove forbidden phrases."
        messages = build_messages(system_prompt=SYSTEM_BASE, user_prompt=repair)
        text = call_openrouter(
            messages=messages,
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            max_tokens=DEFAULT_MAX_TOKENS,
            seed=DEFAULT_SEED,
            api_key=api_key,
        )
        text = sanitize_text(text, preserve_think=True, scrub_cot=True)

    return text


def main() -> None:
    init_env(os.path.join(BASE, ".env"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out", type=str, default=os.path.join(BASE, "processed/dna_dpo_v2.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=os.path.join(BASE, "logs/create_dataset_v2.ckpt"))
    parser.add_argument("--errlog", type=str, default=os.path.join(BASE, "logs/dataset_v2_errors.jsonl"))
    parser.add_argument("--progress", type=str, default=os.path.join(BASE, "logs/dataset_v2_progress.jsonl"))
    parser.add_argument("--safety_csv", type=str, default=os.path.join(BASE, "cards/safety_review.csv"))
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    ensure_dirs([
        os.path.dirname(args.out),
        os.path.dirname(args.errlog),
        os.path.dirname(args.progress),
        os.path.dirname(args.safety_csv),
    ])

    ds = load_dataset("neko-llm/dna_dpo_hh-rlhf", split=args.split)
    fields = set(getattr(ds, "column_names", []))
    question_key = "question" if "question" in fields else ("prompt" if "prompt" in fields else "input")
    preferred_key = "preferred_output" if "preferred_output" in fields else ("chosen" if "chosen" in fields else None)
    if preferred_key is None:
        raise RuntimeError("preferred_output/ chosen field not found in dataset")

    # simple keyword categorizer (English-only input expected)
    def categorize(q: str, ex: Dict[str, Any]) -> str:
        s=q.lower()
        if any(k in s for k in ["fever","dosage","diagnose","medication","symptom","treatment","cancer","psa","ekg","antibiotics"]): return "MEDICAL_ADVICE"
        if any(k in s for k in ["legal","contract","sue","lease","llc","non-compete","copyright","fair use","evict","small claims"]): return "LEGAL_ADVICE"
        if any(k in s for k in ["invest","stock","crypto","options","tax","ira","day trading","loan","pre-ipo"]): return "FINANCIAL_ADVICE"
        if any(k in s for k in ["address","phone","email","social media","ip address","ssn","gps","account name"]): return "PRIVACY_VIOLATION"
        # escalate to refusal for specific risk areas/harm
        risk = (ex.get("risk_area") or "").lower() + " " + (ex.get("type_of_harm") or "").lower()
        if any(w in risk for w in ["privacy", "personal", "pii", "information hazard"]):
            return "PRIVACY_VIOLATION"
        return "OTHER"

    # resume
    processed_ids: set[str] = set()
    if args.resume and os.path.exists(args.ckpt):
        with open(args.ckpt, "r", encoding="utf-8") as f:
            processed_ids = {line.strip() for line in f if line.strip()}

    out_f = open(args.out, "a", encoding="utf-8")
    err_f = open(args.errlog, "a", encoding="utf-8")
    ckpt_f = open(args.ckpt, "a", encoding="utf-8")
    prog_f = open(args.progress, "a", encoding="utf-8")
    out_lock = threading.Lock(); err_lock = threading.Lock(); ckpt_lock = threading.Lock(); prog_lock = threading.Lock()

    def write_out(obj: Dict[str, Any]) -> None:
        with out_lock:
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
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
        os.makedirs(os.path.join(BASE, "cards"), exist_ok=True)
        with open(args.safety_csv, "a", encoding="utf-8", newline="") as fcsv:
            writer = csv.writer(fcsv)
            if fcsv.tell() == 0:
                writer.writerow(["id", "category", "question", "timestamp"])
            writer.writerow([id_, category, question, datetime.utcnow().isoformat() + "Z"])

    def process_example(i: int, ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        try:
            question = ex.get(question_key, "")
            old_pref = ex.get(preferred_key, "")
            raw_id = ex.get("id")
            ex_id = str(raw_id) if raw_id not in (None, "") else sha256_hex16(question)
            if args.resume and ex_id in processed_ids:
                return ex_id, None, "skipped"

            category = categorize(question, ex)

            # Always regenerate for safety consistency (no reuse for OTHER)
            new_pref = generate_ideal_response(question, category, {}, api_key=args.api_key)

            out_ex: Dict[str, Any] = dict(ex)
            out_ex["question"] = question
            out_ex["preferred_output"] = new_pref
            out_ex["non_preferred_output"] = old_pref
            out_ex["category"] = category
            out_ex["id"] = ex_id
            out_ex["_meta"] = {
                "model": DEFAULT_MODEL,
                "decoding": {"temperature": DEFAULT_TEMPERATURE, "top_p": DEFAULT_TOP_P, "max_tokens": DEFAULT_MAX_TOKENS, "seed": DEFAULT_SEED},
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }

            if category == "PRIVACY_VIOLATION":
                append_safety(ex_id, category, question)
            return ex_id, out_ex, None
        except Exception as e:
            return None, None, f"{type(e).__name__}: {e}"

    written = 0
    futures: List[cf.Future[Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as exs:
        submitted = 0
        for i, ex in enumerate(ds):
            if args.limit is not None and submitted >= args.limit:
                break
            futures.append(exs.submit(process_example, i, ex))
            submitted += 1
        for fut in cf.as_completed(futures):
            ex_id, out_obj, err = fut.result()
            if err is not None:
                write_err({"id": ex_id, "error": err, "ts": datetime.utcnow().isoformat() + "Z"})
                continue
            if out_obj is None or ex_id is None:
                continue
            write_out(out_obj)
            write_ckpt(ex_id)
            written += 1
            write_prog({"written": written, "last_id": ex_id, "ts": datetime.utcnow().isoformat() + "Z"})

    out_f.close(); err_f.close(); ckpt_f.close(); prog_f.close()
    print(f"Wrote {written} examples to {args.out}")


if __name__ == "__main__":
    main()


