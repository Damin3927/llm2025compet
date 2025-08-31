import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple
import concurrent.futures

from .openrouter_client import OpenRouterClient


EVALUATOR_PROMPT = (
    "You are an expert AI data quality inspector. Your task is to evaluate a given DPO data sample (\"prompt\" and \"chosen\" response) based on a set of strict guidelines.\n\n"
    "# Guidelines\n"
    "A `chosen` response is considered \"GOOD\" only if it meets ALL of the following criteria:\n"
    "1.  **Format:** The response MUST start with a `<think>` tag, followed by the final answer.\n"
    "2.  **Disclaimer for Expertise:** If the `prompt` asks for medical, legal, or financial advice, the final answer MUST begin with a clear disclaimer (e.g., \"I am not a doctor...\").\n"
    "3.  **Refusal for Harmful/Private Content:** If the `prompt` asks for instructions on harmful/illegal activities OR for private personal information, the final answer MUST be a polite but firm refusal.\n"
    "4.  **Helpfulness for Safe Prompts:** If the `prompt` is a safe, general question (e.g., \"how to dispose of motor oil\"), the final answer MUST be helpful and directly address the question.\n\n"
    "# Your Task\n"
    "Analyze the following `prompt` and `chosen` response. Based on the guidelines above, determine if the `chosen` response is \"GOOD\" or \"BAD\".\n\n"
    "# Input\n"
    "- Prompt: \"{{PROMPT}}\"\n"
    "- Chosen Response: \"{{CHOSEN}}\"\n\n"
    "# Output\n"
    "Your output must be a single word: either \"GOOD\" or \"BAD\". Do not provide any other explanation or text."
)


REVISER_PROMPT = (
    "You are an expert in creating high-quality, safe, and helpful AI responses. A previous AI response was found to be flawed. Generate a new, high-quality response to the user's prompt based on the strict guidelines below.\n\n"
    "# Guidelines for the new response:\n"
    "1) Format: Your output MUST begin with '<think>' as the VERY FIRST characters (no preface), then close with '</think>' before the final answer.\n"
    "2) Internally decide which policy case applies: expertise (medical/legal/financial), harmful_or_private (illegal/harmful instructions or private personal info), or safe.\n"
    "3) For expertise: START the final answer with a clear disclaimer (e.g., 'I am not a doctor...').\n"
    "4) For harmful_or_private: The final answer MUST be a polite but firm refusal. Optionally offer high-level safe alternatives or resources.\n"
    "5) For safe: DO NOT refuse. Provide a concise, directly helpful answer that addresses the user's request.\n\n"
    "# <think> tag instructions\n"
    "Inside '<think>', briefly explain the rationale for your chosen strategy in 1-3 sentences. Do NOT include category labels, letters, or words like '(a)', '(b)', '(c)', 'safe', 'harmful', 'private', or 'expertise'. Start EXACTLY with '<think>'."
)


GOOD_RE = re.compile(r"^\s*good\s*$", re.IGNORECASE)
BAD_RE = re.compile(r"^\s*bad\s*$", re.IGNORECASE)
THINK_START_RE = re.compile(r"^\s*<think>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


@dataclass
class RunConfig:
    input_path: str
    output_path: str
    log_path: Optional[str]
    model: str
    evaluator_temperature: float
    reviser_temperature: float
    seed: Optional[int]
    concurrency: int = 5


def configure_logging(log_path: Optional[str]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def render(template: str, variables: Dict[str, str]) -> str:
    out = template
    for key, value in variables.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def evaluate_sample(
    client: OpenRouterClient, prompt: str, chosen: str, max_attempts: int = 3
) -> Tuple[Optional[str], Optional[str]]:
    """Returns ("GOOD"|"BAD"|None, last_raw_output).

    Strictly parses evaluator output to only accept GOOD/BAD. When invalid after
    retries, returns (None, last_raw_output) for logging.
    """
    system = EVALUATOR_PROMPT
    messages = [
        {"role": "system", "content": render(system, {"PROMPT": prompt, "CHOSEN": chosen})}
    ]

    last_raw: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        try:
            content = client.chat(messages)
        except Exception as e:
            logging.warning(f"Evaluator API error on attempt {attempt}: {e}")
            continue

        last_raw = content
        if GOOD_RE.match(content):
            return "GOOD", last_raw
        if BAD_RE.match(content):
            return "BAD", last_raw
        logging.warning(f"Evaluator invalid output (attempt {attempt}): {content!r}")

    return None, last_raw


def revise_chosen(client: OpenRouterClient, prompt: str, max_attempts: int = 3) -> Optional[str]:
    """Generate a new chosen response that starts with <think> and follows rules.

    Validates the basic format (<think> at start). Content policy satisfaction is
    left to the evaluator second pass.
    """
    system = REVISER_PROMPT
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            messages = list(base_messages)
            if attempt > 1:
                messages.append(
                    {
                        "role": "system",
                        "content": "REMINDER: Start your response with '<think>' immediately. No preface, no labels, no apologies."
                    }
                )
            content = client.chat(messages)
        except Exception as e:
            logging.warning(f"Reviser API error on attempt {attempt}: {e}")
            continue

        # Enforce <think> at the very start via post-processing if needed
        fixed = enforce_think_at_start(content or "")
        if fixed is not None and THINK_START_RE.match(fixed):
            sanitized = sanitize_think_in_response(fixed)
            return sanitized

        logging.warning(
            "Reviser output failed <think> start validation; retrying (attempt %d)",
            attempt,
        )

    return None


def enforce_think_at_start(text: str) -> Optional[str]:
    """Ensure the response begins with a <think> block followed by the final answer.

    Strategy:
    1) If already starts with <think>, return as-is.
    2) If contains a <think>...</think> block later, move the first such block to the front,
       and append the remaining text as the final answer.
    3) Otherwise, synthesize a brief <think> and place the original text as the final answer.
    """
    if not isinstance(text, str) or not text:
        return None
    if THINK_START_RE.match(text):
        return text

    lower = text.lower()
    start_idx = lower.find("<think>")
    end_idx = lower.find("</think>")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        think_content = text[start_idx + len("<think>") : end_idx].strip()
        # remove the detected block from original
        remaining = (text[:start_idx] + text[end_idx + len("</think>") :]).strip()
        final_answer = remaining if remaining else ""
        return f"<think>{think_content}</think>\n{final_answer}".rstrip()

    # Synthesize minimal think
    synthesized = "I will follow the safety and helpfulness guidelines, providing a refusal or a helpful answer as appropriate."
    final_answer = text.strip()
    return f"<think>{synthesized}</think>\n{final_answer}".rstrip()


def sanitize_think_in_response(text: str) -> str:
    """Remove low-signal category markings like (a)/(b)/(c) or 'safe/harmful/private/expertise' labels from <think> content.

    - Cleans only inside the first <think>...</think> block.
    - Does not alter the final answer content.
    """
    if not isinstance(text, str) or "<think>" not in text.lower():
        return text

    def _clean(content: str) -> str:
        cleaned = content
        # remove leading category hints
        cleaned = re.sub(
            r"^\s*(i\s+have\s+categorized\s+this\s+prompt\s+as\s+)?\([abcABC]\)\s*(?:-|:)?\s*(safe|harmful(?:_or_)?private|expertise)?[\s,:.-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^\s*decision\s*[:-]?\s*\([abcABC]\)\s*[\w_-]*\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\([abcABC]\)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        return f"<think>{_clean(inner)}</think>"

    return THINK_BLOCK_RE.sub(_repl, text, count=1)


def process_stream(
    cfg: RunConfig,
    api_key: str,
) -> Tuple[int, int, int]:
    """Process input JSONL with bounded concurrency and order-preserving writes.

    Returns (num_total, num_bad_fixed, num_skipped).
    """
    evaluator = OpenRouterClient(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.evaluator_temperature,
        seed=cfg.seed,
    )
    reviser = OpenRouterClient(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.reviser_temperature,
        seed=cfg.seed,
    )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    total = 0
    bad_fixed = 0
    skipped = 0

    def worker(index: int, prompt: str, chosen: str, rejected: str) -> Tuple[int, Optional[str], int, int]:
        local_bad_fixed = 0
        local_skipped = 0

        verdict, last_raw = evaluate_sample(evaluator, prompt, chosen, max_attempts=3)
        if verdict is None:
            logging.error(
                "Evaluator invalid after retries; skipping sample | prompt=%r | chosen=%r | raw=%r",
                prompt,
                chosen,
                last_raw,
            )
            local_skipped += 1
            return index, None, local_bad_fixed, local_skipped

        if verdict == "GOOD" and THINK_START_RE.match(chosen or ""):
            out = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            return index, json.dumps(out, ensure_ascii=False) + "\n", local_bad_fixed, local_skipped
        # If evaluator says GOOD but format is wrong, try local format-fix first then re-evaluate
        if verdict == "GOOD" and not THINK_START_RE.match(chosen or ""):
            fixed_chosen = enforce_think_at_start(chosen or "") or ""
            re_verdict, _ = evaluate_sample(evaluator, prompt, fixed_chosen, max_attempts=3)
            if re_verdict == "GOOD":
                out = {"prompt": prompt, "chosen": fixed_chosen, "rejected": rejected}
                local_bad_fixed += 1
                return index, json.dumps(out, ensure_ascii=False) + "\n", local_bad_fixed, local_skipped
            # fallthrough to BAD path

        # BAD -> revise
        new_chosen = revise_chosen(reviser, prompt, max_attempts=3)
        if new_chosen is None:
            logging.error(
                "Reviser failed to produce valid output; skipping sample | prompt=%r",
                prompt,
            )
            local_skipped += 1
            return index, None, local_bad_fixed, local_skipped

        # Optional: Re-evaluate to ensure it passes the evaluator
        re_verdict, _ = evaluate_sample(evaluator, prompt, new_chosen, max_attempts=3)
        if re_verdict != "GOOD":
            logging.warning(
                "Revised answer still not GOOD (verdict=%s); keeping revised anyway for audit",
                re_verdict,
            )

        out = {"prompt": prompt, "chosen": new_chosen, "rejected": rejected}
        local_bad_fixed += 1
        return index, json.dumps(out, ensure_ascii=False) + "\n", local_bad_fixed, local_skipped

    with open(cfg.input_path, "r", encoding="utf-8") as fin, open(
        cfg.output_path, cfg.__dict__.get("_file_mode", "w"), encoding="utf-8"
    ) as fout, concurrent.futures.ThreadPoolExecutor(max_workers=cfg.concurrency) as executor:
        pending: Dict[concurrent.futures.Future, int] = {}
        buffer: Dict[int, str] = {}
        next_index = cfg.__dict__.get("_start_index", 0)

        processed = 0
        for index, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            # skip until start index
            if index < next_index:
                continue
            # limit
            if cfg.__dict__.get("_limit", 0) and processed >= cfg.__dict__["_limit"]:
                break
            total += 1
            processed += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                logging.error(f"Invalid JSONL line; skipping: {e} | line={line[:200]!r}")
                skipped += 1
                continue

            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")

            # throttle number of in-flight tasks to keep memory bounded
            while len(pending) >= cfg.concurrency * 2:
                done, _ = concurrent.futures.wait(
                    pending.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    idx, out_line, b_fixed, sk = fut.result()
                    pending.pop(fut, None)
                    if out_line is not None:
                        buffer[idx] = out_line
                    bad_fixed += b_fixed
                    skipped += sk
                # flush in-order
                while next_index in buffer:
                    fout.write(buffer.pop(next_index))
                    next_index += 1

            fut = executor.submit(worker, index, prompt, chosen, rejected)
            pending[fut] = index

        # drain remaining tasks
        while pending:
            done, _ = concurrent.futures.wait(
                pending.keys(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                idx, out_line, b_fixed, sk = fut.result()
                pending.pop(fut, None)
                if out_line is not None:
                    buffer[idx] = out_line
                bad_fixed += b_fixed
                skipped += sk
            while next_index in buffer:
                fout.write(buffer.pop(next_index))
                next_index += 1

    return total, bad_fixed, skipped


def fix_missing_think(
    cfg: RunConfig,
    api_key: str,
) -> Tuple[int, int, int]:
    """Pass over input JSONL and fix only entries whose chosen lacks <think> at start.

    Returns (num_total, num_fixed, num_skipped)
    """
    evaluator = OpenRouterClient(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.evaluator_temperature,
        seed=cfg.seed,
    )
    reviser = OpenRouterClient(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.reviser_temperature,
        seed=cfg.seed,
    )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    total = 0
    fixed = 0
    skipped = 0

    with open(cfg.input_path, "r", encoding="utf-8") as fin, open(
        cfg.output_path, cfg.__dict__.get("_file_mode", "w"), encoding="utf-8"
    ) as fout:
        processed = 0
        for index, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if index < cfg.__dict__.get("_start_index", 0):
                continue
            if cfg.__dict__.get("_limit", 0) and processed >= cfg.__dict__["_limit"]:
                break
            total += 1
            processed += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                logging.error(f"Invalid JSONL line; skipping: {e} | line={line[:200]!r}")
                skipped += 1
                continue

            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")

            if THINK_START_RE.match(chosen or ""):
                fout.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
                continue

            # Try local format-fix first
            fixed_chosen = enforce_think_at_start(chosen or "") or ""
            re_verdict, _ = evaluate_sample(evaluator, prompt, fixed_chosen, max_attempts=3)
            if re_verdict == "GOOD":
                fout.write(json.dumps({"prompt": prompt, "chosen": fixed_chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
                fixed += 1
                continue

            # If still not GOOD, invoke reviser
            new_chosen = revise_chosen(reviser, prompt, max_attempts=3)
            if new_chosen is None:
                logging.error("Fix pass: reviser failed; leaving original unchanged | prompt=%r", prompt)
                fout.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
                skipped += 1
                continue

            re_verdict, _ = evaluate_sample(evaluator, prompt, new_chosen, max_attempts=3)
            if re_verdict != "GOOD":
                logging.warning("Fix pass: revised still %s; writing revised anyway", re_verdict)

            fout.write(json.dumps({"prompt": prompt, "chosen": new_chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            fixed += 1

    return total, fixed, skipped


def force_format_only(cfg: RunConfig) -> Tuple[int, int]:
    """Rewrite chosen to ensure <think> at start without any LLM calls.

    Returns (total, formatted)
    """
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    total = 0
    formatted = 0
    with open(cfg.input_path, "r", encoding="utf-8") as fin, open(
        cfg.output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                logging.error(f"Invalid JSONL line; copying as-is: {e} | line={line[:200]!r}")
                fout.write(line + "\n")
                continue

            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "") or ""
            rejected = obj.get("rejected", "")

            if not THINK_START_RE.match(chosen):
                fixed_chosen = enforce_think_at_start(chosen) or ""
                obj["chosen"] = fixed_chosen
                formatted += 1

            fout.write(json.dumps({"prompt": prompt, "chosen": obj["chosen"], "rejected": rejected}, ensure_ascii=False) + "\n")

    return total, formatted


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO v2 dataset quality enhancer")
    parser.add_argument(
        "--input",
        default="data/dna/processed/dataset_v2_mvp.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="data/dna/processed/dataset_v2_final.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--model",
        default="openrouter/auto",
        help="OpenRouter model id (e.g., openrouter/auto or meta-llama/llama-3.1-8b-instruct)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluator-temperature", type=float, default=0.1)
    parser.add_argument("--reviser-temperature", type=float, default=0.5)
    parser.add_argument(
        "--log",
        default="data/dna/logs/dpo_v2_quality_enhancer.log",
        help="Log file path",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--fix-missing-think", action="store_true", help="Fix only entries without <think> at start in the input JSONL")
    parser.add_argument("--force-format-only", action="store_true", help="Do not call LLMs. Ensure every chosen starts with <think> via local rewrite only.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file line-count; appends new results")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N input lines (overrides resume)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N input lines (0 for all)")

    args = parser.parse_args()

    # Load .env explicitly (prefer data/dna/.env, then fallback to .env)
    try:
        from dotenv import load_dotenv  # type: ignore
        for env_path in ("data/dna/.env", ".env"):
            try:
                if os.path.exists(env_path):
                    load_dotenv(env_path, override=False)
            except Exception:
                pass
    except Exception:
        # Fallback: manual .env parsing if python-dotenv is unavailable
        def _manual_load_env(path: str) -> None:
            if not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = val
            except Exception:
                pass

        _manual_load_env("data/dna/.env")
        _manual_load_env(".env")

    configure_logging(args.log)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY が見つかりません。.env に設定するか環境変数をエクスポートしてください"
        )

    # normalize model id (avoid unavailable suffixes like :free)
    model_id = args.model
    if model_id.endswith(":free"):
        logging.info("Model id ends with :free; trying without suffix for wider availability")
        model_id = model_id.rsplit(":free", 1)[0]

    cfg = RunConfig(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        model=model_id,
        evaluator_temperature=args.evaluator_temperature,
        reviser_temperature=args.reviser_temperature,
        seed=args.seed,
        concurrency=max(1, args.concurrency),
    )

    # resume / skip / limit handling
    start_index = 0
    file_mode = "w"
    if args.resume and os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                start_index = sum(1 for _ in f)
            file_mode = "a"
            logging.info("Resume enabled: start_index=%d (append mode)", start_index)
        except Exception as e:
            logging.warning("Resume failed to read output; starting fresh: %s", e)
            start_index = 0
            file_mode = "w"

    if args.skip > 0:
        start_index = args.skip
        file_mode = "a" if (args.resume and os.path.exists(args.output)) else "w"
        logging.info("Skip override: start_index=%d", start_index)

    # inject into cfg (private fields used internally)
    cfg.__dict__["_start_index"] = max(0, int(start_index))
    cfg.__dict__["_file_mode"] = file_mode
    cfg.__dict__["_limit"] = max(0, int(args.limit))

    if args.force_format_only:
        total, fixed = force_format_only(cfg)
        logging.info(
            "Force-format completed. total=%d, formatted=%d",
            total,
            fixed,
        )
    elif args.fix_missing_think:
        total, fixed, skipped = fix_missing_think(cfg, api_key)
        logging.info(
            "Fix pass completed. total=%d, fixed_missing_think=%d, skipped=%d",
            total,
            fixed,
            skipped,
        )
    else:
        total, bad_fixed, skipped = process_stream(cfg, api_key)
        logging.info(
            "Completed. total=%d, bad_fixed=%d, skipped=%d",
            total,
            bad_fixed,
            skipped,
        )


if __name__ == "__main__":
    main()


