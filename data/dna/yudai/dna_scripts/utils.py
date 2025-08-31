import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "meta-llama/llama-3.1-70b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 512
DEFAULT_SEED = 42

_yaml_cache_lock = threading.Lock()
_yaml_cache: Dict[str, Dict[str, Any]] = {}


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def init_env(env_path: Optional[str] = None) -> None:
    if load_dotenv is None:
        return
    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)
    base = "/home/argo/llm2025compet/data/dna"
    candidates.append(os.path.join(base, ".env"))
    candidates.append(os.path.join(os.getcwd(), ".env"))
    for path in candidates:
        try:
            if os.path.exists(path):
                load_dotenv(path, override=False)
                break
        except Exception:
            continue


def sha256_hex16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def detect_language(text: str) -> str:
    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text)
    ratio = len(japanese_chars) / max(1, len(text))
    return "ja" if ratio > 0.05 else "en"


def sanitize_text(text: str, preserve_think: bool = False, scrub_cot: bool = False) -> str:
    """Sanitize output.
    - If preserve_think is False: drop <think>...</think>
    - If preserve_think is True: keep <think>...</think>; optionally scrub inside when scrub_cot=True
    - Always scrub the final answer section for PII
    """
    think_part = ""
    final_part = text
    m = re.search(r"(?is)(<think>[\s\S]*?</think>)([\s\S]*)$", text)
    if m:
        think_part = m.group(1)
        final_part = m.group(2)

    def scrub(s: str) -> str:
        s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", s)
        s = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[REDACTED_PHONE]", s)
        addr_pattern = r"\b(zip|street|road|avenue|city|state|st\.|rd\.|ave\.)\b"
        s = re.sub(addr_pattern, "[REDACTED_ADDR]", s, flags=re.IGNORECASE)
        return s

    if not preserve_think:
        # Drop think entirely; sanitize combined text
        combined = final_part
        combined = scrub(combined)
        return combined

    # preserve think; optionally scrub inside CoT
    if scrub_cot and think_part:
        inner = re.sub(r"(?is)^<think>|</think>$", "", think_part)
        inner = scrub(inner)
        think_part = f"<think>{inner}</think>"

    final_part = scrub(final_part)
    return f"{think_part}{final_part}"


@dataclass
class QualityResult:
    passed: bool
    reasons: List[str]


def quality_check(text: str, category: str, cfg: Dict[str, Any], lang: str) -> QualityResult:
    reasons: List[str] = []
    min_len = int(cfg.get("min_len", 60))
    max_len = int(cfg.get("max_len", 3500))
    if len(text) < min_len:
        reasons.append(f"too_short:{len(text)}")
    if len(text) > max_len:
        reasons.append(f"too_long:{len(text)}")
    req = cfg.get("guards", {}).get("required_phrases", {}).get(category, {}).get(lang, [])
    for phrase in req:
        if phrase not in text:
            reasons.append(f"missing_required:{phrase}")
    forb = cfg.get("guards", {}).get("forbidden_phrases", {}).get(category, {}).get(lang, [])
    for phrase in forb:
        if phrase and phrase in text:
            reasons.append(f"contains_forbidden:{phrase}")
    return QualityResult(passed=(len(reasons) == 0), reasons=reasons)


def _build_client(api_key: Optional[str] = None) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is required. `pip install openai>=1.90.0`.")
    k = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=k)


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    timeout_seconds: int = 60,
    max_retries: int = 3,
    api_key: Optional[str] = None,
) -> str:
    client = _build_client(api_key)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                timeout=timeout_seconds,
            )
            text = resp.choices[0].message.content
            return text or ""
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"OpenRouter generation failed: {last_err}")


