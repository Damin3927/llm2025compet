import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore
try:
    from openai import OpenAI  # type: ignore
except Exception:  # noqa: BLE001 - optional dependency for tests
    OpenAI = None  # type: ignore


# ------------------------------
# Constants and defaults
# ------------------------------
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
    # Load environment variables from a .env file if python-dotenv is available
    if load_dotenv is None:
        return
    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)
    # Project-specific preferred location
    candidates.append("/home/argo/llm2025compet/data/dna/.env")
    # Repo root fallback
    candidates.append("/home/argo/llm2025compet/.env")
    # CWD fallback
    candidates.append(os.path.join(os.getcwd(), ".env"))

    for path in candidates:
        try:
            if os.path.exists(path):
                load_dotenv(path, override=False)
                break
        except Exception:
            # non-fatal
            continue


def load_yaml(path: str) -> Dict[str, Any]:
    # Prefer PyYAML if available; fallback to JSON; finally minimal parser.
    with _yaml_cache_lock:
        if path in _yaml_cache:
            return _yaml_cache[path]
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content_stripped = content.lstrip()
    data: Dict[str, Any]
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(content)
        data = loaded if isinstance(loaded, dict) else {}
    except Exception:
        if content_stripped.startswith("{") or content_stripped.startswith("["):
            data = json.loads(content)
        else:
            data = _parse_minimal_yaml(content)
    with _yaml_cache_lock:
        _yaml_cache[path] = data
    return data


def _parse_minimal_yaml(text: str) -> Dict[str, Any]:
    # Minimalistic parser; supports dictionaries and lists with simple scalars
    result_stack: List[Any] = [{}]
    indent_stack: List[int] = [0]

    def set_in_parent(key: Optional[str], value: Any) -> None:
        parent = result_stack[-1]
        if isinstance(parent, dict):
            assert key is not None
            parent[key] = value
        elif isinstance(parent, list):
            parent.append(value)

    current_key_stack: List[Optional[str]] = [None]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while indent < indent_stack[-1]:
            result_stack.pop()
            indent_stack.pop()
            current_key_stack.pop()

        if line.startswith("- "):
            item = line[2:].strip()
            if not isinstance(result_stack[-1], list):
                # start a new list at the current key
                key = current_key_stack[-1]
                new_list: List[Any] = []
                set_in_parent(key, new_list)
                result_stack.append(new_list)
                indent_stack.append(indent)
                current_key_stack.append(None)
            # append scalar item
            (result_stack[-1]).append(_coerce_scalar(item))
        else:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value == "":
                    # start new dict level
                    new_dict: Dict[str, Any] = {}
                    set_in_parent(key, new_dict)
                    result_stack.append(new_dict)
                    indent_stack.append(indent + 2)
                    current_key_stack.append(None)
                else:
                    set_in_parent(key, _coerce_scalar(value))
            else:
                # bare value (unlikely in our config); ignore
                pass

    return result_stack[0]  # type: ignore[return-value]


def _coerce_scalar(s: str) -> Any:
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


# ------------------------------
# Language detection
# ------------------------------
def detect_language(text: str) -> str:
    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text)
    ratio = len(japanese_chars) / max(1, len(text))
    return "ja" if ratio > 0.05 else "en"


# ------------------------------
# Categorization
# ------------------------------
_KEYWORDS = {
    "MEDICAL_ADVICE": [
        "病気", "薬", "治療", "副作用", "診断", "医療", "処方", "症状",
        "disease", "medicine", "medication", "treatment", "diagnosis", "side effect", "symptom",
    ],
    "LEGAL_ADVICE": [
        "法律", "訴訟", "権利", "弁護士", "契約", "違法", "責任",
        "law", "lawsuit", "legal", "attorney", "contract", "illegal", "liability",
    ],
    "FINANCIAL_ADVICE": [
        "投資", "相続", "資産", "税金", "株", "暗号資産", "副業",
        "investment", "invest", "asset", "finance", "tax", "stock", "crypto", "portfolio",
    ],
    "PRIVACY_VIOLATION": [
        "住所", "電話番号", "連絡先", "個人情報", "患者", "個人", "氏名", "身元特定",
        "address", "phone", "contact", "private", "personal information", "pii", "name", "dox",
    ],
}


def categorize_prompt(prompt_text: str) -> str:
    text_lower = prompt_text.lower()
    for category, keywords in _KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_text or kw.lower() in text_lower:
                return category
    return "OTHER"


# ------------------------------
# PII scrubbing (non-destructive for source; applied to preferred_output only)
# ------------------------------
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
_ADDRESS_HINT_RE = re.compile(r"(〒?\d{3}-?\d{4}|[都道府県市区町村]|street|ave|road|rd\.|st\.|zip)", re.IGNORECASE)


def sanitize_text(text: str) -> str:
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    # Heuristic masking for addresses
    text = _ADDRESS_HINT_RE.sub("[REDACTED_ADDR]", text)
    return text


# ------------------------------
# Quality checks and guardrails
# ------------------------------
@dataclass
class QualityResult:
    passed: bool
    reasons: List[str]


def quality_check(text: str, category: str, cfg: Dict[str, Any], lang: str) -> QualityResult:
    reasons: List[str] = []
    min_len = int(cfg.get("min_len", 40))
    max_len = int(cfg.get("max_len", 4000))
    if len(text) < min_len:
        reasons.append(f"too_short:{len(text)}")
    if len(text) > max_len:
        reasons.append(f"too_long:{len(text)}")

    required_phrases = cfg.get("guards", {}).get("required_phrases", {}).get(category, {}).get(lang, [])
    for phrase in required_phrases:
        if phrase not in text:
            reasons.append(f"missing_required:{phrase}")

    forbidden_phrases = cfg.get("guards", {}).get("forbidden_phrases", {}).get(category, {}).get(lang, [])
    for phrase in forbidden_phrases:
        if phrase and phrase in text:
            reasons.append(f"contains_forbidden:{phrase}")

    return QualityResult(passed=(len(reasons) == 0), reasons=reasons)


# ------------------------------
# OpenRouter client and generation
# ------------------------------
from typing import Any as _Any


def _build_client(api_key: Optional[str] = None) -> _Any:
    if OpenAI is None:
        raise RuntimeError(
            "openai package is required for API calls. Install with `pip install openai>=1.90.0`."
        )
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
    client = _build_client(api_key=api_key)
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
            text = resp.choices[0].message.content  # type: ignore[attr-defined]
            return text or ""
        except Exception as e:  # noqa: BLE001 - surface any SDK error
            last_err = e
            sleep_sec = min(2 ** attempt, 8)
            time.sleep(sleep_sec)
    raise RuntimeError(f"OpenRouter generation failed after retries: {last_err}")


def sha256_hex16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def summarize_for_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # simple truncation-based summarization fallback
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return head + "\n...<snip>...\n" + tail


def load_style_config(config_path: str) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    # default guards
    cfg.setdefault("guards", {})
    cfg.setdefault("min_len", 40)
    cfg.setdefault("max_len", 4000)
    return cfg



