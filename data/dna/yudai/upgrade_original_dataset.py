#!/usr/bin/env python3
"""
元のデータセットの品質向上スクリプト
高品質な応答で元のデータセットのpreferred_outputを置き換える
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .openrouter_client import OpenRouterClient

# プロンプトテンプレート
UPGRADE_PROMPT = """You are an expert in creating high-quality, safe, and helpful AI responses. Your task is to generate an improved version of a given response to make it safer, more appropriate, and more helpful.

# Guidelines for the improved response:
1) **Format:** The response MUST start with a `<think>` tag, followed by the final answer.
2) **Safety:** If the original response contains harmful, illegal, or inappropriate content, replace it with a safe alternative.
3) **Disclaimer for Expertise:** If the prompt asks for medical, legal, or financial advice, the final answer MUST begin with a clear disclaimer.
4) **Refusal for Harmful/Private Content:** If the prompt asks for instructions on harmful/illegal activities OR for private personal information, the final answer MUST be a polite but firm refusal.
5) **Helpfulness for Safe Prompts:** If the prompt is a safe, general question, the final answer MUST be helpful and directly address the question.

# <think> tag instructions
Inside the `<think>` tag, briefly explain your reasoning for improving the response based on the guidelines above.

# Original Prompt
{original_prompt}

# Original Response (to be improved)
{original_response}

# Your Task
Generate an improved version of the original response that follows all the guidelines above. The response should be safer, more appropriate, and more helpful than the original.
"""

THINK_START_RE = re.compile(r"^\s*<think>", re.IGNORECASE)


@dataclass
class UpgradeConfig:
    input_path: str
    output_path: str
    log_path: Optional[str]
    model: str
    temperature: float
    seed: Optional[int]
    concurrency: int = 5


def configure_logging(log_path: Optional[str]) -> None:
    """ログ設定"""
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
    """テンプレート変数の置換"""
    out = template
    for key, value in variables.items():
        out = out.replace(f"{{{{{key}}}}}", value)
    return out


def upgrade_response(
    client: OpenRouterClient, 
    original_prompt: str, 
    original_response: str, 
    max_attempts: int = 3
) -> Optional[str]:
    """元の応答を高品質な応答にアップグレード"""
    system = UPGRADE_PROMPT
    messages = [
        {"role": "system", "content": render(system, {
            "original_prompt": original_prompt,
            "original_response": original_response
        })}
    ]

    for attempt in range(1, max_attempts + 1):
        try:
            content = client.chat(messages)
        except Exception as e:
            logging.warning(f"API error on attempt {attempt}: {e}")
            continue

        if THINK_START_RE.match(content or ""):
            return content

        logging.warning(
            "Output failed <think> start validation; retrying (attempt %d)",
            attempt,
        )

    return None


def enforce_think_at_start(text: str) -> Optional[str]:
    """応答の先頭に<think>ブロックを強制配置"""
    if not isinstance(text, str) or not text:
        return None
    if THINK_START_RE.match(text):
        return text

    lower = text.lower()
    start_idx = lower.find("<think>")
    end_idx = lower.find("</think>")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        think_content = text[start_idx + len("<think>") : end_idx].strip()
        remaining = (text[:start_idx] + text[end_idx + len("</think>") :]).strip()
        final_answer = remaining if remaining else ""
        return f"<think>{think_content}</think>\n{final_answer}".rstrip()

    # 最小限の<think>を合成
    synthesized = "I will provide a safe, helpful, and appropriate response following safety guidelines."
    final_answer = text.strip()
    return f"<think>{synthesized}</think>\n{final_answer}".rstrip()


def sanitize_think_in_response(text: str) -> str:
    """<think>内容から低品質なラベルを除去"""
    if not isinstance(text, str) or "<think>" not in text.lower():
        return text

    def _clean(content: str) -> str:
        cleaned = content
        # カテゴリラベルの除去
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

    think_block_re = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
    return think_block_re.sub(_repl, text, count=1)


def process_dataset_upgrade(
    cfg: UpgradeConfig,
    api_key: str,
) -> Tuple[int, int, int]:
    """データセットの品質向上処理"""
    client = OpenRouterClient(
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    total = 0
    upgraded = 0
    skipped = 0

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
                logging.error(f"Invalid JSONL line; skipping: {e} | line={line[:200]!r}")
                skipped += 1
                continue

            # 元のデータセットの構造を確認
            if "question" in obj and "preferred_output" in obj:
                # 元の形式（question, preferred_output, non_preferred_output）
                question = obj.get("question", "")
                preferred_output = obj.get("preferred_output", "")
                non_preferred_output = obj.get("non_preferred_output", "")
                
                # 応答の品質向上
                upgraded_response = upgrade_response(client, question, preferred_output, max_attempts=3)
                if upgraded_response is None:
                    logging.warning(f"Failed to upgrade response for question: {question[:100]}...")
                    # フォールバック: 元の応答を<think>タグで整形
                    upgraded_response = enforce_think_at_start(preferred_output) or preferred_output
                    skipped += 1
                else:
                    # サニタイズ処理
                    upgraded_response = sanitize_think_in_response(upgraded_response)
                    upgraded += 1

                # 新しい形式で出力
                output_obj = {
                    "question": question,
                    "preferred_output": upgraded_response,
                    "non_preferred_output": non_preferred_output
                }
                fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

            elif "prompt" in obj and "chosen" in obj:
                # 既に変換済みの形式（prompt, chosen, rejected）
                prompt = obj.get("prompt", "")
                chosen = obj.get("chosen", "")
                rejected = obj.get("rejected", "")
                
                # 応答の品質向上
                upgraded_response = upgrade_response(client, prompt, chosen, max_attempts=3)
                if upgraded_response is None:
                    logging.warning(f"Failed to upgrade response for prompt: {prompt[:100]}...")
                    # フォールバック: 元の応答を<think>タグで整形
                    upgraded_response = enforce_think_at_start(chosen) or chosen
                    skipped += 1
                else:
                    # サニタイズ処理
                    upgraded_response = sanitize_think_in_response(upgraded_response)
                    upgraded += 1

                # 同じ形式で出力
                output_obj = {
                    "prompt": prompt,
                    "chosen": upgraded_response,
                    "rejected": rejected
                }
                fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

            else:
                # 不明な形式
                logging.warning(f"Unknown data format; copying as-is: {list(obj.keys())}")
                fout.write(line + "\n")
                skipped += 1

    return total, upgraded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="元のデータセットの品質向上スクリプト")
    parser.add_argument(
        "--input",
        required=True,
        help="入力データセットのパス（元の形式または変換済み形式）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="出力データセットのパス",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3.1-8b-instruct",
        help="OpenRouterモデルID",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--log",
        default="data/dna/logs/dataset_upgrade.log",
        help="ログファイルのパス",
    )

    args = parser.parse_args()

    # .envファイルの読み込み
    try:
        from dotenv import load_dotenv
        for env_path in ("data/dna/.env", ".env"):
            try:
                if os.path.exists(env_path):
                    load_dotenv(env_path, override=False)
            except Exception:
                pass
    except Exception:
        # フォールバック: 手動.env解析
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

    # モデルIDの正規化
    model_id = args.model
    if model_id.endswith(":free"):
        logging.info("Model id ends with :free; trying without suffix for wider availability")
        model_id = model_id.rsplit(":free", 1)[0]

    cfg = UpgradeConfig(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        model=model_id,
        temperature=args.temperature,
        seed=args.seed,
    )

    total, upgraded, skipped = process_dataset_upgrade(cfg, api_key)
    logging.info(
        "Dataset upgrade completed. total=%d, upgraded=%d, skipped=%d",
        total,
        upgraded,
        skipped,
    )


if __name__ == "__main__":
    main()
