#!/usr/bin/env python3
"""
OpenRouterを使わない<think>タグ埋め込みスクリプト（さくらDC用）

デフォルト: ルールベース（ヒューリスティック）で安全な短いメタ推論文を生成。
オプション: --use_transformers 指定時はローカルLLM（Transformers）で生成。

さくらDCのノード（CPU 120C/1.5TB, GPU H100x8）で安定動作するよう、
オフライン前提の実装。GPUはTransformers使用時のみ任意。
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Iterator, Tuple
from pathlib import Path


try:
    from datasets import load_dataset  # type: ignore
    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


class OfflineThinkGenerator:
    """オフライン<think>生成器（ヒューリスティック + 任意でTransformers）"""

    def __init__(self, use_transformers: bool = False, transformer_model: Optional[str] = None, device: str = "cpu"):
        self.use_transformers = use_transformers
        self.device = device
        self.pipeline = None

        if self.use_transformers:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
                import torch  # type: ignore
                model_name = transformer_model or "Qwen/Qwen2.5-1.5B-Instruct"
                tok = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else None)
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tok,
                    device=0 if (device == "cuda" and torch.cuda.is_available()) else -1,
                )
            except Exception as e:
                logging.warning(f"Transformers初期化に失敗しました。ヒューリスティックにフォールバック: {e}")
                self.use_transformers = False
                self.pipeline = None

    @staticmethod
    def _looks_refusal(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.lower()
        patterns = [
            r"\bi can't\b",
            r"\bcan not\b",
            r"\bunable to\b",
            r"\bi will not\b",
            r"\bsorry\b",
            r"\brefuse\b",
            r"\bdecline\b",
            r"できません",
            r"致し兼ねます",
        ]
        return any(re.search(p, t) for p in patterns)

    @staticmethod
    def _looks_sensitive(question: str) -> bool:
        if not isinstance(question, str):
            return False
        t = question.lower()
        keywords = [
            "bomb", "explosive", "weapon", "kill", "murder", "assassinate",
            "hack", "malware", "ddos", "phishing",
            "drug", "meth", "cocaine",
            "hate", "racist", "sexist",
            "personal data", "password", "credential",
        ]
        return any(k in t for k in keywords)

    def _heuristic_preferred(self, question: str, answer: str) -> str:
        q = question or ""
        a = answer or ""
        if self._looks_refusal(a) or self._looks_sensitive(q):
            return (
                "The preferred response aligns with safety guidelines by avoiding harmful or privacy-violating details. "
                "It focuses on responsible behavior and explains that refusal or redirection is appropriate for unsafe requests."
            )
        return (
            "The preferred response is concise and policy-aligned. It provides helpful information without exposing unsafe, illegal, or private content, "
            "and it stays within safety and factual boundaries."
        )

    def _heuristic_non_preferred(self, question: str, answer: str) -> str:
        return (
            "The non-preferred response is problematic because it risks encouraging harmful, illegal, or privacy-violating behavior, "
            "or it may include misleading or discriminatory content. A safer approach would be to refuse and provide a high-level, responsible alternative."
        )

    def _gen_with_transformers(self, prompt: str) -> Optional[str]:
        if not self.pipeline:
            return None
        try:
            out = self.pipeline(
                prompt,
                max_new_tokens=160,
                temperature=0.2,
                do_sample=False,
                pad_token_id=0,
            )
            if isinstance(out, list) and out:
                text = out[0].get("generated_text")
                if isinstance(text, str):
                    return text.strip()
        except Exception as e:
            logging.warning(f"Transformers生成エラー: {e}")
        return None

    def _prompt_preferred(self, question: str, answer: str) -> str:
        return (
            "Task: Generate a concise, policy-level thinking process that justifies the preferred answer.\n\n"
            "Guidelines:\n"
            "- Do NOT restate harmful details; keep it high-level and safe.\n"
            "- If the question is inappropriate, explain why refusal is appropriate.\n"
            "- English only, 2-4 sentences.\n\n"
            f"Question: {question}\n"
            f"Preferred Answer: {answer}\n\n"
            "Output:"
        )

    def _prompt_non_preferred(self, question: str, answer: str) -> str:
        return (
            "Task: Briefly explain why the non-preferred answer is problematic at a policy level.\n\n"
            "Guidelines:\n"
            "- Do NOT include dangerous details.\n"
            "- Refer to safety, legality, privacy, and non-discrimination.\n"
            "- English only, 2-4 sentences.\n\n"
            f"Question: {question}\n"
            f"Non-preferred Answer: {answer}\n\n"
            "Output:"
        )

    @staticmethod
    def _has_think_tag(text: Any) -> bool:
        if not isinstance(text, str):
            return False
        t = text.lower()
        return "<think>" in t and "</think>" in t

    @staticmethod
    def _add_think_tag(text: Any, think_content: Optional[str]) -> Any:
        if not isinstance(text, str) or not think_content:
            return text
        return f"<think>{think_content}</think>{text}"

    def process_item(self, item: Dict[str, Any], apply_to_non_preferred: bool) -> Dict[str, Any]:
        res = item.copy()
        rq = item.get("question")
        rp = item.get("preferred_output")
        rnp = item.get("non_preferred_output")

        question: str = str(rq) if isinstance(rq, (str, int, float)) else ""
        preferred_output: str = str(rp) if isinstance(rp, (str, int, float)) else ""
        non_preferred_output: str = str(rnp) if isinstance(rnp, (str, int, float)) else ""

        item_id = item.get("id", "unknown")
        logging.info(f"ID {item_id}: 処理開始")

        # preferred
        if preferred_output and not self._has_think_tag(preferred_output):
            if self.use_transformers:
                prompt = self._prompt_preferred(question, preferred_output)
                tc = self._gen_with_transformers(prompt)
                if not tc:
                    tc = self._heuristic_preferred(question, preferred_output)
            else:
                tc = self._heuristic_preferred(question, preferred_output)

            res["preferred_output"] = self._add_think_tag(preferred_output, tc)
            logging.info(f"ID {item_id}: preferred_outputに<think>タグ付与")
        else:
            logging.info(f"ID {item_id}: preferred_outputは既存<think>タグあり or 空")

        # non-preferred（任意）
        if apply_to_non_preferred and non_preferred_output:
            if not self._has_think_tag(non_preferred_output):
                if self.use_transformers:
                    prompt2 = self._prompt_non_preferred(question, non_preferred_output)
                    tc2 = self._gen_with_transformers(prompt2)
                    if not tc2:
                        tc2 = self._heuristic_non_preferred(question, non_preferred_output)
                else:
                    tc2 = self._heuristic_non_preferred(question, non_preferred_output)

                res["non_preferred_output"] = self._add_think_tag(non_preferred_output, tc2)
                logging.info(f"ID {item_id}: non_preferred_outputに<think>タグ付与")
            else:
                logging.info(f"ID {item_id}: non_preferred_outputは既存<think>タグあり")

        logging.info(f"ID {item_id}: 処理完了")
        return res


def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = p / f"think_tags_{ts}_{start_index}-{end_index-1}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_existing_ids(output_file: str) -> set:
    existing: set = set()
    p = Path(output_file)
    if not p.exists():
        return existing
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict) and "id" in data:
                        existing.add(data["id"])
                except Exception:
                    continue
    except Exception as e:
        logging.error(f"既存IDの読み込みエラー: {e}")
    return existing


def should_overwrite_file(output_file: str, start_index: int) -> bool:
    p = Path(output_file)
    if not p.exists():
        return True
    try:
        with p.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
            if not first:
                return True
            data = json.loads(first)
            if not isinstance(data, dict):
                return True
            if data.get("id") != start_index:
                return True
            if p.stat().st_size < 1000:
                return True
    except Exception:
        return True
    return False


def iter_range(dataset: Any, start_index: int, end_index: int) -> Iterator[Tuple[int, Dict[str, Any]]]:
    try:
        total = len(dataset)  # type: ignore
        end = min(end_index, total)
        for i in range(start_index, end):
            yield i, dataset[i]  # type: ignore[index]
    except Exception:
        for i, item in enumerate(dataset):  # type: ignore[assignment]
            if i < start_index:
                continue
            if i >= end_index:
                break
            yield i, item


def validate_arguments(args: argparse.Namespace) -> bool:
    if args.start_index < 0:
        print("エラー: start_indexは0以上である必要があります")
        return False
    if args.end_index <= args.start_index:
        print("エラー: end_indexはstart_indexより大きい必要があります")
        return False
    if args.use_transformers and args.device not in ("cpu", "cuda"):
        print("エラー: deviceは'cpu'または'cuda'")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="<think>タグ埋め込み（OpenRouter非依存）")
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True, help="この値は含まない")
    parser.add_argument("--output_file", type=str, help="未指定時は自動命名")
    parser.add_argument("--dataset_name", type=str, default="argo11/DNA_DPO_hh-rlhf")
    parser.add_argument("--log_dir", type=str, default="data/dna/logs")
    parser.add_argument("--apply_to_non_preferred", action="store_true")
    parser.add_argument("--use_transformers", action="store_true", help="ローカルLLMで生成（任意）")
    parser.add_argument("--transformer_model", type=str, help="Transformersモデル名（任意）")
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda（Transformers使用時）")

    args = parser.parse_args()

    if not validate_arguments(args):
        return 1

    if not args.output_file:
        args.output_file = f"data/dna/think_tagged_{args.start_index}-{args.end_index-1}.jsonl"

    setup_logging(args.log_dir, args.start_index, args.end_index)
    logging.info(f"開始: {args.start_index} - {args.end_index - 1}")
    logging.info(f"データセット: {args.dataset_name}")
    logging.info(f"出力: {args.output_file}")
    logging.info(f"Transformers: {args.use_transformers} / device={args.device} / model={args.transformer_model or ''}")

    if not DATASETS_AVAILABLE:
        logging.error("datasetsライブラリが見つかりません。pip install datasets を実行してください。")
        return 1

    try:
        dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        return 1

    generator = OfflineThinkGenerator(
        use_transformers=args.use_transformers,
        transformer_model=args.transformer_model,
        device=args.device,
    )

    overwrite = should_overwrite_file(args.output_file, args.start_index)
    existing_ids = set() if overwrite else load_existing_ids(args.output_file)
    logging.info(f"ファイルモード: {'w' if overwrite else 'a'} / 既存件数: {len(existing_ids)}")

    processed = 0
    errors = 0
    skipped = 0

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"

    try:
        with out_path.open(mode, encoding="utf-8") as f:
            iterator: Iterator[Tuple[int, Dict[str, Any]]] = iter_range(dataset, args.start_index, args.end_index)
            if TQDM_AVAILABLE:
                from typing import Iterable as _Iterable
                iterator = tqdm(iterator, desc="<think>タグ埋め込み処理", unit="it")  # type: ignore[assignment]

            for i, item in iterator:
                try:
                    item_id = item.get("id", f"index_{i}")
                    if item_id in existing_ids:
                        skipped += 1
                        continue

                    out = generator.process_item(item, apply_to_non_preferred=args.apply_to_non_preferred)
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    processed += 1
                except Exception as e:
                    logging.error(f"ID {item.get('id', f'index_{i}')} 処理エラー: {e}")
                    errors += 1
    except Exception as e:
        logging.error(f"ファイル出力エラー: {e}")
        return 1

    msg = f"処理完了 - 処理件数: {processed}, エラー件数: {errors}, スキップ件数: {skipped}"
    logging.info(msg)
    print(msg)
    print(f"出力ファイル: {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









