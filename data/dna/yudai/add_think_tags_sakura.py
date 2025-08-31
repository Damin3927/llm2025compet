#!/usr/bin/env python3
"""
さくらDCサーバ（Slurm環境）での実行を想定した<think>タグ埋め込みスクリプト（別バージョン）

ポイント:
- ログは `data/dna/logs/` に保存
- Hugging Face Datasets を使用（Iterableな場合にも対応）
- OpenRouter API（openaiパッケージ経由）で短いメタ推論文を生成
- デフォルトでは preferred_output のみ付与。必要なら --apply_to_non_preferred を指定
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Iterable, Iterator, Tuple
from pathlib import Path


# 依存関係（存在しなければエラーメッセージを出す）
try:
    from datasets import load_dataset  # type: ignore
    DATASETS_AVAILABLE = True
except Exception:
    DATASETS_AVAILABLE = False

try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# .env を読む（存在すれば）
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


class ThinkTagGenerator:
    """<think>タグ生成クラス（OpenRouter経由）"""

    def __init__(self, api_key: str, model_name: str, apply_to_non_preferred: bool = False):
        if not OPENAI_AVAILABLE:
            raise ImportError("openaiライブラリが必要です。pip install openai を実行してください。")

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_name = model_name
        self.apply_to_non_preferred = apply_to_non_preferred

    def _call_llm(self, prompt: str) -> Optional[str]:
        """LLMをコールし、content文字列のみを安全に抽出（reasoning拡張などには依存しない）"""
        system_prompt = (
            "You are an expert at generating thinking processes based on given questions and answers.\n"
            "Analyze the content and produce a short, policy-level reasoning.\n"
            "IMPORTANT: Output ONLY in English."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.2,
            )

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                # openai>=1.x 互換：choice.message.content を優先
                try:
                    content = getattr(getattr(choice, "message", None), "content", None)
                except Exception:
                    content = None

                if isinstance(content, str) and content.strip():
                    return content.strip()

            logging.warning("LLM応答からcontentを取得できませんでした")
            return None
        except Exception as e:
            logging.error(f"LLM呼び出しエラー: {e}")
            return None

    def _generate_think_for_preferred(self, question: str, answer: str) -> Optional[str]:
        prompt = (
            "Task: Generate a concise, policy-level thinking process that justifies the preferred answer.\n\n"
            "Guidelines:\n"
            "- Do NOT restate harmful details; keep it high-level and safe.\n"
            "- If the question is inappropriate, explain why refusal is appropriate.\n"
            "- English only, 2-4 sentences.\n\n"
            f"Question: {question}\n"
            f"Preferred Answer: {answer}\n\n"
            "Output:"
        )
        return self._call_llm(prompt)

    def _generate_think_for_non_preferred(self, question: str, answer: str) -> Optional[str]:
        prompt = (
            "Task: Briefly explain why the non-preferred answer is problematic at a policy level.\n\n"
            "Guidelines:\n"
            "- Do NOT include dangerous details.\n"
            "- Refer to safety, legality, privacy, and non-discrimination.\n"
            "- English only, 2-4 sentences.\n\n"
            f"Question: {question}\n"
            f"Non-preferred Answer: {answer}\n\n"
            "Output:"
        )
        return self._call_llm(prompt)

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

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """1件処理（preferredは必須。非preferredはフラグで制御）"""
        result = item.copy()

        raw_question = item.get("question")
        raw_preferred = item.get("preferred_output")
        raw_non_preferred = item.get("non_preferred_output")

        # Noneガードと型の正規化
        question: str = str(raw_question) if isinstance(raw_question, (str, int, float)) else ""
        preferred_output: str = str(raw_preferred) if isinstance(raw_preferred, (str, int, float)) else ""
        non_preferred_output: str = (
            str(raw_non_preferred) if isinstance(raw_non_preferred, (str, int, float)) else ""
        )

        item_id = item.get("id", "unknown")
        logging.info(f"ID {item_id}: 処理開始")

        # preferred
        if preferred_output and not self._has_think_tag(preferred_output):
            tc = self._generate_think_for_preferred(question, preferred_output)
            if tc:
                result["preferred_output"] = self._add_think_tag(preferred_output, tc)
                logging.info(f"ID {item_id}: preferred_outputに<think>タグ付与")
            else:
                logging.warning(f"ID {item_id}: preferred_outputの<think>生成に失敗")
        else:
            logging.info(f"ID {item_id}: preferred_outputは既に<think>タグあり")

        # non-preferred（必要時のみ）
        if self.apply_to_non_preferred and non_preferred_output:
            if not self._has_think_tag(non_preferred_output):
                tc2 = self._generate_think_for_non_preferred(question, non_preferred_output)
                if tc2:
                    result["non_preferred_output"] = self._add_think_tag(non_preferred_output, tc2)
                    logging.info(f"ID {item_id}: non_preferred_outputに<think>タグ付与")
                else:
                    logging.warning(f"ID {item_id}: non_preferred_outputの<think>生成に失敗")
            else:
                logging.info(f"ID {item_id}: non_preferred_outputは既に<think>タグあり")

        logging.info(f"ID {item_id}: 処理完了")
        return result


def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = path / f"think_tags_{ts}_{start_index}-{end_index-1}.log"
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
    """インデックスアクセス不可(IterableDataset)でも範囲反復できるようにする"""
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
    if args.model_name and not args.model_name.strip():
        print("エラー: model_nameは空文字列にできません")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="<think>タグ埋め込みスクリプト（さくらDC用別版）")
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--end_index", type=int, required=True, help="この値は含まない")
    parser.add_argument("--output_file", type=str, help="未指定時は自動命名")
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen/qwen3-32b-instruct",
        help="OpenRouter対応モデル名",
    )
    parser.add_argument("--log_dir", type=str, default="data/dna/logs")
    parser.add_argument("--api_key", type=str, help="OPENROUTER_API_KEY（未指定時は環境変数を参照）")
    parser.add_argument("--dataset_name", type=str, default="argo11/DNA_DPO_hh-rlhf")
    parser.add_argument("--apply_to_non_preferred", action="store_true")

    args = parser.parse_args()

    if not validate_arguments(args):
        return 1

    if not args.output_file:
        args.output_file = f"data/dna/think_tagged_{args.start_index}-{args.end_index-1}.jsonl"

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("エラー: OPENROUTER_API_KEYが未設定です。--api_key か環境変数で設定してください。")
        return 1

    setup_logging(args.log_dir, args.start_index, args.end_index)
    logging.info(f"開始: {args.start_index} - {args.end_index - 1}")
    logging.info(f"モデル: {args.model_name}")
    logging.info(f"データセット: {args.dataset_name}")
    logging.info(f"出力: {args.output_file}")

    if not DATASETS_AVAILABLE:
        logging.error("datasetsライブラリが見つかりません。pip install datasets を実行してください。")
        return 1

    try:
        logging.info("データセット読み込み中...")
        dataset = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        return 1

    try:
        generator = ThinkTagGenerator(api_key, args.model_name, apply_to_non_preferred=args.apply_to_non_preferred)
    except Exception as e:
        logging.error(f"生成器初期化エラー: {e}")
        return 1

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
            iterator = iter_range(dataset, args.start_index, args.end_index)
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, desc="<think>タグ埋め込み処理", unit="it")  # type: ignore

            for i, item in iterator:
                try:
                    item_id = item.get("id", f"index_{i}")
                    if item_id in existing_ids:
                        skipped += 1
                        continue

                    result = generator.process_item(item)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
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


