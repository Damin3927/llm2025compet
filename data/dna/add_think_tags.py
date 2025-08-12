#!/usr/bin/env python3
"""
<think>タグ埋め込みスクリプト
argo11/DNA_DPO_hh-rlhf データセットに<think>タグを埋め込む

このスクリプトは、DPOトレーニング用データセットに必要な<think>タグを
OpenRouter APIを使用して自動生成します。
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# 依存関係のインポート（オプショナル）
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasetsライブラリがインストールされていません。")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openaiライブラリがインストールされていません。")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdmライブラリがインストールされていません。")

# .envファイルの読み込みを試行（エラーが発生しても続行）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenvライブラリがインストールされていません。")
except Exception as e:
    print(f"Warning: .envファイルの読み込みに失敗しました: {e}")


class ThinkTagGenerator:
    """<think>タグ生成を行うメインクラス"""

    def __init__(self, api_key: str, model_name: str):
        if not OPENAI_AVAILABLE:
            raise ImportError("openaiライブラリが必要です。pip install openai でインストールしてください。")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def _call_llm(self, prompt: str) -> Optional[str]:
        """LLM APIを呼び出し"""
        system_prompt = """You are an expert at generating thinking processes based on given questions and answers.
Analyze the content of the question and answer, and generate a short thinking process that leads to that answer.
The thinking process should be concise and include the judgment that forms the basis for the answer.
IMPORTANT: Output ONLY in English. Do not use any other language."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API呼び出しエラー: {e}")
            return None

    def _generate_think_for_preferred(self, question: str, answer: str) -> Optional[str]:
        """好ましい応答用の<think>タグ内容を生成"""
        prompt = f"""Analyze the following question and preferred answer, then generate a thinking process that explains why this response is appropriate.

Question: {question}
Preferred Answer: {answer}

Explain the judgment process that led to this response in a concise manner. If the question is inappropriate, explain why it's appropriate to refuse to answer it.
IMPORTANT: Output ONLY in English."""
        
        return self._call_llm(prompt)

    def _generate_think_for_non_preferred(self, question: str, answer: str) -> Optional[str]:
        """好ましくない応答用の<think>タグ内容を生成"""
        prompt = f"""Analyze the following question and non-preferred answer, then generate a thinking process that explains why this response is problematic.

Question: {question}
Non-preferred Answer: {answer}

Explain the reasons why this response is problematic or the inappropriate judgment process in a concise manner.
IMPORTANT: Output ONLY in English."""
        
        return self._call_llm(prompt)

    def _has_think_tag(self, text: str) -> bool:
        """既に<think>タグが存在するかチェック"""
        if not isinstance(text, str):
            return False
        return "<think>" in text.lower() and "</think>" in text.lower()

    def _add_think_tag(self, text: str, think_content: str) -> str:
        """テキストの先頭に<think>タグを追加"""
        if not think_content or not isinstance(text, str):
            return text
        return f"<think>{think_content}</think>{text}"

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """単一のデータアイテムを処理"""
        if not isinstance(item, dict):
            logging.warning("Invalid item format: not a dictionary")
            return item
            
        result = item.copy()
        
        # 必須フィールドの存在チェック
        required_fields = ["question", "preferred_output", "non_preferred_output"]
        for field in required_fields:
            if field not in item:
                logging.warning(f"Missing required field: {field}")
                return item
        
        question = item["question"]
        preferred_output = item["preferred_output"]
        non_preferred_output = item["non_preferred_output"]
        
        # preferred_outputの処理
        if not self._has_think_tag(preferred_output):
            think_content = self._generate_think_for_preferred(question, preferred_output)
            if think_content:
                result["preferred_output"] = self._add_think_tag(preferred_output, think_content)
                logging.info(f"ID {item.get('id', 'unknown')}: preferred_outputに<think>タグを追加")
        
        # non_preferred_outputの処理
        if not self._has_think_tag(non_preferred_output):
            think_content = self._generate_think_for_non_preferred(question, non_preferred_output)
            if think_content:
                result["non_preferred_output"] = self._add_think_tag(non_preferred_output, think_content)
                logging.info(f"ID {item.get('id', 'unknown')}: non_preferred_outputに<think>タグを追加")
        
        return result


def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    """ログ設定"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"think_tags_{timestamp}_{start_index}-{end_index-1}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_existing_ids(output_file: str) -> set:
    """既存の処理済みIDを読み込み"""
    existing_ids = set()
    output_path = Path(output_file)
    
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            existing_ids.add(data['id'])
                    except json.JSONDecodeError as e:
                        logging.warning(f"Line {line_num}: JSON解析エラー - {e}")
                        continue
        except Exception as e:
            logging.error(f"ファイル読み込みエラー: {e}")
    
    return existing_ids


def validate_arguments(args: argparse.Namespace) -> bool:
    """コマンドライン引数の検証"""
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


def main():
    parser = argparse.ArgumentParser(
        description="<think>タグ埋め込みスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python add_think_tags.py --start_index 0 --end_index 10
  python add_think_tags.py --start_index 100 --end_index 200 --model_name "qwen/qwen3-32b"
  python add_think_tags.py --start_index 0 --end_index 50 --output_file "custom_output.jsonl"
        """
    )
    
    parser.add_argument("--start_index", type=int, required=True, 
                       help="処理開始インデックス")
    parser.add_argument("--end_index", type=int, required=True, 
                       help="処理終了インデックス（この値は含まれない）")
    parser.add_argument("--output_file", type=str, 
                       help="出力ファイル名（指定しない場合は自動生成）")
    parser.add_argument("--model_name", type=str, 
                       default="qwen/qwen3-32b", 
                       help="使用するモデル名（OpenRouter対応、デフォルト: Qwen3-32B）")
    parser.add_argument("--log_dir", type=str, 
                       default="data/dna/logs", 
                       help="ログファイル保存ディレクトリ")
    parser.add_argument("--api_key", type=str, 
                       help="OpenRouter APIキー（環境変数より優先）")
    parser.add_argument("--dataset_name", type=str, 
                       default="argo11/DNA_DPO_hh-rlhf", 
                       help="使用するデータセット名")
    
    args = parser.parse_args()
    
    # 引数の検証
    if not validate_arguments(args):
        return 1
    
    # 出力ファイル名の設定
    if not args.output_file:
        args.output_file = f"data/dna/think_tagged_{args.start_index}-{args.end_index-1}.jsonl"
    
    # APIキーの設定（優先順位: コマンドライン引数 > 環境変数）
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("エラー: OPENROUTER_API_KEYが設定されていません")
        print("以下のいずれかの方法で設定してください:")
        print("1. --api_key コマンドライン引数で指定")
        print("2. 環境変数 OPENROUTER_API_KEY で設定")
        print("3. .envファイルで設定")
        return 1
    
    # ログ設定
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    logging.info(f"<think>タグ埋め込み処理開始: {args.start_index} - {args.end_index-1}")
    logging.info(f"モデル: {args.model_name}")
    logging.info(f"データセット: {args.dataset_name}")
    logging.info(f"出力ファイル: {args.output_file}")
    
    # 依存関係のチェック
    if not DATASETS_AVAILABLE:
        logging.error("datasetsライブラリがインストールされていません")
        print("エラー: 必要なライブラリがインストールされていません")
        print("pip install datasets でインストールしてください")
        return 1
    
    # データセット読み込み
    try:
        logging.info("データセット読み込み中...")
        dataset = load_dataset(args.dataset_name, split="train")
        logging.info(f"データセット読み込み完了: {len(dataset)} 件")
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        print(f"エラー: データセットの読み込みに失敗しました: {e}")
        return 1
    
    # <think>タグ生成器初期化
    try:
        generator = ThinkTagGenerator(api_key, args.model_name)
        logging.info("<think>タグ生成器初期化完了")
    except Exception as e:
        logging.error(f"生成器初期化エラー: {e}")
        print(f"エラー: <think>タグ生成器の初期化に失敗しました: {e}")
        return 1
    
    # 既存IDチェック
    existing_ids = load_existing_ids(args.output_file)
    logging.info(f"既存処理済み件数: {len(existing_ids)}")
    
    # 処理実行
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with output_path.open("a", encoding="utf-8") as f:
            # プログレスバーの設定
            if TQDM_AVAILABLE:
                iterator = tqdm(range(args.start_index, min(args.end_index, len(dataset))), 
                               desc="<think>タグ埋め込み処理")
            else:
                iterator = range(args.start_index, min(args.end_index, len(dataset)))
            
            for i in iterator:
                item = dataset[i]
                item_id = item.get('id', f'index_{i}')
                
                # スキップチェック
                if item_id in existing_ids:
                    skipped_count += 1
                    continue
                
                logging.info(f"処理中 ID: {item_id}")
                
                try:
                    # <think>タグ埋め込み実行
                    result = generator.process_item(item)
                    
                    # ファイル出力
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    
                    processed_count += 1
                    logging.info(f"ID {item_id}: 処理完了")
                    
                except Exception as e:
                    logging.error(f"ID {item_id} 処理エラー: {e}")
                    error_count += 1
                    
    except Exception as e:
        logging.error(f"ファイル出力エラー: {e}")
        print(f"エラー: ファイル出力に失敗しました: {e}")
        return 1
    
    # 結果の表示
    result_message = f"処理完了 - 処理件数: {processed_count}, エラー件数: {error_count}, スキップ件数: {skipped_count}"
    logging.info(result_message)
    print(result_message)
    print(f"出力ファイル: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
