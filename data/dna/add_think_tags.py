#!/usr/bin/env python3
"""
<think>タグ埋め込みスクリプト
argo11/DNA_DPO_hh-rlhf データセットに<think>タグを埋め込む
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

class ThinkTagGenerator:
    """<think>タグ生成を行うメインクラス"""

    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name

    def _call_llm(self, prompt: str) -> Optional[str]:
        """LLM APIを呼び出し"""
        system_prompt = """あなたは、与えられた質問と回答に基づいて、その回答に至るまでの思考プロセスを生成する専門家です。
質問の内容と回答を分析し、その回答を導き出すための短い思考プロセスを生成してください。
思考プロセスは簡潔で、回答の根拠となる判断を含めてください。"""
        
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
        prompt = f"""以下の質問と好ましい応答を分析し、なぜこの応答が適切なのかを示す思考プロセスを生成してください。

質問: {question}
好ましい応答: {answer}

この応答に至るまでの判断プロセスを簡潔に説明してください。特に、質問が不適切な場合は、なぜ拒否することが適切なのかを説明してください。"""
        
        return self._call_llm(prompt)

    def _generate_think_for_non_preferred(self, question: str, answer: str) -> Optional[str]:
        """好ましくない応答用の<think>タグ内容を生成"""
        prompt = f"""以下の質問と好ましくない応答を分析し、なぜこの応答が問題があるのかを示す思考プロセスを生成してください。

質問: {question}
好ましくない応答: {answer}

この応答が問題である理由や、適切でない判断プロセスを簡潔に説明してください。"""
        
        return self._call_llm(prompt)

    def _has_think_tag(self, text: str) -> bool:
        """既に<think>タグが存在するかチェック"""
        return "<think>" in text.lower() and "</think>" in text.lower()

    def _add_think_tag(self, text: str, think_content: str) -> str:
        """テキストの先頭に<think>タグを追加"""
        if not think_content:
            return text
        return f"<think>{think_content}</think>{text}"

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """単一のデータアイテムを処理"""
        result = item.copy()
        
        question = item["question"]
        preferred_output = item["preferred_output"]
        non_preferred_output = item["non_preferred_output"]
        
        # preferred_outputの処理
        if not self._has_think_tag(preferred_output):
            think_content = self._generate_think_for_preferred(question, preferred_output)
            if think_content:
                result["preferred_output"] = self._add_think_tag(preferred_output, think_content)
                logging.info(f"ID {item['id']}: preferred_outputに<think>タグを追加")
        
        # non_preferred_outputの処理
        if not self._has_think_tag(non_preferred_output):
            think_content = self._generate_think_for_non_preferred(question, non_preferred_output)
            if think_content:
                result["non_preferred_output"] = self._add_think_tag(non_preferred_output, think_content)
                logging.info(f"ID {item['id']}: non_preferred_outputに<think>タグを追加")
        
        return result


def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"think_tags_{timestamp}_{start_index}-{end_index-1}.log")
    
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
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue
    return existing_ids


def main():
    parser = argparse.ArgumentParser(description="<think>タグ埋め込みスクリプト")
    parser.add_argument("--start_index", type=int, required=True, help="処理開始インデックス")
    parser.add_argument("--end_index", type=int, required=True, help="処理終了インデックス")
    parser.add_argument("--output_file", type=str, help="出力ファイル名")
    parser.add_argument("--model_name", type=str, default="mistralai/mistral-7b-instruct", help="使用モデル")
    parser.add_argument("--log_dir", type=str, default="data/dna/logs", help="ログディレクトリ")
    parser.add_argument("--api_key", type=str, help="OpenRouter APIキー")
    parser.add_argument("--dataset_name", type=str, default="argo11/DNA_DPO_hh-rlhf", help="使用するデータセット名")
    
    args = parser.parse_args()
    
    # 出力ファイル名の設定
    if not args.output_file:
        args.output_file = f"think_tagged_{args.start_index}-{args.end_index-1}.jsonl"
    
    # APIキーの設定
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("エラー: OPENROUTER_API_KEYが設定されていません")
        return
    
    # ログ設定
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    logging.info(f"<think>タグ埋め込み処理開始: {args.start_index} - {args.end_index-1}")
    logging.info(f"モデル: {args.model_name}")
    logging.info(f"データセット: {args.dataset_name}")
    logging.info(f"出力ファイル: {args.output_file}")
    
    # データセット読み込み
    try:
        logging.info("データセット読み込み中...")
        dataset = load_dataset(args.dataset_name, split="train")
        logging.info(f"データセット読み込み完了: {len(dataset)} 件")
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        return
    
    # <think>タグ生成器初期化
    try:
        generator = ThinkTagGenerator(api_key, args.model_name)
        logging.info("<think>タグ生成器初期化完了")
    except Exception as e:
        logging.error(f"生成器初期化エラー: {e}")
        return
    
    # 既存IDチェック
    existing_ids = load_existing_ids(args.output_file)
    logging.info(f"既存処理済み件数: {len(existing_ids)}")
    
    # 処理実行
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(args.start_index, min(args.end_index, len(dataset))), desc="<think>タグ埋め込み処理"):
            item = dataset[i]
            item_id = item['id']
            
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
    
    logging.info(f"処理完了 - 処理件数: {processed_count}, エラー件数: {error_count}, スキップ件数: {skipped_count}")


if __name__ == "__main__":
    main()
