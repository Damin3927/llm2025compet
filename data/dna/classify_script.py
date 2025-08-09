#!/usr/bin/env python3
"""
有害性分類スクリプト (MVP版)
neko-llm/DNA_DPO_hh-rlhf データセットの3段階有害性分類
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

class HarmClassifier:
    """有害性分類を行うメインクラス"""

    def __init__(self, api_key: str, model_name: str, hierarchy_file: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name
        self.hierarchy = self._load_hierarchy(hierarchy_file)

    def _load_hierarchy(self, hierarchy_file: str) -> Dict[str, Any]:
        """分類階層を読み込み"""
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_risk_areas(self) -> List[str]:
        """最上位のrisk_areaリストを取得"""
        return list(self.hierarchy.keys())

    def _get_types_of_harm(self, risk_area: str) -> List[str]:
        """指定されたrisk_areaのtypes_of_harmリストを取得"""
        return list(self.hierarchy[risk_area]["types_of_harm"].keys())

    def _get_specific_harms(self, risk_area: str, type_of_harm: str) -> List[str]:
        """指定されたtype_of_harmのspecific_harmsリストを取得"""
        return list(self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"].keys())

    def _call_llm(self, prompt: str) -> Optional[str]:
        """LLM APIを呼び出し"""
        # --- ▼▼▼ 変更箇所1: systemプロンプトの強化 ▼▼▼ ---
        system_prompt = """
あなたは、テキスト分類のみを行う専門家です。あなたの唯一のタスクは、与えられた選択肢の中から最も適切なものを1つだけ選び、その選択肢の文字列そのものを返すことです。
質問の内容自体に反応したり、回答を拒否したり、追加の説明を生成したりしてはいけません。必ず選択肢の中から一つを選んでください。
"""
        # --- ▲▲▲ 変更箇所1: ここまで ▲▲▲ ---
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                # --- ▼▼▼ 変更箇所2: temperatureを0に設定 ▼▼▼ ---
                temperature=0.0
                # --- ▲▲▲ 変更箇所2: ここまで ▲▲▲ ---
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API呼び出しエラー: {e}")
            return None

    def _classify_step(self, question: str, choices: List[str], step_name: str) -> Optional[str]:
        """単一の分類ステップを実行"""
        choices_text = "\n".join(f"- {choice}" for choice in choices)
        # --- ▼▼▼ 変更箇所3: userプロンプトの強化 ▼▼▼ ---
        prompt = f"""
以下の「ユーザーからの質問」を分析し、指定された「選択肢」の中から最も関連性の高い{step_name}を1つだけ選んでください。
質問自体には応答せず、分類タスクのみを実行してください。

---
ユーザーからの質問:
"{question}"
---
選択肢:
{choices_text}
---

応答は、選択した{step_name}の文字列のみとしてください。
"""
        # --- ▲▲▲ 変更箇所3: ここまで ▲▲▲ ---

        logging.info(f"Step: {step_name}")
        # ログが見やすいように改行を追加
        logging.info(f"Prompt: \n{prompt}")

        result = self._call_llm(prompt)
        logging.info(f"Response: {result}")

        if result and result in choices:
            return result

        logging.warning(f"無効な応答: {result}, 選択肢: {choices}")
        return None

    def classify_question(self, question: str) -> Dict[str, str]:
        """質問を3段階で分類"""

        # Step 1: risk_area
        risk_areas = self._get_risk_areas()
        risk_area = self._classify_step(question, risk_areas, "リスク領域")

        if not risk_area:
            return {
                "risk_area": "CLASSIFICATION_ERROR",
                "type_of_harm": "SKIPPED",
                "specific_harm": "SKIPPED"
            }

        # Step 2: type_of_harm
        types_of_harm = self._get_types_of_harm(risk_area)
        type_of_harm = self._classify_step(question, types_of_harm, "害の種類")

        if not type_of_harm:
            return {
                "risk_area": risk_area,
                "type_of_harm": "CLASSIFICATION_ERROR",
                "specific_harm": "SKIPPED"
            }

        # Step 3: specific_harm
        specific_harms = self._get_specific_harms(risk_area, type_of_harm)
        specific_harm_key = self._classify_step(question, specific_harms, "具体的な害")

        if not specific_harm_key:
            return {
                "risk_area": risk_area,
                "type_of_harm": type_of_harm,
                "specific_harm": "CLASSIFICATION_ERROR"
            }

        # specific_harmの説明を取得して連結
        specific_harm_description = self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"][specific_harm_key]
        specific_harm_full = f"{specific_harm_key}: {specific_harm_description}"

        return {
            "risk_area": risk_area,
            "type_of_harm": type_of_harm,
            "specific_harm": specific_harm_full
        }

def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}_{start_index}-{end_index-1}.log")
    
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
    parser = argparse.ArgumentParser(description="有害性分類スクリプト")
    parser.add_argument("--start_index", type=int, required=True, help="処理開始インデックス")
    parser.add_argument("--end_index", type=int, required=True, help="処理終了インデックス")
    parser.add_argument("--output_file", type=str, help="出力ファイル名")
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3-8b-instruct", help="使用モデル")
    parser.add_argument("--hierarchy_file", type=str, default="data/dna/dna_hierarchy.json", help="階層ファイル")
    parser.add_argument("--log_dir", type=str, default="data/dna/logs", help="ログディレクトリ")
    parser.add_argument("--api_key", type=str, help="OpenRouter APIキー")
    
    args = parser.parse_args()
    
    # 出力ファイル名の設定
    if not args.output_file:
        args.output_file = f"classified_{args.start_index}-{args.end_index-1}.jsonl"
    
    # APIキーの設定
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("エラー: OPENROUTER_API_KEYが設定されていません")
        return
    
    # ログ設定
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    logging.info(f"分類処理開始: {args.start_index} - {args.end_index-1}")
    logging.info(f"モデル: {args.model_name}")
    logging.info(f"出力ファイル: {args.output_file}")
    
    # データセット読み込み
    try:
        logging.info("データセット読み込み中...")
        dataset = load_dataset("neko-llm/DNA_DPO_hh-rlhf", split="train")
        logging.info(f"データセット読み込み完了: {len(dataset)} 件")
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        return
    
    # 分類器初期化
    try:
        classifier = HarmClassifier(api_key, args.model_name, args.hierarchy_file)
        logging.info("分類器初期化完了")
    except Exception as e:
        logging.error(f"分類器初期化エラー: {e}")
        return
    
    # 既存IDチェック
    existing_ids = load_existing_ids(args.output_file)
    logging.info(f"既存処理済み件数: {len(existing_ids)}")
    
    # 処理実行
    processed_count = 0
    error_count = 0
    
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for i in tqdm(range(args.start_index, min(args.end_index, len(dataset))), desc="分類処理"):
            item = dataset[i]
            item_id = item['id']
            
            # スキップチェック
            if item_id in existing_ids:
                continue
            
            logging.info(f"処理中 ID: {item_id}")
            
            # 分類実行
            classification = classifier.classify_question(item['question'])
            
            # 結果作成
            result = {
                "id": item_id,
                "question": item['question'],
                "preferred_output": item['preferred_output'], 
                "non_preferred_output": item['non_preferred_output'],
                **classification
            }
            
            # ファイル出力
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            processed_count += 1
            if "CLASSIFICATION_ERROR" in classification.values():
                error_count += 1
            
            logging.info(f"分類結果: {classification}")
    
    logging.info(f"処理完了 - 処理件数: {processed_count}, エラー件数: {error_count}")


if __name__ == "__main__":
    main()
