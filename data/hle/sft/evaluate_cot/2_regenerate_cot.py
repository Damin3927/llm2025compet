#!/usr/bin/env python3
"""
CoT再生成スクリプト

低品質なCoTを評価結果の改善提案に基づいて再生成する。
強みを維持しながら弱点を改善し、より高品質な学習データを生成。

デフォルトでグレードC以下を対象とし、--gradeオプションで変更可能。

使用方法：
python 2_regenerate_cot.py --dataset dataset.jsonl [--grade B] [--ids 1,5,17]
"""

import json
import os
import argparse
from pathlib import Path
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import time
from typing import Dict, Any, List
from datetime import datetime

# .envファイルを読み込み
load_dotenv()

# モデル設定
MODEL = "deepseek/deepseek-r1-0528:free"
API_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.2
MAX_TOKENS = 40000
RETRY_ATTEMPTS = 3

# グレード順序の定義（高い方が良い）
GRADE_ORDER = {'A': 4, 'B': 3, 'C': 2, 'D': 1}


def should_regenerate(grade: str, threshold_grade: str) -> bool:
    """指定グレード以下かどうか判定
    
    Args:
        grade: アイテムのグレード
        threshold_grade: 閾値グレード
    
    Returns:
        再生成すべきかどうか
    """
    return GRADE_ORDER.get(grade, 0) <= GRADE_ORDER.get(threshold_grade, 0)


def parse_ids(ids_str: str) -> List[str]:
    """カンマ区切りのID文字列をリストに変換
    
    Args:
        ids_str: カンマ区切りのID文字列（例: "1,5,17"）
    
    Returns:
        IDのリスト
    """
    return [id.strip() for id in ids_str.split(',')]


def regenerate_cot(
    client,
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any],
) -> str:
    """改善提案に基づくCoT再生成
    
    Args:
        client: OpenAI APIクライアント
        question: 問題文
        answer: 答え
        previous_cot: 前回生成したCoT
        evaluation_details: 評価の詳細情報
    
    Returns:
        再生成されたCoT（thinkタグなし）
    """
    # System prompt
    system_prompt = """You are a problem-solving expert who learns from feedback and has the ability to improve solutions.

Improvement guidelines:
1. Focus on improving weaknesses identified in the evaluation
2. Maintain aspects identified as strengths
3. Ensure logical completeness and independence
4. Strengthen step-by-step derivation and verification
5. Explicitly address common mistakes
6. Add metacognitive elements (e.g., why you chose certain methods)

Important notes:
- Generate as a completely independent solution (avoid references to previous attempts)
- Avoid external references (e.g., "according to textbooks", "generally speaking")
- Divide the thought process into clear steps
- Always verify the validity of results
- Always reach the final answer"""

    # 評価詳細から情報を抽出
    previous_grade = evaluation_details.get('grade', 'Unknown')
    strengths = evaluation_details.get('strengths', [])
    weaknesses = evaluation_details.get('weaknesses', [])
    improvement_suggestions = evaluation_details.get('improvement_suggestions', [])
    learning_value_scores = evaluation_details.get('learning_value_scores', {})

    # Build formatted text
    strengths_text = "\n".join(f"• {s}" for s in strengths) if strengths else "• None identified"
    weaknesses_text = "\n".join(f"• {w}" for w in weaknesses) if weaknesses else "• None identified"
    suggestions_text = "\n".join(f"• {s}" for s in improvement_suggestions) if improvement_suggestions else "• None provided"

    # Build detailed scores
    scores_lines = []
    if learning_value_scores:
        scores_lines.append(f"• Method explanation: {learning_value_scores.get('method_explanation', 0)}/10")
        scores_lines.append(f"• Step-by-step derivation: {learning_value_scores.get('step_by_step', 0)}/10")
        scores_lines.append(f"• Verification and checking: {learning_value_scores.get('verification', 0)}/10")
        scores_lines.append(f"• Handling common mistakes: {learning_value_scores.get('common_mistakes', 0)}/10")
        scores_lines.append(f"• Domain-specific insight: {learning_value_scores.get('domain_insight', 0)}/10")
        scores_lines.append(f"• Metacognitive elements: {learning_value_scores.get('metacognitive', 0)}/10")
    scores_text = "\n".join(scores_lines) if scores_lines else "• No scores available"

    # 前回のCoTの抽出（<think>タグがある場合）
    if '<think>' in previous_cot and '</think>' in previous_cot:
        start = previous_cot.find('<think>') + len('<think>')
        end = previous_cot.find('</think>')
        previous_cot_content = previous_cot[start:end].strip()
    else:
        previous_cot_content = previous_cot

    # User prompt
    user_prompt = f"""Problem:
{question}

Expected answer:
{answer}

===============================================================================
Previous solution:
{previous_cot_content}

===============================================================================
Evaluation results:

Grade: {previous_grade}

Strengths (to maintain):
{strengths_text}

Weaknesses (to improve):
{weaknesses_text}

Detailed scores:
{scores_text}

Improvement suggestions:
{suggestions_text}

===============================================================================

Based on the feedback above, generate an improved solution.
Maintain the strengths while focusing on improving the weaknesses and low-scoring areas."""

    # API呼び出しとリトライ処理
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                wait_time = 2 ** attempt  # 1, 2, 4秒
                print(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error after {RETRY_ATTEMPTS} attempts: {e}")
                return None


def load_jsonl(file_path: Path) -> List[Dict]:
    """JSONLファイルを読み込む"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """JSONLファイルに保存する"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description="低品質なCoTを改善提案に基づいて再生成する"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='データセットファイル（JSONL形式）'
    )
    parser.add_argument(
        '--grade',
        type=str,
        default='C',
        choices=['A', 'B', 'C', 'D'],
        help='再生成する最大グレード（デフォルト: C）'
    )
    parser.add_argument(
        '--ids',
        type=str,
        help='処理するIDをカンマ区切りで指定（例: 1,5,17）'
    )
    args = parser.parse_args()

    # データセットファイルの確認
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset file not found: {dataset_path}")
        return

    # API設定
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment")
        print("Please set it in .env file or environment variable")
        return

    client = openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # データ読み込み
    print(f"Loading data from: {dataset_path}")
    all_data = load_jsonl(dataset_path)
    print(f"Loaded {len(all_data)} items")

    # 再生成対象のフィルタリング
    items_to_regenerate = []
    skipped_unevaluated = []

    for item in all_data:
        # metadata.cot_historyの確認
        if not item.get('metadata', {}).get('cot_history'):
            item_id = str(item.get('id', 'unknown'))
            skipped_unevaluated.append(item_id)
            continue

        cot_history = item['metadata']['cot_history']
        latest_history = cot_history[-1]
        item_id = str(item.get('id', ''))

        # 最新のCoTが未評価の場合はスキップ
        if latest_history.get('evaluation') is None:
            skipped_unevaluated.append(item_id)
            continue

        evaluation = latest_history['evaluation']
        grade = evaluation.get('grade', 'D')

        # ID指定がある場合
        if args.ids:
            target_ids = parse_ids(args.ids)
            if item_id in target_ids:
                items_to_regenerate.append(item)
        # ID指定がない場合は指定グレード以下を対象
        elif should_regenerate(grade, args.grade):
            items_to_regenerate.append(item)

    # スキップしたアイテムの報告
    if skipped_unevaluated:
        print(f"\nSkipped {len(skipped_unevaluated)} items without evaluation:")
        sample = skipped_unevaluated[:10]
        print(f"  IDs: {', '.join(sample)}{'...' if len(skipped_unevaluated) > 10 else ''}")

    if not items_to_regenerate:
        if args.ids:
            print("\nNo items found with specified IDs or they don't have evaluations")
        else:
            print(f"\nNo items to regenerate (no Grade {args.grade} or lower items found)")
        return

    print(f"\nFound {len(items_to_regenerate)} items to regenerate")

    # ID指定時または詳細表示
    if args.ids or len(items_to_regenerate) <= 10:
        print("\nItems to regenerate:")
        for item in items_to_regenerate[:10]:
            item_id = item.get('id', 'unknown')
            grade = item['metadata']['cot_history'][-1]['evaluation'].get('grade', 'Unknown')
            print(f"  ID {item_id}: Grade {grade}")
        if len(items_to_regenerate) > 10:
            print(f"  ... and {len(items_to_regenerate) - 10} more")

    # 再生成実行
    success_count = 0
    failed_items = []

    for item in tqdm(items_to_regenerate, desc="Regenerating CoTs"):
        item_id = item.get('id', 'unknown')

        # 最新の評価結果と前回のCoTを取得
        latest_history = item['metadata']['cot_history'][-1]
        latest_evaluation = latest_history['evaluation']
        previous_cot = latest_history.get('output', '')

        # 評価詳細を構築
        evaluation_details = {
            'grade': latest_evaluation.get('grade', 'D'),
            'strengths': latest_evaluation.get('strengths', []),
            'weaknesses': latest_evaluation.get('weaknesses', []),
            'improvement_suggestions': latest_evaluation.get('improvement_suggestions', []),
            'learning_value_scores': latest_evaluation.get('learning_value_scores', {}),
        }

        # 必要なフィールドの確認
        if not item.get('question') or not item.get('answer'):
            print(f"\nWarning: Item {item_id} missing required fields, skipping")
            failed_items.append(item_id)
            continue

        # CoT再生成
        output = regenerate_cot(
            client,
            item["question"],
            item["answer"],
            previous_cot,
            evaluation_details
        )

        if output:
            # 最終形式（CoT + 答え）
            final_output = f"<think>{output}</think>{item['answer']}"

            # 新しい履歴エントリを追加
            new_history_entry = {
                "timestamp": datetime.now().isoformat(),
                "output": final_output,
                "evaluation": None,  # 未評価
            }

            # cot_historyに追加
            item['metadata']['cot_history'].append(new_history_entry)

            # outputフィールドを最新のものに更新
            item['output'] = final_output

            success_count += 1
        else:
            print(f"\nWarning: Failed to regenerate CoT for item {item_id}")
            failed_items.append(item_id)

    # ファイル保存
    save_jsonl(all_data, dataset_path)

    # 結果サマリー
    print(f"\n=== 再生成完了 ===")
    print(f"成功: {success_count}/{len(items_to_regenerate)} items")
    if failed_items:
        print(f"失敗: {len(failed_items)} items")
        print(f"  Failed IDs: {', '.join(failed_items[:10])}{'...' if len(failed_items) > 10 else ''}")
    print(f"結果を保存しました: {dataset_path}")
    
    if success_count > 0:
        print(f"\n次のステップ:")
        print(f"  python evaluate_cot.py --dataset {dataset_path}")


if __name__ == "__main__":
    main()