#!/usr/bin/env python3
"""
生成されたCoTを評価し、学習データとして採用可能かを判定する

必須要件（一つでも満たさない場合はグレードD）：
- 独立性：外部参照なしで問題を解いているか
- 論理的完全性：論理の飛躍なく結論まで到達しているか
- 正確性：物理法則や数学的手法が正しく適用されているか
- 解答到達：正しい答えに到達しているか

学習価値スコア（各10点満点の6観点）：
- 方法選択の説明：なぜその解法を選んだか説明しているか
- 段階的な導出：思考過程が明確なステップに分かれているか
- 検証と確認：結果の妥当性を確認しているか
- よくある誤りへの対処：間違いやすい点を指摘しているか
- 物理的洞察：式の物理的意味を説明しているか
- メタ認知的要素：思考の理由や行き詰まった時の対処を示しているか

グレード（6観点の平均スコアに基づく）：
- A: 優秀な学習データ（8.0点以上）
- B: 良好な学習データ（6.0点以上）
- C: 使用可能な学習データ（4.0点以上）
- D: 学習データとして不適切（必須要件未達成または4.0点未満）
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
import time
from datetime import datetime
import logging

# .envファイルを読み込み
load_dotenv()

# デフォルト設定
MODEL = "deepseek/deepseek-r1-0528:free"
API_BASE_URL = "https://openrouter.ai/api/v1"


class CoTEvaluator:
    """生成されたCoTの品質を評価し、学習データとしての採用可否を判定する"""

    def __init__(self, llm_client=None, debug=False):
        self.llm_client = llm_client
        self.debug = debug
        if debug:
            # デバッグログの設定
            logging.basicConfig(
                filename='evaluation_debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s'
            )

    def evaluate_with_llm(self, question: str, solution: str, cot: str, answer: str) -> Dict:
        """LLMを使用してCoTの品質を評価し、詳細な評価結果を返す"""
        # Learning data value-focused evaluation prompt
        prompt = f"""Evaluate the following CoT (Chain of Thought) from the perspective of its value as training data for machine learning models.

[Problem]
{question}

[Expected Answer]
{answer}

[Reference: Textbook Solution]
{solution}

[CoT to Evaluate]
{cot}

Important: This CoT is training data for models to learn "the thinking process from problem to solution".
Evaluate logical completeness and educational value, not exploratory expressions or style.

## Mandatory Requirements (If any fails, overall grade is D)

1. **Independence (No External References)**
   - No references to external solutions like "formal solution", "they say", etc.
   - Appears to solve the problem independently
   - Judgment: Pass/Fail
   - Provide reason and specific examples

2. **Logical Completeness**
   - Connected from premise to conclusion without logical leaps
   - Clear rationale for each step
   - No omission of important reasoning steps
   - Judgment: Pass/Fail
   - Provide reason

3. **Physical/Mathematical Correctness**
   - Physics laws and mathematical methods applied correctly
   - No major conceptual errors
   - Calculation errors OK if approach is correct
   - Judgment: Pass/Fail
   - Provide reason

4. **Reaching the Answer**
   - Arrives at correct (or reasonable) answer
   - Doesn't give up midway
   - Judgment: Pass/Fail
   - Provide reason

## Learning Value Requirements (Score 0-10 for training data value)

5. **Method Selection Explanation** (0-10)
   - Explains why this solution method was chosen
   - Compares with other methods or states advantages
   - Learner can understand "when to use this method"

6. **Step-by-Step Derivation** (0-10)
   - Thinking process divided into clear steps
   - Clear what's being done at each step
   - Detailed enough for learners to track and reproduce

7. **Verification and Checking** (0-10)
   - Validates result reasonableness
   - Uses alternative methods, dimensional analysis, limit checks, etc.
   - Teaches "how to verify the answer is correct"

8. **Handling Common Mistakes** (0-10)
   - Points out error-prone areas
   - Explicitly states points requiring attention
   - Shows how to avoid traps beginners fall into

9. **Physical Insight** (0-10)
   - Explains physical meaning of equations
   - Provides physical interpretation of results
   - Demonstrates essential understanding of the problem

10. **Metacognitive Elements** (0-10)
    - Explicitly states reasoning for decisions in thought process
    - Shows how to handle getting stuck
    - Explains not just "how to think" but "why think this way"

## Overall Assessment

Based on above evaluation:
- 3 particularly strong points as training data
- 3 deficient points as training data
- 2 specific improvement suggestions for better training data

## Response Format

{{
    "mandatory_requirements": {{
        "independence": {{
            "passed": true/false,
            "reason": "reason",
            "examples": ["specific examples if any"]
        }},
        "logical_completeness": {{
            "passed": true/false,
            "reason": "reason"
        }},
        "correctness": {{
            "passed": true/false,
            "reason": "reason"
        }},
        "answer_reached": {{
            "passed": true/false,
            "reason": "reason"
        }}
    }},
    "learning_value_scores": {{
        "method_explanation": {{
            "score": 0-10,
            "reason": "reason"
        }},
        "step_by_step": {{
            "score": 0-10,
            "reason": "reason"
        }},
        "verification": {{
            "score": 0-10,
            "reason": "reason"
        }},
        "common_mistakes": {{
            "score": 0-10,
            "reason": "reason"
        }},
        "physical_insight": {{
            "score": 0-10,
            "reason": "reason"
        }},
        "metacognitive": {{
            "score": 0-10,
            "reason": "reason"
        }}
    }},
    "strengths": ["strength as training data 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness as training data 1", "weakness 2", "weakness 3"],
    "improvement_suggestions": ["improvement 1", "improvement 2"]
}}"""

        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in evaluating the quality of training data for machine learning. Focus on logical completeness and educational value, not style or expressions.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=8000,
                )

                result_text = response.choices[0].message.content

                # デバッグ: 生のレスポンスをログに記録
                if self.debug:
                    logging.debug(f"\n{'='*80}")
                    logging.debug(f"Question ID: {question[:50]}...")
                    logging.debug(f"Raw LLM Response:\n{result_text}")
                    logging.debug(f"{'='*80}\n")

                # JSONをパース
                try:
                    result = json.loads(result_text)
                    result['evaluation_success'] = True
                    return result
                except json.JSONDecodeError as e:
                    if self.debug:
                        logging.error(f"JSON Parse Error: {str(e)}")
                        logging.error(f"Failed to parse: {result_text[:500]}...")

                    if attempt == 2:
                        return {'evaluation_success': False, 'error': f'Invalid JSON response: {str(e)}'}

            except Exception as e:
                if self.debug:
                    logging.error(f"LLM API Error: {type(e).__name__}: {str(e)}")

                if attempt < 2:
                    wait_time = 2**attempt
                    print(f"LLM evaluation error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {'evaluation_success': False, 'error': f'LLM evaluation error: {type(e).__name__}: {str(e)}'}

    def calculate_grade(self, evaluation: Dict) -> Dict:
        """評価結果を基にA〜Dのグレードを判定する"""
        if not evaluation.get('evaluation_success'):
            return {'grade': 'D', 'reason': evaluation.get('error', 'Evaluation failed'), 'score': 0}  # エラー時もD評価

        # 必須要件チェック
        mandatory = evaluation.get('mandatory_requirements', {})
        mandatory_passed = all(
            [
                mandatory.get('independence', {}).get('passed', False),
                mandatory.get('logical_completeness', {}).get('passed', False),
                mandatory.get('correctness', {}).get('passed', False),
                mandatory.get('answer_reached', {}).get('passed', False),
            ]
        )

        if not mandatory_passed:
            # どの必須要件で失敗したか詳細を記録
            failed_requirements = []
            if not mandatory.get('independence', {}).get('passed', False):
                failed_requirements.append('外部参照あり')
            if not mandatory.get('logical_completeness', {}).get('passed', False):
                failed_requirements.append('論理的に不完全')
            if not mandatory.get('correctness', {}).get('passed', False):
                failed_requirements.append('重大な誤りあり')
            if not mandatory.get('answer_reached', {}).get('passed', False):
                failed_requirements.append('解答に到達せず')

            return {
                'grade': 'D',
                'reason': 'Mandatory requirements not met',
                'failed_requirements': failed_requirements,
                'score': 0,
            }

        # 学習価値スコアの計算
        learning_value = evaluation.get('learning_value_scores', {})
        scores = [
            learning_value.get('method_explanation', {}).get('score', 0),
            learning_value.get('step_by_step', {}).get('score', 0),
            learning_value.get('verification', {}).get('score', 0),
            learning_value.get('common_mistakes', {}).get('score', 0),
            learning_value.get('physical_insight', {}).get('score', 0),
            learning_value.get('metacognitive', {}).get('score', 0),
        ]

        average_score = sum(scores) / len(scores) if scores else 0

        # グレード判定（学習データとしての価値基準）
        if average_score >= 8:
            grade = 'A'  # 優秀な学習データ
        elif average_score >= 6:
            grade = 'B'  # 良好な学習データ
        elif average_score >= 4:
            grade = 'C'  # 使用可能な学習データ
        else:
            grade = 'D'  # 学習データとして不適切

        return {
            'grade': grade,
            'score': average_score,
            'learning_value_scores': scores,
            'reason': f'Average learning value score: {average_score:.1f}',
        }

    def evaluate(self, question: str, solution: str, cot: str, answer: str) -> Dict:
        """CoTの評価を実行し、グレードとレポートを生成する"""
        # CoTの抽出（<think>タグがある場合）
        if '<think>' in cot and '</think>' in cot:
            start = cot.find('<think>') + len('<think>')
            end = cot.find('</think>')
            cot_content = cot[start:end].strip()
        else:
            cot_content = cot

        # LLMによる評価
        evaluation = self.evaluate_with_llm(question, solution, cot_content, answer)

        # グレード計算
        grade_info = self.calculate_grade(evaluation)

        return {
            'grade': grade_info['grade'],
            'score': grade_info['score'],
            'evaluation': evaluation,
            'grade_info': grade_info,
        }


def parse_ids(ids_str: str) -> List[str]:
    """カンマ区切りのID文字列をリストに変換

    Args:
        ids_str: カンマ区切りのID文字列（例: "1,5,17"）

    Returns:
        IDのリスト
    """
    return [id.strip() for id in ids_str.split(',')]


def main():
    parser = argparse.ArgumentParser(description="生成されたCoTの品質を評価する")
    parser.add_argument('dataset', help='データセット名（phybench, physreason）')
    parser.add_argument('--ids', type=str, help='処理するIDをカンマ区切りで指定（例: 1,5,17）')
    parser.add_argument('--debug', action='store_true', help='デバッグログを出力')
    parser.add_argument('--force', action='store_true', help='既に評価済みのアイテムも強制的に再評価')
    args = parser.parse_args()

    # データセット名に基づいてパスを設定
    if args.dataset not in ['phybench', 'physreason']:
        print(f"Error: Unknown dataset '{args.dataset}'. Use 'phybench' or 'physreason'.")
        return

    # 最新の生成ファイルを取得
    generated_dir = Path(f'data/{args.dataset}/generated')
    generated_files = sorted(generated_dir.glob('generated_cot_*.jsonl'))
    if not generated_files:
        print(f"Error: No generated files found in {generated_dir}")
        return

    input_path = generated_files[-1]

    # LLMクライアントの初期化
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    llm_client = openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # 評価器の初期化
    evaluator = CoTEvaluator(llm_client, debug=args.debug)

    if args.debug:
        print("Debug mode enabled. Logs will be written to evaluation_debug.log")

    # 生成データの読み込み
    with open(input_path, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]

    # 評価対象のインデックスを記録
    indices_to_evaluate = []
    if args.ids:
        target_ids = parse_ids(args.ids)
        target_ids_set = set(target_ids)

        for i, item in enumerate(all_data):
            item_id = str(item.get('id', ''))
            if item_id in target_ids_set:
                indices_to_evaluate.append(i)

        # 見つからなかったIDを警告
        found_ids = {str(all_data[i].get('id', '')) for i in indices_to_evaluate}
        missing_ids = target_ids_set - found_ids
        if missing_ids:
            print(f"Warning: IDs not found: {', '.join(sorted(missing_ids))}")

        print(f"Filtering by IDs: {', '.join(target_ids)}")
    else:
        # ID指定がない場合は全て評価対象
        indices_to_evaluate = list(range(len(all_data)))

    print(f"\nEvaluating {len(indices_to_evaluate)} CoTs from {input_path}")
    print("Focus: Learning data value (not style or expressions)")

    # 評価実行
    grade_counts = defaultdict(int)
    total_score = 0
    valid_scores = 0
    evaluated_count = 0
    skipped_count = 0

    for idx in tqdm(indices_to_evaluate, desc="Evaluating CoTs"):
        item = all_data[idx]
        item_id = item.get('id')

        # 最新のCoTに既に評価がある場合はスキップ（--forceオプションがない場合）
        if not args.force and item.get('metadata', {}).get('cot_history'):
            latest_cot = item['metadata']['cot_history'][-1]
            if latest_cot.get('evaluation') is not None:
                existing_grade = latest_cot['evaluation'].get('grade', 'Unknown')
                # tqdmの出力を乱さないようにpostfixで表示
                tqdm.write(f"Item {item_id}: Already evaluated (Grade {existing_grade}), skipping")
                skipped_count += 1
                continue

        # metadata から original_solution を取得
        if not item.get('metadata', {}).get('original_solution'):
            print(f"Warning: Skipping item {item_id} - no original_solution in metadata")
            continue
        original_solution = item['metadata']['original_solution']

        # 必要なフィールドの確認
        required_fields = ['question', 'output', 'answer']
        if not all(key in item for key in required_fields):
            print(f"Warning: Skipping item {item_id} - missing required fields")
            continue

        # 評価実行
        try:
            result = evaluator.evaluate(
                question=item['question'], solution=original_solution, cot=item['output'], answer=item['answer']
            )

            # 評価が成功したかチェック
            if not result['evaluation'].get('evaluation_success', True):
                # APIエラーの場合は評価をスキップ
                error_msg = result['evaluation'].get('error', 'Unknown error')
                print(f"Error evaluating item {item_id}: {error_msg}")
                continue  # evaluation: null のまま残す

            # メタデータとして評価結果を追加
            if 'metadata' not in item:
                item['metadata'] = {}

            # cot_historyが存在しない場合はエラー
            if not item['metadata'].get('cot_history') or len(item['metadata']['cot_history']) == 0:
                print(f"Error: item {item_id} has no cot_history")
                continue

            # cot_historyの最新エントリに評価結果を追加
            item['metadata']['cot_history'][-1]['evaluation'] = {
                'timestamp': datetime.now().isoformat(),
                'grade': result['grade'],
                'score': result['score'],
                'passed_requirements': {
                    'independence': result['evaluation'].get('mandatory_requirements', {}).get('independence', {}).get('passed', False),
                    'logical_completeness': result['evaluation'].get('mandatory_requirements', {}).get('logical_completeness', {}).get('passed', False),
                    'correctness': result['evaluation'].get('mandatory_requirements', {}).get('correctness', {}).get('passed', False),
                    'answer_reached': result['evaluation'].get('mandatory_requirements', {}).get('answer_reached', {}).get('passed', False)
                },
                'learning_value_scores': {
                    'method_explanation': result['evaluation'].get('learning_value_scores', {}).get('method_explanation', {}).get('score', 0),
                    'step_by_step': result['evaluation'].get('learning_value_scores', {}).get('step_by_step', {}).get('score', 0),
                    'verification': result['evaluation'].get('learning_value_scores', {}).get('verification', {}).get('score', 0),
                    'common_mistakes': result['evaluation'].get('learning_value_scores', {}).get('common_mistakes', {}).get('score', 0),
                    'physical_insight': result['evaluation'].get('learning_value_scores', {}).get('physical_insight', {}).get('score', 0),
                    'metacognitive': result['evaluation'].get('learning_value_scores', {}).get('metacognitive', {}).get('score', 0)
                },
                'strengths': result['evaluation'].get('strengths', []),
                'weaknesses': result['evaluation'].get('weaknesses', []),
                'improvement_suggestions': result['evaluation'].get('improvement_suggestions', []),
            }

            evaluated_count += 1

            # 統計情報の更新
            grade_counts[result['grade']] += 1
            # 全ての結果をスコア計算に含める
            total_score += result['score']
            valid_scores += 1

            # 評価完了後、即座にファイルに書き込む
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in all_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

        except Exception as e:
            print(f"Error evaluating item {item.get('id', 'unknown')}: {str(e)}")
            # エラーの場合は評価をスキップして evaluation: null のまま残す
            continue

    # 統計情報の表示
    print(f"\n=== 評価完了（学習データ価値基準） ===")
    print(f"評価実行数: {evaluated_count}")
    print(f"スキップ済み（既評価）: {skipped_count}")
    print(f"総データ数: {len(all_data)}")

    for grade in ['A', 'B', 'C', 'D']:
        count = grade_counts[grade]
        percentage = count / evaluated_count * 100 if evaluated_count > 0 else 0
        print(f"Grade {grade}: {count} ({percentage:.1f}%)")

    if valid_scores > 0:
        avg_score = total_score / valid_scores
        print(f"\n平均学習価値スコア: {avg_score:.2f}/10")

        # グレードA,B,Cの数を表示
        acceptable = sum(
            1
            for item in all_data
            if item.get('metadata', {}).get('cot_history')
            and len(item['metadata']['cot_history']) > 0
            and item['metadata']['cot_history'][-1].get('evaluation')
            and item['metadata']['cot_history'][-1]['evaluation'].get('grade') in ['A', 'B', 'C']
        )
        print(f"学習データとして採用可能（Grade A/B/C）: {acceptable} ({acceptable/len(all_data)*100:.1f}%)")

    print(f"\n評価結果を元ファイルに保存しました: {input_path}")


if __name__ == "__main__":
    main()
