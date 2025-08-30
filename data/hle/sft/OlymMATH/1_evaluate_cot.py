#!/usr/bin/env python3
"""
CoT評価スクリプト

CoT（Chain of Thought）の品質を評価し、学習データとしての
適性をグレード（A〜D）で判定する。

評価は必須要件4項目と学習価値6観点で行い、
結果はmetadata.cot_historyに記録される。

使用方法：
python 1_evaluate_cot.py --dataset dataset.jsonl [--ids 1,5,17]
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
import time
from datetime import datetime

# .envファイルを読み込み
load_dotenv()

# モデル設定
MODEL = "deepseek/deepseek-r1-0528:free"
API_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.1  # 評価の一貫性を重視した低い値
MAX_TOKENS = 8000  # 詳細な評価結果を取得するための十分な長さ


class CoTEvaluator:
    """生成されたCoTの品質を評価し、学習データとしての採用可否を判定する"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def evaluate_with_llm(self, question: str, cot: str, answer: str) -> Dict:
        """LLMを使用してCoTの品質を評価し、詳細な評価結果を返す"""
        
        prompt = f"""Evaluate the following CoT (Chain of Thought) from the perspective of its value as training data for machine learning models.

[Problem]
{question}

[Expected Answer]
{answer}

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

3. **Correctness**
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

9. **Domain Insight** (0-10)
   - Explains meaning of equations
   - Provides interpretation of results
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
        "domain_insight": {{
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
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                result_text = response.choices[0].message.content

                # JSONをパース
                try:
                    result = json.loads(result_text)
                    result['evaluation_success'] = True
                    return result
                except json.JSONDecodeError as e:
                    if attempt == 2:
                        return {'evaluation_success': False, 'error': f'Invalid JSON response: {str(e)}'}

            except Exception as e:
                if attempt < 2:
                    wait_time = 2**attempt
                    print(f"LLM evaluation error: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {'evaluation_success': False, 'error': f'LLM evaluation error: {type(e).__name__}: {str(e)}'}

    def calculate_grade(self, evaluation: Dict) -> Dict:
        """評価結果を基にA〜Dのグレードを判定する"""
        if not evaluation.get('evaluation_success'):
            return {'grade': 'D', 'reason': evaluation.get('error', 'Evaluation failed'), 'score': 0}

        # 必須要件チェック
        mandatory = evaluation.get('mandatory_requirements', {})
        mandatory_passed = all([
            mandatory.get('independence', {}).get('passed', False),
            mandatory.get('logical_completeness', {}).get('passed', False),
            mandatory.get('correctness', {}).get('passed', False),
            mandatory.get('answer_reached', {}).get('passed', False),
        ])

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
            learning_value.get('domain_insight', {}).get('score', 0),
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

    def evaluate(self, question: str, cot: str, answer: str) -> Dict:
        """CoTの評価を実行し、グレードとレポートを生成する"""
        # CoTの抽出（<think>タグがある場合）
        if '<think>' in cot and '</think>' in cot:
            start = cot.find('<think>') + len('<think>')
            end = cot.find('</think>')
            cot_content = cot[start:end].strip()
        else:
            cot_content = cot

        # LLMによる評価
        evaluation = self.evaluate_with_llm(question, cot_content, answer)

        # グレード計算
        grade_info = self.calculate_grade(evaluation)

        return {
            'grade': grade_info['grade'],
            'score': grade_info['score'],
            'evaluation': evaluation,
            'grade_info': grade_info,
        }


def parse_ids(ids_str: str) -> List[str]:
    """カンマ区切りのID文字列をリストに変換"""
    return [id.strip() for id in ids_str.split(',')]


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
        description="汎用CoT評価スクリプト - 任意のデータセットのCoT品質を評価"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='評価対象のデータセットファイル（JSONL形式）'
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

    # 出力ファイルは常にデータセットファイルを上書き
    output_path = dataset_path

    # LLMクライアントの初期化
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in environment")
        print("Please set it in .env file or environment variable")
        return

    llm_client = openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)

    # 評価器の初期化
    evaluator = CoTEvaluator(llm_client)

    # データの読み込み
    print(f"Loading data from: {dataset_path}")
    all_data = load_jsonl(dataset_path)
    print(f"Loaded {len(all_data)} items")

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

    print(f"\nEvaluating {len(indices_to_evaluate)} CoTs")
    print("Focus: Learning data value (not style or expressions)")

    # 評価実行
    grade_counts = defaultdict(int)
    total_score = 0
    valid_scores = 0
    evaluated_count = 0
    skipped_count = 0

    for idx in tqdm(indices_to_evaluate, desc="Evaluating CoTs"):
        item = all_data[idx]
        item_id = item.get('id', f'index_{idx}')

        # 既に評価がある場合はスキップ
        # metadata.cot_history構造のチェック
        if item.get('metadata', {}).get('cot_history'):
            latest_cot = item['metadata']['cot_history'][-1]
            if latest_cot.get('evaluation') is not None:
                existing_grade = latest_cot['evaluation'].get('grade', 'Unknown')
                tqdm.write(f"Item {item_id}: Already evaluated (Grade {existing_grade}), skipping")
                skipped_count += 1
                continue

        # 必要なフィールドの確認
        required_fields = ['question', 'output', 'answer']
        if not all(key in item for key in required_fields):
            print(f"Warning: Skipping item {item_id} - missing required fields")
            missing = [f for f in required_fields if f not in item]
            print(f"  Missing: {missing}")
            continue

        # 評価実行
        try:
            result = evaluator.evaluate(
                question=item['question'],
                cot=item['output'],
                answer=item['answer']
            )

            # 評価が成功したかチェック
            if not result['evaluation'].get('evaluation_success', True):
                error_msg = result['evaluation'].get('error', 'Unknown error')
                print(f"Error evaluating item {item_id}: {error_msg}")
                continue

            # 評価結果の構造化
            evaluation_data = {
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
                    'domain_insight': result['evaluation'].get('learning_value_scores', {}).get('domain_insight', {}).get('score', 0),
                    'metacognitive': result['evaluation'].get('learning_value_scores', {}).get('metacognitive', {}).get('score', 0)
                },
                'strengths': result['evaluation'].get('strengths', []),
                'weaknesses': result['evaluation'].get('weaknesses', []),
                'improvement_suggestions': result['evaluation'].get('improvement_suggestions', []),
            }

            # 評価結果の保存
            # metadata.cot_history構造がない場合は作成（初回評価）
            if not item.get('metadata'):
                item['metadata'] = {}
            
            if not item['metadata'].get('cot_history'):
                # 初回評価：cot_history構造を作成
                item['metadata']['cot_history'] = [{
                    'timestamp': datetime.now().isoformat(),
                    'output': item['output'],
                    'evaluation': evaluation_data
                }]
            else:
                # 既存のcot_historyがある場合：最後の要素に評価を追加
                item['metadata']['cot_history'][-1]['evaluation'] = evaluation_data

            evaluated_count += 1

            # 統計情報の更新
            grade_counts[result['grade']] += 1
            total_score += result['score']
            valid_scores += 1

        except Exception as e:
            print(f"Error evaluating item {item_id}: {str(e)}")
            continue

    # ファイル保存
    save_jsonl(all_data, output_path)
    print(f"\n評価結果を保存しました: {output_path}")

    # 統計情報の表示
    print(f"\n=== 評価完了（学習データ価値基準） ===")
    print(f"評価実行数: {evaluated_count}")
    print(f"スキップ済み（既評価）: {skipped_count}")
    print(f"総データ数: {len(all_data)}")

    if evaluated_count > 0:
        for grade in ['A', 'B', 'C', 'D']:
            count = grade_counts[grade]
            percentage = count / evaluated_count * 100
            print(f"Grade {grade}: {count} ({percentage:.1f}%)")

        if valid_scores > 0:
            avg_score = total_score / valid_scores
            print(f"\n平均学習価値スコア: {avg_score:.2f}/10")

            # グレードA,B,Cの数を表示
            acceptable = grade_counts['A'] + grade_counts['B'] + grade_counts['C']
            print(f"学習データとして採用可能（Grade A/B/C）: {acceptable} ({acceptable/evaluated_count*100:.1f}%)")


if __name__ == "__main__":
    main()
