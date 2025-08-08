#!/usr/bin/env python3
"""PhysReasonデータセットをインポート"""

import json
import argparse
from pathlib import Path

def extract_category(theorems_list):
    """物理分野を抽出（Theoremフィールドから）"""
    # 簡易的なカテゴリ判定
    theorem_str = ' '.join(theorems_list).lower()
    if 'motion' in theorem_str or 'newton' in theorem_str or 'force' in theorem_str or 'momentum' in theorem_str:
        return 'MECHANICS'
    elif 'electric' in theorem_str or 'magnetic' in theorem_str or 'circuit' in theorem_str:
        return 'ELECTROMAGNETISM'
    elif 'wave' in theorem_str or 'optic' in theorem_str or 'interference' in theorem_str:
        return 'OPTICS'
    elif 'thermo' in theorem_str or 'heat' in theorem_str or 'temperature' in theorem_str:
        return 'THERMODYNAMICS'
    else:
        return 'PHYSICS'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default='./data/physreason/original/PhysReason_full')
    parser.add_argument('output', nargs='?', default='./data/physreason/preprocessed/dataset.jsonl')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem_dir in sorted(input_dir.glob('cal_problem_*')):
            problem_file = problem_dir / 'problem.json'
            if not problem_file.exists():
                continue
                
            with open(problem_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            
            # 画像付き問題をスキップ
            if data.get('question_image_list'):
                continue
            
            # 問題文の統合
            question_parts = [data['question_structure']['context']]
            sub_q_count = 0
            for i in range(1, 10):
                key = f'sub_question_{i}'
                if key in data['question_structure']:
                    sub_q_count += 1
                    question_parts.append(f'({i}) {data["question_structure"][key]}')
            question = '\n\n'.join(question_parts)
            
            # 解法の統合（steps_analysisの情報も含める）
            solution_parts = []
            steps_analysis = data.get('steps_analysis', {})
            
            for sub_q_key in sorted(data['explanation_steps'].keys()):
                sub_q_num = sub_q_key.replace('sub_question_', '')
                solution_parts.append(f'### Part ({sub_q_num})')
                
                for step_key in sorted(data['explanation_steps'][sub_q_key].keys()):
                    step_text = data['explanation_steps'][sub_q_key][step_key]
                    solution_parts.append(f'{step_key.capitalize().replace("_", " ")}: {step_text}')
                    
                    # steps_analysisから詳細情報を追加
                    if step_key in steps_analysis:
                        step_info = steps_analysis[step_key]
                        if 'physical_theorem' in step_info:
                            solution_parts.append(f'- Physical theorem: {step_info["physical_theorem"]}')
                        if 'result_quantity' in step_info and step_info['result_quantity']:
                            for result in step_info['result_quantity']:
                                if result.get('name'):
                                    solution_parts.append(f'- Result: {result["name"]}')
                                    if result.get('symbol'):
                                        solution_parts.append(f'  - Symbol: {result["symbol"]}')
                                    if result.get('equation'):
                                        solution_parts.append(f'  - Equation: {result["equation"]}')
                                    if result.get('value') and result.get('unit'):
                                        solution_parts.append(f'  - Value: {result["value"]} {result["unit"]}')
                    solution_parts.append('')
                    
                solution_parts.append('')
            original_solution = '\n'.join(solution_parts).strip()
            
            # 答えの統合
            if isinstance(data['answer'], list):
                answer_parts = []
                for i, ans in enumerate(data['answer'], 1):
                    answer_parts.append(f'({i}) {ans}')
                answer = '\n'.join(answer_parts)
            else:
                answer = data['answer']
            
            # カテゴリの抽出（Theoremフィールドから）
            theorems = data.get('Theorem', [])
            category = extract_category(theorems)
            
            # データの出力
            imported_data = {
                "id": count,
                "question": question,
                "original_solution": original_solution,
                "answer": answer,
                "metadata": {
                    "source": "PhysReason",
                    "source_file": "PhysReason_full",
                    "original_id": problem_dir.name,
                    "difficulty": data.get('difficulty'),
                    "category": category,
                    "num_subquestions": sub_q_count,
                    "theorems": theorems
                }
            }
            
            json.dump(imported_data, f, ensure_ascii=False)
            f.write('\n')
            count += 1
    
    print(f"Done! Processed: {count} items")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()