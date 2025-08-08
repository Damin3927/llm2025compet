#!/usr/bin/env python3
"""
低品質なCoTを改善提案に基づいて再生成する

処理対象：
- ID指定なし：グレードC/DのCoTを自動的に再生成
- ID指定あり：指定されたIDのCoTを強制的に再生成（グレード問わず）

再生成方法：
- 前回の評価で得られた改善提案をプロンプトに含める
- 同じプロンプトバージョンで再度生成を実行
- 生成履歴（cot_history）に新しいエントリとして追加
"""

import json
import os
import argparse
from pathlib import Path
import openai
from tqdm import tqdm
from dotenv import load_dotenv
import time
import yaml
from typing import Dict, Any, Optional, List
from datetime import datetime

# .envファイルを読み込み
load_dotenv()

# デフォルト設定
DEFAULT_MODEL = "deepseek/deepseek-r1-0528:free"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_config(version: str, prompt_type: str = "generate") -> Dict[str, Any]:
    """プロンプト設定をYAMLから読み込む

    Args:
        version: プロンプトバージョン（必須）
        prompt_type: プロンプトのタイプ（'generate' or 'regenerate'、デフォルト: 'generate'）

    Returns:
        プロンプト設定の辞書
    """
    if not version:
        raise ValueError("Prompt version must be specified")

    # レジストリを読み込み
    registry_path = PROMPTS_DIR / "registry.yaml"
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)

    # version_infoをversionとtypeの両方で取得
    version_info = next(
        (v for v in registry['versions'] if v['version'] == version and v.get('type', 'generate') == prompt_type), None
    )

    if not version_info:
        # 利用可能なバージョンとタイプの組み合わせを表示
        available = [(v['version'], v.get('type', 'generate')) for v in registry['versions']]
        raise ValueError(f"Version {version} with type {prompt_type} not found. Available: {available}")

    # プロンプトファイルを読み込み
    with open(PROMPTS_DIR / version_info['file'], 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"\nLoaded prompt version: {version}")
    print(f"  Status: {version_info['status']}")
    print(f"  Summary: {version_info['description']}")

    return config


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
    question,
    answer,
    original_solution,
    config: Dict[str, Any],
    previous_cot: str,
    evaluation_details: Dict[str, Any],
):
    """改善提案を含むプロンプトでCoTを再生成する

    Args:
        client: OpenAI APIクライアント
        question: 問題文
        answer: 答え
        original_solution: 元の解法
        config: プロンプト設定
        previous_cot: 前回生成したCoT
        evaluation_details: 評価の詳細情報（grade, strengths, weaknesses, scores, suggestions）
    """
    system_prompt = config['prompts']['system']

    # 評価詳細から情報を抽出
    previous_grade = evaluation_details.get('grade', 'Unknown')
    strengths = evaluation_details.get('strengths', [])
    weaknesses = evaluation_details.get('weaknesses', [])
    improvement_suggestions = evaluation_details.get('improvement_suggestions', [])
    learning_value_scores = evaluation_details.get('learning_value_scores', {})

    # フォーマット用のテキストを構築
    strengths_text = "\n".join(f"• {strength}" for strength in strengths) if strengths else "• None identified"
    weaknesses_text = "\n".join(f"• {weakness}" for weakness in weaknesses) if weaknesses else "• None identified"
    improvement_text = (
        "\n".join(f"• {suggestion}" for suggestion in improvement_suggestions)
        if improvement_suggestions
        else "• None provided"
    )

    # スコアの詳細を構築
    scores_lines = []
    if learning_value_scores:
        scores_lines.append(f"• Method explanation: {learning_value_scores.get('method_explanation', 0)}/10")
        scores_lines.append(f"• Step-by-step derivation: {learning_value_scores.get('step_by_step', 0)}/10")
        scores_lines.append(f"• Verification: {learning_value_scores.get('verification', 0)}/10")
        scores_lines.append(f"• Common mistakes handling: {learning_value_scores.get('common_mistakes', 0)}/10")
        scores_lines.append(f"• Physical insight: {learning_value_scores.get('physical_insight', 0)}/10")
        scores_lines.append(f"• Metacognitive elements: {learning_value_scores.get('metacognitive', 0)}/10")
    scores_text = "\n".join(scores_lines) if scores_lines else "• No scores available"

    # 前回のCoTの抽出（<think>タグがある場合）
    if '<think>' in previous_cot and '</think>' in previous_cot:
        start = previous_cot.find('<think>') + len('<think>')
        end = previous_cot.find('</think>')
        previous_cot_content = previous_cot[start:end].strip()
    else:
        previous_cot_content = previous_cot

    # テンプレートを使用してユーザープロンプトを生成
    user_template = config['prompts']['user_template']
    final_prompt = user_template.format(
        question=question,
        original_solution=original_solution,
        previous_cot=previous_cot_content,
        grade=previous_grade,
        strengths=strengths_text,
        weaknesses=weaknesses_text,
        scores=scores_text,
        improvement_suggestions=improvement_text,
    )

    # パラメータを取得
    params = config['parameters']
    model = params.get('model', DEFAULT_MODEL)
    temperature = params.get('temperature', 0.2)
    max_tokens = params.get('max_tokens', 30000)
    retry_attempts = params.get('retry_attempts', 3)

    for attempt in range(retry_attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retry_attempts - 1:  # 最後の試行でない場合
                wait_time = 2**attempt  # 1, 2, 4秒
                print(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error after {retry_attempts} attempts: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description="低品質なCoTを改善提案に基づいて再生成する")
    parser.add_argument('dataset', help='データセット名（phybench, physreason）')
    parser.add_argument(
        '--prompt-version', type=str, default='3.0', help='使用するプロンプトバージョン（デフォルト: 3.0）'
    )
    parser.add_argument('--ids', type=str, help='処理するIDをカンマ区切りで指定（例: 1,5,17）')
    parser.add_argument('--force', action='store_true', help='未評価のCoTがあっても強制的に再生成')
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
    print(f"Loading evaluated data from: {input_path}")

    # プロンプト設定を読み込み
    try:
        config = load_prompt_config(args.prompt_version, prompt_type='regenerate')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading prompt config: {e}")
        return

    # API設定
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set")
        return

    api_base_url = config['parameters'].get('api_base_url', DEFAULT_API_BASE_URL)
    client = openai.OpenAI(api_key=api_key, base_url=api_base_url)

    # データ読み込み
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 再生成対象のフィルタリング
    items_to_regenerate = []
    skipped_unevaluated = []

    for item in data:
        # 評価結果が存在するか確認
        if not item.get('metadata', {}).get('cot_history'):
            continue

        cot_history = item['metadata']['cot_history']
        latest_history = cot_history[-1]
        item_id = str(item.get('id', ''))

        # 最新のCoTが未評価の場合はスキップ（--forceオプションがない限り）
        if not args.force and latest_history.get('evaluation') is None:
            skipped_unevaluated.append(item_id)
            continue

        evaluation = latest_history.get('evaluation')
        if not evaluation:
            continue

        grade = evaluation.get('grade', 'D')

        # ID指定がある場合
        if args.ids:
            target_ids = parse_ids(args.ids)
            if item_id in target_ids:
                items_to_regenerate.append(item)
        # ID指定がない場合はGrade D/Cのみ
        elif grade in ['D', 'C']:
            items_to_regenerate.append(item)

    # スキップしたアイテムの報告
    if skipped_unevaluated:
        print(f"\nSkipped {len(skipped_unevaluated)} items with unevaluated latest CoT (use --force to regenerate):")
        # 最初の10個だけ表示
        sample = skipped_unevaluated[:10]
        print(f"  IDs: {', '.join(sample)}{'...' if len(skipped_unevaluated) > 10 else ''}")

    if not items_to_regenerate:
        if args.ids:
            print("\nNo items found with specified IDs or they don't have evaluations")
            if skipped_unevaluated:
                print("Some items were skipped because their latest CoT is not evaluated yet.")
        else:
            print("\nNo items to regenerate (no Grade D/C items found or all have unevaluated CoTs)")
        return

    print(f"\nFound {len(items_to_regenerate)} items to regenerate")

    # ID指定時は各アイテムのGradeを表示
    if args.ids:
        for item in items_to_regenerate:
            item_id = item.get('id', 'unknown')
            grade = item['metadata']['cot_history'][-1]['evaluation'].get('grade', 'Unknown')
            print(f"  ID {item_id}: Grade {grade}")

    # 再生成実行
    success_count = 0

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

        if not evaluation_details['improvement_suggestions']:
            print(f"\nWarning: No improvement suggestions for item {item_id}, skipping")
            continue

        # original_solutionを取得
        original_solution = item['metadata'].get('original_solution')
        if not original_solution:
            print(f"\nWarning: No original_solution for item {item_id}, skipping")
            continue

        # CoT再生成
        output = regenerate_cot(
            client, item["question"], item["answer"], original_solution, config, previous_cot, evaluation_details
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

            # 再生成完了後、即座にファイルに書き込む
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        else:
            print(f"\nWarning: Failed to regenerate CoT for item {item_id}")

    print(f"\nDone! Successfully regenerated {success_count}/{len(items_to_regenerate)} items")
    print(f"Results updated in: {input_path}")


if __name__ == "__main__":
    main()
