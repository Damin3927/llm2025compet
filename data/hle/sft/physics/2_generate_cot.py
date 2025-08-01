#!/usr/bin/env python3
"""既存の教科書的解法から自然な思考過程（CoT）を生成"""

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

# .envファイルを読み込み
load_dotenv()

# デフォルト設定
DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt_config(version: str) -> Dict[str, Any]:
    """プロンプト設定をYAMLから読み込む
    
    Args:
        version: プロンプトバージョン（必須）
    
    Returns:
        プロンプト設定の辞書
    """
    if not version:
        raise ValueError("Prompt version must be specified")
    
    # レジストリを読み込み
    registry_path = PROMPTS_DIR / "registry.yaml"
    with open(registry_path, 'r', encoding='utf-8') as f:
        registry = yaml.safe_load(f)
    
    # version_info取得
    version_info = next((v for v in registry['versions'] if v['version'] == version), None)
    
    if not version_info:
        available = [v['version'] for v in registry['versions']]
        raise ValueError(f"Version {version} not found. Available: {available}")
    
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

def generate_cot(client, question, answer, original_solution, config: Dict[str, Any]):
    """既存の解法を自然な思考過程に変換
    
    Args:
        client: OpenAI APIクライアント
        question: 問題文
        answer: 答え
        original_solution: 元の解法
        config: プロンプト設定
    """
    system_prompt = config['prompts']['system']
    
    # テンプレートを使用してユーザープロンプトを生成
    user_template = config['prompts']['user_template']
    prompt = user_template.format(
        question=question,
        original_solution=original_solution,
        answer=answer
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
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < retry_attempts - 1:  # 最後の試行でない場合
                wait_time = 2 ** attempt  # 1, 2, 4秒
                print(f"Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Error after {retry_attempts} attempts: {e}")
                return None

def main():
    parser = argparse.ArgumentParser(description="既存の解法から自然な思考過程（CoT）を生成")
    parser.add_argument('dataset', help='データセット名（phybench, physreason）')
    parser.add_argument('--prompt-version', type=str, required=True, help='使用するプロンプトバージョン（必須）')
    parser.add_argument('--ids', type=str, help='処理するIDをカンマ区切りで指定（例: 1,5,17）')
    args = parser.parse_args()
    
    # データセット名に基づいてパスを設定
    if args.dataset not in ['phybench', 'physreason']:
        print(f"Error: Unknown dataset '{args.dataset}'. Use 'phybench' or 'physreason'.")
        return
    
    input_path = Path(f'data/{args.dataset}/preprocessed/dataset.jsonl')
    output_dir = Path(f'data/{args.dataset}/generated')
    
    # プロンプト設定を読み込み
    try:
        config = load_prompt_config(args.prompt_version)
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
    print(f"Loading {input_path}...")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # ID指定がある場合はフィルタリング
    if args.ids:
        target_ids = parse_ids(args.ids)
        filtered_data = []
        
        for item in data:
            item_id = str(item.get('id', ''))
            if item_id in target_ids:
                filtered_data.append(item)
        
        # 見つからなかったIDを警告
        found_ids = {str(item.get('id', '')) for item in filtered_data}
        missing_ids = set(target_ids) - found_ids
        if missing_ids:
            print(f"Warning: IDs not found: {', '.join(sorted(missing_ids))}")
        
        data = filtered_data
        print(f"Processing {len(data)} items with IDs: {', '.join([str(item.get('id', '?')) for item in data])}")
    
    # 出力準備
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # プロンプトバージョンと実行日時を含むファイル名を生成
    timestamp = time.strftime('%Y%m%d_%H%M')
    version = config['metadata']['version']
    output_file = output_dir / f"generated_cot_v{version}_{timestamp}.jsonl"
    
    print(f"Output file: {output_file}")
    
    # CoT生成
    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(data, desc="Generating CoT")):
            # 必須フィールドの確認
            if not all(key in item for key in ["question", "answer", "original_solution"]):
                print(f"\nWarning: Skipping item {i} - missing required fields")
                continue
            
            # CoT生成
            output = generate_cot(client, item["question"], item["answer"], item["original_solution"], config)
            
            if output:
                # 最終形式（CoT + 答え）
                final_output = f"<think>{output}</think>{item['answer']}"
                
                final_data = {
                    "id": item.get("id", f"generated_{success_count + 1}"),
                    "question": item["question"],
                    "output": final_output,
                    "answer": item["answer"]
                }
                
                json.dump(final_data, f, ensure_ascii=False)
                f.write('\n')
                f.flush()
                success_count += 1
            else:
                print(f"\nWarning: Failed to generate CoT for item {i} (id: {item.get('id', 'unknown')})")
    
    print(f"\nDone! Success: {success_count}/{len(data)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()