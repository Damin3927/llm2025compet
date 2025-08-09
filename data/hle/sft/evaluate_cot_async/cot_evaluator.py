#!/usr/bin/env python3
"""
CoT評価クラス

CoT（Chain of Thought）の品質を評価し、学習データとしての
適性をグレード（A〜D）で判定するクラス。

評価は必須要件4項目と学習価値6観点で行い、
結果はmetadata.cot_historyに記録される。

使用方法：
from cot_evaluator import CoTEvaluationProcessor

processor = CoTEvaluationProcessor()
processor.evaluate_dataset('dataset.jsonl', output_file='evaluated_dataset.jsonl')
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import asyncio
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict
import time
from datetime import datetime
import re

# Hugging Face datasets support (optional)
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: Hugging Face datasets not available. Install with: pip install datasets")

# .envファイルを読み込み
load_dotenv()

# モデル設定（単一モデルのデフォルト。複数指定は --evaluator-models で上書き）
DEFAULT_MODEL = "deepseek/deepseek-r1-0528:free"
DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.0  # 評価の一貫性を最重視するため0に固定
DEFAULT_MAX_TOKENS = 8000  # 詳細な評価結果を取得するための十分な長さ


def _build_evaluation_prompt(question: str, cot: str, answer: str) -> str:
    """評価用プロンプトを構築する。"""
    return f"""Evaluate the following CoT (Chain of Thought) from the perspective of its value as training data for machine learning models.

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


def _safe_json_parse(result_text: str) -> Dict[str, Any]:
    """Try to parse JSON; if it fails, attempt to extract the largest JSON object substring."""
    try:
        return json.loads(result_text)
    except Exception:
        # Try to find a JSON object by matching braces
        start = result_text.find('{')
        end = result_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            candidate = result_text[start:end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass
    raise json.JSONDecodeError("Invalid JSON after attempts", result_text, 0)


class CoTEvaluator:
    """生成されたCoTの品質を評価し、学習データとしての採用可否を判定する"""

    def __init__(self, llm_client=None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.llm_client = llm_client
        self.api_key = api_key
        self.base_url = base_url

    def evaluate_with_llm(self, question: str, cot: str, answer: str, model_name: Optional[str] = None) -> Dict:
        """LLMを使用してCoTの品質を評価し、詳細な評価結果を返す（同期・単一モデル）"""
        prompt = _build_evaluation_prompt(question, cot, answer)

        use_model = model_name or DEFAULT_MODEL

        for attempt in range(3):
            try:
                response = self.llm_client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in evaluating the quality of training data for machine learning. Focus on logical completeness and educational value, not style or expressions.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=DEFAULT_TEMPERATURE,
                    max_tokens=DEFAULT_MAX_TOKENS,
                )

                result_text = response.choices[0].message.content

                # JSONをパース
                try:
                    result = _safe_json_parse(result_text)
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

    async def _evaluate_one_async(self, async_client: Any, model_name: str, question: str, cot_content: str, answer: str) -> Dict[str, Any]:
        prompt = _build_evaluation_prompt(question, cot_content, answer)
        try:
            response = await async_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in evaluating the quality of training data for machine learning. Focus on logical completeness and educational value, not style or expressions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            result_text = response.choices[0].message.content
            parsed = _safe_json_parse(result_text)
            parsed['evaluation_success'] = True
            return parsed
        except Exception as e:
            return {"evaluation_success": False, "error": f"{type(e).__name__}: {str(e)}"}

    def _aggregate_multi(self, model_names: List[str], evals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """複数モデルの評価結果を集約し、単一の評価形式に整形する。"""
        by_model = {}
        valid_evals = []
        for m, ev in zip(model_names, evals):
            by_model[m] = ev
            if ev.get('evaluation_success'):
                valid_evals.append(ev)

        # 1) Mandatory majority
        keys = ["independence", "logical_completeness", "correctness", "answer_reached"]
        mandatory = {}
        for k in keys:
            votes = [bool(e.get('mandatory_requirements', {}).get(k, {}).get('passed', False)) for e in valid_evals]
            passed = sum(votes) >= max(1, (len(votes) + 1) // 2)  # majority
            mandatory[k] = {"passed": passed, "reason": "Aggregated", "examples": []} if k == "independence" else {"passed": passed, "reason": "Aggregated"}

        # 2) Learning scores mean
        score_keys = [
            "method_explanation",
            "step_by_step",
            "verification",
            "common_mistakes",
            "domain_insight",
            "metacognitive",
        ]
        learning_scores = {}
        for sk in score_keys:
            vals = []
            for e in valid_evals:
                v = e.get('learning_value_scores', {}).get(sk, {}).get('score', None)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            avg = sum(vals) / len(vals) if vals else 0.0
            learning_scores[sk] = {"score": round(avg, 2), "reason": "Aggregated"}

        # 3) Text fields: merge & dedup up to 5
        def _merge_list(key: str) -> List[str]:
            seen = set()
            merged = []
            for e in valid_evals:
                for item in e.get(key, []) or []:
                    if isinstance(item, str) and item not in seen:
                        seen.add(item)
                        merged.append(item)
                        if len(merged) >= 5:
                            return merged
            return merged

        strengths = _merge_list('strengths')
        weaknesses = _merge_list('weaknesses')
        improvements = _merge_list('improvement_suggestions')

        aggregated_eval = {
            'mandatory_requirements': mandatory,
            'learning_value_scores': learning_scores,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'improvement_suggestions': improvements,
            'by_model': by_model,
        }
        return aggregated_eval

    async def evaluate_multi(self, question: str, cot: str, answer: str, model_names: List[str], concurrency: int = 4) -> Dict[str, Any]:
        """複数モデルで並行評価し、集約結果と詳細を返す。"""
        # CoTの抽出（<think>タグがある場合）
        if '<think>' in cot and '</think>' in cot:
            start = cot.find('<think>') + len('<think>')
            end = cot.find('</think>')
            cot_content = cot[start:end].strip()
        else:
            cot_content = cot

        async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(model_name: str):
            async with sem:
                return await self._evaluate_one_async(async_client, model_name, question, cot_content, answer)

        tasks = [asyncio.create_task(_bounded(m)) for m in model_names]
        evals = await asyncio.gather(*tasks)
        aggregated_eval = self._aggregate_multi(model_names, evals)
        grade_info = self.calculate_grade(aggregated_eval)

        return {
            'grade': grade_info['grade'],
            'score': grade_info['score'],
            'evaluation': aggregated_eval,
            'grade_info': grade_info,
        }


class CoTEvaluationProcessor:
    """CoT評価のメイン処理クラス"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        CoT評価プロセッサを初期化
        
        Args:
            api_key: OpenAI API key (Noneの場合は環境変数から取得)
            base_url: API base URL (Noneの場合はデフォルト)
            model: 使用するモデル名 (Noneの場合はデフォルト)
            temperature: 生成温度
            max_tokens: 最大トークン数
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided and OPENROUTER_API_KEY not set in environment")
        
        self.base_url = base_url or DEFAULT_API_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # LLMクライアントの初期化
        self.llm_client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 評価器の初期化
        self.evaluator = CoTEvaluator(self.llm_client, api_key=self.api_key, base_url=self.base_url)
    
    def parse_ids(self, ids_str: str) -> List[str]:
        """カンマ区切りのID文字列をリストに変換"""
        return [id.strip() for id in ids_str.split(',')]
    
    def detect_dataset_format(self, dataset_path: str) -> str:
        """データセットの形式を自動検出する"""
        if dataset_path.startswith('hf://') or '/' in dataset_path and ':' in dataset_path:
            return 'huggingface'
        elif dataset_path.endswith('.jsonl'):
            return 'jsonl'
        elif dataset_path.endswith('.json'):
            return 'json'
        else:
            # デフォルトはJSONLとして扱う
            return 'jsonl'
    
    def load_huggingface_dataset(self, dataset_name: str, config: Optional[str] = None, split: Optional[str] = 'train') -> List[Dict]:
        """Hugging Faceデータセットを読み込む"""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("Hugging Face datasets not available. Install with: pip install datasets")
        
        print(f"Loading Hugging Face dataset: {dataset_name}")
        if config:
            print(f"  Config: {config}")
        print(f"  Split: {split}")
        
        dataset = load_dataset(dataset_name, config, split=split)
        return dataset.to_list()
    
    def load_json_file(self, file_path: Path) -> List[Dict]:
        """JSONファイルを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # JSONの形式に応じて処理
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # データセットが辞書形式の場合、一般的なキーを探す
            common_keys = ['data', 'items', 'examples', 'samples', 'records']
            for key in common_keys:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # 見つからない場合は辞書自体をリストとして返す
            return [data]
        else:
            raise ValueError(f"Unsupported JSON format: {type(data)}")
    
    def load_jsonl_file(self, file_path: Path) -> List[Dict]:
        """JSONLファイルを読み込む"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def load_dataset_file(self, dataset_path: str, config: Optional[str] = None, split: Optional[str] = 'train') -> List[Dict]:
        """データセットファイルを形式に応じて読み込む"""
        format_type = self.detect_dataset_format(dataset_path)
        
        if format_type == 'huggingface':
            return self.load_huggingface_dataset(dataset_path, config, split)
        elif format_type == 'json':
            file_path = Path(dataset_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            return self.load_json_file(file_path)
        elif format_type == 'jsonl':
            file_path = Path(dataset_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            return self.load_jsonl_file(file_path)
        else:
            raise ValueError(f"Unsupported dataset format: {format_type}")
    
    def save_dataset(self, data: List[Dict], file_path: Path, format_type: str = 'jsonl'):
        """データセットを指定形式で保存する"""
        if format_type == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format_type == 'jsonl':
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
        else:
            raise ValueError(f"Unsupported save format: {format_type}")
    
    async def evaluate_single_item(self, item: Dict, model_names: Optional[List[str]] = None, concurrency: int = 4) -> Dict:
        """
        単一アイテムを評価する
        
        Args:
            item: 評価対象のアイテム（question, output, answerフィールドが必要）
            model_names: 使用するモデル名のリスト（Noneの場合はデフォルト単一モデル）
            concurrency: 並行実行数（複数モデルの場合）
        
        Returns:
            評価結果の辞書
        """
        # 必要なフィールドの確認
        required_fields = ['question', 'output', 'answer']
        if not all(key in item for key in required_fields):
            raise ValueError(f"Missing required fields: {[f for f in required_fields if f not in item]}")

        # 評価実行
        if model_names and len(model_names) > 1:
            result = await self.evaluator.evaluate_multi(
                question=item['question'],
                cot=item['output'],
                answer=item['answer'],
                model_names=model_names,
                concurrency=concurrency,
            )
        else:
            # 単一モデル（デフォルトまたは1件のみ指定）
            single_model = model_names[0] if model_names else None
            if single_model:
                # 同期呼び出しでモデルを上書き
                result = self.evaluator.evaluate_with_llm(
                    question=item['question'],
                    cot=(item['output'][item['output'].find('<think>') + 7:item['output'].find('</think>')].strip() if ('<think>' in item['output'] and '</think>' in item['output']) else item['output']),
                    answer=item['answer'],
                    model_name=single_model,
                )
                grade_info = self.evaluator.calculate_grade(result)
                result = {
                    'grade': grade_info['grade'],
                    'score': grade_info['score'],
                    'evaluation': result,
                    'grade_info': grade_info,
                }
            else:
                result = self.evaluator.evaluate(
                    question=item['question'],
                    cot=item['output'],
                    answer=item['answer']
                )
        
        return result
    
    async def evaluate_dataset(self, 
                        dataset_path: str,
                        config: Optional[str] = None,
                        split: Optional[str] = 'train',
                        output_file: Optional[str] = None,
                        ids: Optional[str] = None,
                        evaluator_models: Optional[str] = None,
                        eval_concurrency: int = 4,
                        output_format: str = 'jsonl',
                        skip_existing: bool = True) -> Dict[str, Any]:
        """
        データセット全体を評価する
        
        Args:
            dataset_path: データセットファイルのパス
            output_file: 出力ファイル名（Noneの場合は入力ファイルを上書き）
            ids: 処理するIDをカンマ区切りで指定（Noneの場合は全て）
            evaluator_models: 複数の評価モデルをカンマ区切りで指定
            eval_concurrency: 複数モデル評価時の同時実行数
            output_format: 出力形式（'jsonl' または 'json'）
            skip_existing: 既に評価済みのアイテムをスキップするか
        
        Returns:
            評価統計情報の辞書
        """
        # データセットの読み込み
        print(f"Loading dataset: {dataset_path}")
        all_data = self.load_dataset_file(dataset_path, config, split)
        print(f"Loaded {len(all_data)} items")
        
        # 出力ファイルの設定
        if output_file:
            output_path = Path(output_file)
            if not output_path.parent.exists():
                os.makedirs(output_path.parent, exist_ok=True)
        else:
            # 入力ファイルと同じ場所に保存（Hugging Faceデータセットの場合はJSONLとして保存）
            if dataset_path.startswith('hf://') or '/' in dataset_path and ':' in dataset_path:
                output_path = Path(f"evaluated_dataset.{output_format}")
            else:
                output_path = Path(dataset_path)
        
        # 評価対象のインデックスを記録
        indices_to_evaluate = []
        if ids:
            target_ids = self.parse_ids(ids)
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
        
        # 複数モデル指定の処理
        model_list: List[str] = [m.strip() for m in evaluator_models.split(',') if m.strip()] if evaluator_models else []
        
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
            if skip_existing and item.get('metadata', {}).get('cot_history'):
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
                result = await self.evaluate_single_item(item, model_list, eval_concurrency)
                
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
                    'multi_model': {
                        'models': model_list if model_list else [self.model],
                        'is_aggregated': True if (model_list and len(model_list) > 1) else False,
                    }
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
        self.save_dataset(all_data, output_path, output_format)
        print(f"\n評価結果を保存しました: {output_path}")
        
        # 統計情報の表示
        print(f"\n=== 評価完了（学習データ価値基準） ===")
        print(f"評価実行数: {evaluated_count}")
        print(f"スキップ済み（既評価）: {skipped_count}")
        print(f"総データ数: {len(all_data)}")
        
        stats = {
            'evaluated_count': evaluated_count,
            'skipped_count': skipped_count,
            'total_count': len(all_data),
            'grade_counts': dict(grade_counts),
            'average_score': total_score / valid_scores if valid_scores > 0 else 0,
            'output_file': str(output_path)
        }
        
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
        
        return stats





async def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(
        description="汎用CoT評価スクリプト - 任意のデータセットのCoT品質を評価"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='評価対象のデータセット名'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Hugging Faceデータセットのconfigを指定'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Hugging Faceデータセットのsplitを指定'
    )
    parser.add_argument(
        '--ids',
        type=str,
        help='処理するIDをカンマ区切りで指定（例: 1,5,17）'
    )
    parser.add_argument(
        '--evaluator-models',
        type=str,
        default='',
        help='複数の評価モデルをカンマ区切りで指定（例: modelA,modelB）。未指定ならデフォルト単一モデル'
    )
    parser.add_argument(
        '--eval-concurrency',
        type=int,
        default=4,
        help='複数モデル評価時の同時実行数（デフォルト: 4）'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        choices=['jsonl', 'json'],
        default='jsonl',
        help='出力形式（デフォルト: jsonl）'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        help='出力ファイル名（未指定の場合は入力ファイルを上書き）'
    )
    args = parser.parse_args()
    
    try:
        # プロセッサの初期化
        processor = CoTEvaluationProcessor()
        
        # データセット評価の実行
        stats = await processor.evaluate_dataset(
            dataset_path=args.dataset,
            output_file=args.output_file,
            ids=args.ids,
            evaluator_models=args.evaluator_models,
            eval_concurrency=args.eval_concurrency,
            output_format=args.output_format
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Results saved to: {stats['output_file']}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    exit(asyncio.run(main()))