import json
import pandas as pd
import torch
import logging
import sys
import os
import re
from typing import Dict, List, Tuple, Optional
from vllm import LLM, SamplingParams
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityImprover:
    def __init__(self, judge_model_name="Qwen/Qwen2.5-14B-Instruct"):
        """品質改善システムの初期化"""
        self.judge_llm = None
        self.judge_model_name = judge_model_name
        self.improvement_log = []
        
    def initialize_judge_model(self):
        """判定用LLMの初期化"""
        logger.info(f"🔧 判定用モデル初期化中: {self.judge_model_name}")
        
        # 軽量な判定用モデル
        self.judge_llm = LLM(
            model=self.judge_model_name,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_model_len=4096,
            max_num_seqs=4
        )
        logger.info("✅ 判定用モデル初期化完了")

    def evaluate_cot_quality(self, problem: str, cot: str, answer: str) -> Tuple[float, List[str]]:
        """CoTの品質を自動評価"""
        score = 0.0
        issues = []
        max_score = 6.0
        
        # 1. 長さチェック (0-1点)
        cot_length = len(cot.strip())
        if 80 <= cot_length <= 800:
            score += 1.0
        elif cot_length < 80:
            issues.append(f"CoTが短すぎます ({cot_length}文字)")
        else:
            issues.append(f"CoTが長すぎます ({cot_length}文字)")
        
        # 2. 論理的接続詞チェック (0-1点)
        reasoning_patterns = [
            r'\b(because|since|therefore|thus|so|hence)\b',
            r'\b(first|second|third|finally|next|then)\b', 
            r'\b(まず|次に|そして|最後に|なので|理由は|従って)\b'
        ]
        
        reasoning_count = sum(len(re.findall(pattern, cot, re.IGNORECASE)) for pattern in reasoning_patterns)
        if reasoning_count >= 2:
            score += 1.0
        else:
            issues.append(f"論理的接続詞が不足 (検出数: {reasoning_count})")
        
        # 3. 問題との関連性チェック (0-1点)
        problem_words = set(re.findall(r'\w+', problem.lower()))
        cot_words = set(re.findall(r'\w+', cot.lower()))
        if len(problem_words) > 0:
            overlap_ratio = len(problem_words & cot_words) / len(problem_words)
            if overlap_ratio >= 0.25:
                score += 1.0
            else:
                issues.append(f"問題との関連性が低い (重複率: {overlap_ratio:.2f})")
        
        # 4. 段落構造チェック (0-1点)
        sentences = re.split(r'[.!?。？！]\s*', cot)
        if len(sentences) >= 3:
            score += 1.0
        else:
            issues.append(f"推論ステップが不足 (文数: {len(sentences)})")
        
        # 5. 安全性チェック (0-1点)
        harmful_patterns = [
            r'\b(kill|murder|harm|attack|violence|illegal|hack)\b',
            r'\b(殺|害|攻撃|違法|ハック|暴力)\b'
        ]
        
        is_harmful_request = any(re.search(pattern, problem, re.IGNORECASE) for pattern in harmful_patterns)
        has_refusal = any(phrase in cot.lower() for phrase in ["sorry", "can't help", "申し訳", "お手伝いできません"])
        
        if is_harmful_request and has_refusal:
            score += 1.0  # 適切な拒否
        elif not is_harmful_request and not has_refusal:
            score += 1.0  # 通常の回答
        else:
            if is_harmful_request and not has_refusal:
                issues.append("有害な要求に対して適切に拒否していない")
            else:
                issues.append("無害な要求に対して不適切に拒否している")
        
        # 6. 答えとの整合性チェック (0-1点)
        answer_words = set(re.findall(r'\w+', answer.lower()))
        if len(answer_words) > 0:
            answer_overlap = len(answer_words & cot_words) / len(answer_words)
            if answer_overlap >= 0.3:
                score += 1.0
            else:
                issues.append(f"回答との整合性が低い (重複率: {answer_overlap:.2f})")
        
        return score / max_score, issues

    def evaluate_answer_quality(self, problem: str, answer: str) -> Tuple[float, List[str]]:
        """回答の品質を自動評価"""
        score = 0.0
        issues = []
        max_score = 4.0
        
        # 1. 長さチェック (0-1点)
        answer_length = len(answer.strip())
        if 20 <= answer_length <= 2000:
            score += 1.0
        elif answer_length < 20:
            issues.append(f"回答が短すぎます ({answer_length}文字)")
        else:
            issues.append(f"回答が長すぎます ({answer_length}文字)")
        
        # 2. 具体性チェック (0-1点)
        specific_patterns = [
            r'\b\d+\b',  # 数字
            r'\b(step|method|way|approach|technique|方法|手順|ステップ)\b',
            r'\b(example|instance|such as|例えば|例として)\b'
        ]
        
        specific_count = sum(len(re.findall(pattern, answer, re.IGNORECASE)) for pattern in specific_patterns)
        if specific_count >= 2:
            score += 1.0
        else:
            issues.append(f"具体性が不足 (検出数: {specific_count})")
        
        # 3. 問題解決性チェック (0-1点)
        problem_keywords = re.findall(r'\w+', problem.lower())
        answer_addresses_problem = any(word in answer.lower() for word in problem_keywords[:5])
        
        if answer_addresses_problem:
            score += 1.0
        else:
            issues.append("問題に対する直接的な回答が不足")
        
        # 4. 安全性と適切性チェック (0-1点)
        harmful_patterns = [
            r'\b(kill|murder|harm|attack|violence|illegal|hack)\b',
            r'\b(殺|害|攻撃|違法|ハック|暴力)\b'
        ]
        
        is_harmful_request = any(re.search(pattern, problem, re.IGNORECASE) for pattern in harmful_patterns)
        has_harmful_content = any(re.search(pattern, answer, re.IGNORECASE) for pattern in harmful_patterns)
        has_refusal = any(phrase in answer.lower() for phrase in ["sorry", "can't help", "申し訳", "お手伝いできません"])
        
        if is_harmful_request:
            if has_refusal and not has_harmful_content:
                score += 1.0  # 適切な拒否
            else:
                issues.append("有害な要求に対して適切に拒否していない")
        else:
            if not has_harmful_content:
                score += 1.0  # 安全な回答
            else:
                issues.append("無害な要求に対して有害な内容を含む")
        
        return score / max_score, issues

    def llm_judge_quality(self, problem: str, cot: str, answer: str) -> float:
        """LLMによる品質判定"""
        if self.judge_llm is None:
            return 0.5  # デフォルトスコア
        
        judge_prompt = f"""あなたは品質評価の専門家です。以下の推論チェーンと回答を1-5のスケールで評価してください。

問題: {problem}

推論: {cot}

回答: {answer}

評価基準:
1. 論理的一貫性 (推論が論理的に正しいか)
2. 関連性 (問題に対して適切に答えているか)  
3. 明確性 (理解しやすく明確か)
4. 完全性 (十分な説明があるか)
5. 安全性 (有害でない内容か)

"SCORE: X.X" の形式で平均スコアのみを出力してください。"""

        try:
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=50,
                stop=["\n", "。"]
            )
            
            result = self.judge_llm.generate(judge_prompt, sampling_params)[0].outputs[0].text
            score_match = re.search(r'SCORE:\s*(\d+\.?\d*)', result)
            
            if score_match:
                score = float(score_match.group(1))
                return min(max(score / 5.0, 0.0), 1.0)  # 0-1の範囲に正規化
            else:
                logger.warning(f"スコア抽出失敗: {result}")
                return 0.5
                
        except Exception as e:
            logger.error(f"LLM判定エラー: {e}")
            return 0.5

    def improve_cot(self, problem: str, answer: str, current_cot: str, attempt: int = 0) -> str:
        """CoTを改善"""
        improvement_prompts = [
            # 基本改善
            f"""## System Role
あなたは論理的思考の専門家です。与えられた問題と回答に対して、より詳細で論理的な推論過程を生成してください。

## 改善指示
- 各ステップを明確に区切る
- "まず"、"次に"、"そして"、"最後に"などの接続詞を使用
- 推論の根拠を明確に示す
- 100-400文字程度で簡潔に

## 問題
{problem}

## 回答  
{answer}

## 改善された推論
<think>""",
            
            # 詳細化
            f"""## System Role
あなたは教育の専門家です。以下の問題に対してより詳細で分かりやすい段階的推論を提供してください。

## 改善指示
- より具体的な例や説明を含める
- 各推論ステップの理由を明確にする
- 読者が理解しやすい構造にする

## 問題
{problem}

## 目標回答
{answer}

## 詳細推論
<think>""",
            
            # 構造化
            f"""## System Role  
あなたは構造化思考の専門家です。問題を体系的に分析し、論理的な推論チェーンを構築してください。

## 改善指示
- 問題を要素分解する
- 各要素を順序立てて検討する
- 結論に至る明確な論理的道筋を示す

## 問題分析対象
{problem}

## 期待結果
{answer}

## 構造化推論
<think>"""
        ]
        
        if attempt < len(improvement_prompts):
            prompt = improvement_prompts[attempt]
        else:
            prompt = improvement_prompts[0]  # デフォルト
            
        return prompt

    def improve_answer(self, problem: str, current_answer: str, attempt: int = 0) -> str:
        """回答を改善"""
        improvement_prompts = [
            # 基本改善
            f"""## System Role
あなたは回答品質の専門家です。与えられた問題に対してより良い回答を生成してください。

## 改善指示
- より具体的で実用的な回答にする
- 適切な長さ（50-500文字程度）を保つ
- 問題に直接答える
- 必要に応じて安全性に配慮する

## 問題
{problem}

## 現在の回答
{current_answer}

## 改善された回答
""",
            
            # 詳細化
            f"""## System Role
あなたは詳細説明の専門家です。以下の問題により詳しく丁寧に回答してください。

## 改善指示  
- より詳細な情報を提供する
- 具体例や手順を含める
- 読者にとって実用的な内容にする

## 問題
{problem}

## 元回答
{current_answer}

## 詳細回答
""",
            
            # 構造化
            f"""## System Role
あなたは構造化回答の専門家です。問題に対して整理された形で回答してください。

## 改善指示
- 要点を整理して提示する
- 必要に応じて番号付きリストを使用
- 明確で読みやすい構造にする

## 質問
{problem}

## 基礎回答
{current_answer}

## 構造化回答
"""
        ]
        
        if attempt < len(improvement_prompts):
            return improvement_prompts[attempt]
        else:
            return improvement_prompts[0]

    def process_item(self, item: Dict, improvement_llm: LLM) -> Tuple[Dict, bool]:
        """単一アイテムの品質改善処理"""
        item_id = item.get("id", "unknown")
        problem = item.get("problem", "")
        current_output = item.get("output", "")
        current_answer = item.get("answer", "")
        
        # <think>タグからCoTを抽出
        cot_match = re.search(r'<think>(.*?)</think>', current_output, re.DOTALL)
        current_cot = cot_match.group(1).strip() if cot_match else ""
        
        # 現在の品質を評価
        cot_score, cot_issues = self.evaluate_cot_quality(problem, current_cot, current_answer)
        answer_score, answer_issues = self.evaluate_answer_quality(problem, current_answer)
        llm_score = self.llm_judge_quality(problem, current_cot, current_answer)
        
        # 総合スコア計算
        overall_score = (cot_score * 0.4 + answer_score * 0.3 + llm_score * 0.3)
        
        logger.info(f"📊 ID {item_id} - CoT: {cot_score:.2f}, Answer: {answer_score:.2f}, LLM: {llm_score:.2f}, 総合: {overall_score:.2f}")
        
        # 改善が必要かどうか判定
        needs_improvement = overall_score < 0.7
        improved = False
        improvement_reasons = []
        
        if needs_improvement:
            logger.info(f"🔧 ID {item_id} 改善開始 - スコア不足: {overall_score:.2f}")
            
            best_cot = current_cot
            best_answer = current_answer
            best_score = overall_score
            
            # CoT改善
            if cot_score < 0.7:
                improvement_reasons.extend(cot_issues)
                for attempt in range(2):  # 最大2回試行
                    try:
                        improve_prompt = self.improve_cot(problem, current_answer, current_cot, attempt)
                        
                        sampling_params = SamplingParams(
                            temperature=0.3 + (attempt * 0.2),
                            max_tokens=600,
                            stop=["</think>"]
                        )
                        
                        result = improvement_llm.generate(improve_prompt, sampling_params)[0].outputs[0].text
                        new_cot = result.strip()
                        
                        # 新しいCoTを評価
                        new_cot_score, _ = self.evaluate_cot_quality(problem, new_cot, current_answer)
                        
                        if new_cot_score > cot_score:
                            best_cot = new_cot
                            cot_score = new_cot_score
                            logger.info(f"  ✅ CoT改善成功 (試行{attempt+1}): {new_cot_score:.2f}")
                            break
                        else:
                            logger.info(f"  ⚠️ CoT改善失敗 (試行{attempt+1}): {new_cot_score:.2f}")
                            
                    except Exception as e:
                        logger.error(f"  ❌ CoT改善エラー (試行{attempt+1}): {e}")
            
            # Answer改善
            if answer_score < 0.7:
                improvement_reasons.extend(answer_issues)
                for attempt in range(2):  # 最大2回試行
                    try:
                        improve_prompt = self.improve_answer(problem, current_answer, attempt)
                        
                        sampling_params = SamplingParams(
                            temperature=0.3 + (attempt * 0.2),
                            max_tokens=800,
                            stop=["\n\n", "##"]
                        )
                        
                        result = improvement_llm.generate(improve_prompt, sampling_params)[0].outputs[0].text
                        new_answer = result.strip()
                        
                        # 新しい回答を評価
                        new_answer_score, _ = self.evaluate_answer_quality(problem, new_answer)
                        
                        if new_answer_score > answer_score:
                            best_answer = new_answer
                            answer_score = new_answer_score
                            logger.info(f"  ✅ Answer改善成功 (試行{attempt+1}): {new_answer_score:.2f}")
                            break
                        else:
                            logger.info(f"  ⚠️ Answer改善失敗 (試行{attempt+1}): {new_answer_score:.2f}")
                            
                    except Exception as e:
                        logger.error(f"  ❌ Answer改善エラー (試行{attempt+1}): {e}")
            
            # 最終評価
            final_llm_score = self.llm_judge_quality(problem, best_cot, best_answer)
            final_score = (cot_score * 0.4 + answer_score * 0.3 + final_llm_score * 0.3)
            
            if final_score > overall_score:
                # 改善されたアイテムを返す
                improved_output = f"<think>{best_cot}</think>{best_answer}"
                improved_item = item.copy()
                improved_item["output"] = improved_output
                improved_item["answer"] = best_answer
                
                improvement_log = {
                    "id": item_id,
                    "timestamp": datetime.now().isoformat(),
                    "original_score": overall_score,
                    "improved_score": final_score,
                    "improvement": final_score - overall_score,
                    "reasons": improvement_reasons,
                    "cot_improved": best_cot != current_cot,
                    "answer_improved": best_answer != current_answer
                }
                
                self.improvement_log.append(improvement_log)
                
                logger.info(f"🎯 ID {item_id} 改善完了: {overall_score:.2f} → {final_score:.2f} (+{final_score-overall_score:.2f})")
                return improved_item, True
            else:
                logger.info(f"🔄 ID {item_id} 改善効果なし: {overall_score:.2f}")
        
        return item, improved

def main():
    """メイン処理"""
    # 入力ファイルチェック
    input_file = "vanilla_with_cot_vllm.jsonl"
    if not os.path.exists(input_file):
        logger.error(f"❌ 入力ファイルが見つかりません: {input_file}")
        logger.error("先に generate_cot.py を実行してください")
        sys.exit(1)
    
    # データ読み込み
    logger.info(f"📂 データ読み込み中: {input_file}")
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():
                    data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error at line {line_num}: {e}")
    
    logger.info(f"✅ {len(data)} 件のデータを読み込み完了")
    
    # GPU情報確認
    logger.info("🔍 GPU情報を確認中...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"🎯 利用可能GPU数: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.error("❌ CUDA利用不可")
        sys.exit(1)
    
    # 品質改善システム初期化
    improver = QualityImprover()
    improver.initialize_judge_model()
    
    # 改善用LLM初期化
    logger.info("🚀 改善用モデル初期化中...")
    improvement_llm = LLM(
        model="Qwen/Qwen3-32B",
        trust_remote_code=True,
        tensor_parallel_size=2,  # 2GPU使用
        gpu_memory_utilization=0.8,
        max_model_len=6144,
        max_num_seqs=8
    )
    logger.info("✅ 改善用モデル初期化完了")
    
    # 品質改善処理
    improved_data = []
    improvement_count = 0
    
    logger.info("🔧 品質改善処理開始...")
    
    for i, item in enumerate(data):
        logger.info(f"📝 処理中: {i+1}/{len(data)} (ID: {item.get('id', 'unknown')})")
        
        improved_item, was_improved = improver.process_item(item, improvement_llm)
        improved_data.append(improved_item)
        
        if was_improved:
            improvement_count += 1
        
        # 進捗表示
        if (i + 1) % 50 == 0:
            logger.info(f"📊 進捗: {i+1}/{len(data)} 完了, 改善件数: {improvement_count}")
    
    # 結果保存
    output_file = "improved_vanilla_with_cot.jsonl"
    logger.info(f"💾 改善データ保存中: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in improved_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 改善ログ保存
    log_file = f"improvement_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    logger.info(f"📋 改善ログ保存中: {log_file}")
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_items": len(data),
                "improved_items": improvement_count,
                "improvement_rate": improvement_count / len(data) if data else 0
            },
            "improvements": improver.improvement_log
        }, f, ensure_ascii=False, indent=2)
    
    # 統計出力
    logger.info("🎯 品質改善完了!")
    logger.info(f"  📊 総データ数: {len(data)}")
    logger.info(f"  ✅ 改善件数: {improvement_count}")
    logger.info(f"  📈 改善率: {improvement_count/len(data)*100:.1f}%")
    logger.info(f"  💾 出力ファイル: {output_file}")
    logger.info(f"  📋 ログファイル: {log_file}")

if __name__ == "__main__":
    main()