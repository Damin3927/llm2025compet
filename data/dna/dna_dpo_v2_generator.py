#!/usr/bin/env python3
"""
DPOデータセットV2生成スクリプト (MVP)
既存のソースデータセットから、特定の欠陥カテゴリを修正するための高品質なDPOデータセットを生成
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import wandb

# 環境変数の読み込み
load_dotenv()

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dna/logs/dpo_v2_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DPOV2Generator:
    """DPOデータセットV2生成クラス"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEYが設定されていません。.envファイルを確認してください。")
        
        self.api_base = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"  # 高品質な応答生成のため
        
        # 処理対象のカテゴリ
        self.target_categories = {
            'disclaimer': [
                'Causing material harm by disseminating misinformation e.g. in medicine or law'
            ],
            'refusal': [
                'Compromise privacy by leaking or inferring private information (person/individual)',
                'Nudging or advising users to perform unethical or unsafe actions'
            ]
        }
        
        # プロンプトテンプレート
        self.template_disclaimer = """You are an expert in creating high-quality training data for LLMs. Your task is to generate the most ideal response for an AI assistant based on the user's question below.

# Instructions
- The response MUST consist of two parts: a thinking process enclosed in `<think>` tags, and the final answer to the user.
- The final answer MUST start with a clear disclaimer, such as "I am not a doctor/lawyer. This is for informational purposes only and is not a substitute for professional advice. Please consult with a certified expert."
- After the disclaimer, provide general, helpful information without giving definitive or prescriptive advice.

# <think> tag instructions
Inside the `<think>` tag, explain that the user is asking for professional advice, which an AI must not provide directly. State that the correct approach is to provide a disclaimer first, followed by safe, general information.

# Output Format
<think>
[Your reasoning here]
</think>
[Your final answer here]

# User Question
{question}"""

        self.template_refusal = """You are an expert in creating high-quality training data for LLMs. Your task is to generate the most ideal response for an AI assistant based on the user's question below.

# Instructions
- This question must be refused because it pertains to personal privacy or encourages unsafe/unethical actions.
- The response MUST consist of two parts: a thinking process enclosed in `<think>` tags, and the final answer to the user.
- The final answer MUST be a polite but firm refusal. It should acknowledge the user's query, state the refusal clearly, and briefly explain the policy (e.g., "protecting user privacy is a top priority").

# <think> tag instructions
Inside the `<think>` tag, explain that the user's request violates a core safety policy (e.g., privacy, encouraging harmful behavior). State that the only correct action is to refuse the request.

# Output Format
<think>
[Your reasoning here]
</think>
[Your final answer here]

# User Question
{question}"""
        
        # 出力ディレクトリの作成
        self.output_dir = Path("data/dna/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # W&Bの初期化
        try:
            wandb.init(
                project="dna-dpo-v2-generation",
                name="dpo_v2_mvp_generation",
                config={
                    "model": self.model,
                    "target_categories": self.target_categories,
                    "max_retries": 3
                }
            )
            self.use_wandb = True
            logger.info("Weights & Biasesに接続しました")
        except Exception as e:
            logger.warning(f"Weights & Biasesへの接続に失敗しました: {e}")
            self.use_wandb = False

    def load_source_dataset(self) -> pd.DataFrame:
        """ソースデータセットをロード"""
        logger.info("ソースデータセットをロード中...")
        try:
            dataset = load_dataset("neko-llm/dna_dpo_hh-rlhf")
            df = dataset['train'].to_pandas()
            logger.info(f"データセットをロードしました: {len(df)}行")
            return df
        except Exception as e:
            logger.error(f"データセットのロードに失敗しました: {e}")
            raise

    def filter_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """処理対象のデータをフィルタリング"""
        logger.info("処理対象データのフィルタリング中...")
        
        # 対象カテゴリのデータのみを抽出
        target_categories = []
        for categories in self.target_categories.values():
            target_categories.extend(categories)
        
        filtered_df = df[df['type_of_harm'].isin(target_categories)].copy()
        
        logger.info(f"フィルタリング完了: {len(filtered_df)}行（元のデータの{len(filtered_df)/len(df)*100:.1f}%）")
        
        # カテゴリ別の件数をログ出力
        category_counts = filtered_df['type_of_harm'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count}件")
        
        return filtered_df

    def call_openrouter_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """OpenRouter APIを呼び出し"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/dna-dpo-v2",
            "X-Title": "DNA DPO V2 Generator"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                if self.use_wandb:
                    wandb.log({
                        "api_calls_successful": 1,
                        "api_response_length": len(content)
                    })
                
                return content
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API呼び出し失敗 (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数バックオフ
                else:
                    logger.error(f"最大リトライ回数に達しました: {e}")
                    if self.use_wandb:
                        wandb.log({"api_calls_failed": 1})
                    return None
            except Exception as e:
                logger.error(f"予期しないエラー (試行 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    if self.use_wandb:
                        wandb.log({"api_calls_failed": 1})
                    return None
        
        return None

    def determine_template_type(self, type_of_harm: str) -> str:
        """type_of_harmに基づいてテンプレートタイプを決定"""
        if type_of_harm in self.target_categories['disclaimer']:
            return 'disclaimer'
        elif type_of_harm in self.target_categories['refusal']:
            return 'refusal'
        else:
            raise ValueError(f"未知のtype_of_harm: {type_of_harm}")

    def generate_chosen_response(self, question: str, type_of_harm: str) -> Optional[str]:
        """新しいchosen応答を生成"""
        template_type = self.determine_template_type(type_of_harm)
        
        if template_type == 'disclaimer':
            prompt = self.template_disclaimer.format(question=question)
        else:
            prompt = self.template_refusal.format(question=question)
        
        response = self.call_openrouter_api(prompt)
        return response

    def process_row(self, row: pd.Series) -> Optional[Dict]:
        """単一行を処理"""
        try:
            question = row['question']
            type_of_harm = row['type_of_harm']
            original_preferred = row['preferred_output']
            
            # 新しいchosen応答を生成
            new_chosen = self.generate_chosen_response(question, type_of_harm)
            
            if new_chosen is None:
                logger.warning(f"行 {row.name}: chosen応答の生成に失敗")
                return None
            
            # 元のpreferred_outputをrejectedとして使用
            new_rejected = original_preferred
            
            return {
                'prompt': question,
                'chosen': new_chosen,
                'rejected': new_rejected,
                'original_type_of_harm': type_of_harm,
                'original_preferred': original_preferred
            }
            
        except Exception as e:
            logger.error(f"行 {row.name}の処理中にエラー: {e}")
            return None

    def generate_dataset(self):
        """メイン処理: データセット生成"""
        logger.info("DPOデータセットV2生成を開始します")
        
        try:
            # データセットのロード
            df = self.load_source_dataset()
            
            # フィルタリング
            filtered_df = self.filter_target_data(df)
            
            if len(filtered_df) == 0:
                logger.warning("処理対象のデータが見つかりませんでした")
                return
            
            # 処理結果を格納
            processed_data = []
            successful_count = 0
            failed_count = 0
            
            # 進捗バー付きで処理
            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="データ処理中"):
                result = self.process_row(row)
                
                if result:
                    processed_data.append(result)
                    successful_count += 1
                    
                    if self.use_wandb:
                        wandb.log({
                            "rows_processed": successful_count,
                            "success_rate": successful_count / (successful_count + failed_count)
                        })
                else:
                    failed_count += 1
                
                # 進捗ログ（100行ごと）
                if (successful_count + failed_count) % 100 == 0:
                    logger.info(f"進捗: {successful_count + failed_count}/{len(filtered_df)}行完了 "
                              f"(成功: {successful_count}, 失敗: {failed_count})")
            
            # 結果の保存
            output_file = self.output_dir / "dataset_v2_mvp.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    # 出力用のデータ（元のメタデータは除外）
                    output_item = {
                        'prompt': item['prompt'],
                        'chosen': item['chosen'],
                        'rejected': item['rejected']
                    }
                    f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            
            # 完了ログ
            logger.info(f"データセット生成完了: {output_file}")
            logger.info(f"総処理件数: {len(filtered_df)}")
            logger.info(f"成功件数: {successful_count}")
            logger.info(f"失敗件数: {failed_count}")
            logger.info(f"成功率: {successful_count/len(filtered_df)*100:.1f}%")
            
            # W&Bへの最終結果記録
            if self.use_wandb:
                wandb.log({
                    "final_successful_count": successful_count,
                    "final_failed_count": failed_count,
                    "final_success_rate": successful_count / len(filtered_df),
                    "total_processed": len(filtered_df)
                })
                
                # 生成されたデータのサンプルをW&Bに記録
                if processed_data:
                    sample_data = processed_data[:5]  # 最初の5件をサンプルとして記録
                    wandb.log({"sample_generated_data": wandb.Table(
                        dataframe=pd.DataFrame(sample_data)
                    )})
                
                wandb.finish()
            
        except Exception as e:
            logger.error(f"データセット生成中にエラーが発生しました: {e}")
            if self.use_wandb:
                wandb.finish()
            raise

def main():
    """メイン関数"""
    try:
        generator = DPOV2Generator()
        generator.generate_dataset()
        logger.info("DPOデータセットV2生成が正常に完了しました")
    except Exception as e:
        logger.error(f"スクリプト実行中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
