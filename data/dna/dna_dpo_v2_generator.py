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
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import concurrent.futures

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
    
    def __init__(self, max_workers: int = None):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEYが設定されていません。.envファイルを確認してください。")
        
        self.api_base = "https://openrouter.ai/api/v1"
        self.model = "anthropic/claude-3.5-sonnet"  # 高品質な応答生成のため
        
        # 並列処理の設定
        self.max_workers = max_workers or min(cpu_count(), 8)  # 最大8プロセス
        logger.info(f"並列処理設定: {self.max_workers}ワーカー")
        
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
                    "max_retries": 3,
                    "max_workers": self.max_workers
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
            train_data = dataset['train']
            # to_pandas()メソッドの存在を確認
            if hasattr(train_data, 'to_pandas'):
                df = train_data.to_pandas()
            else:
                # 代替方法: リストに変換してからDataFrameに変換
                df = pd.DataFrame(list(train_data))
            
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

    def process_row_parallel(self, row_data: Tuple[int, Dict]) -> Optional[Dict]:
        """並列処理用の行処理（pickle可能な形式）"""
        try:
            idx, row_dict = row_data
            question = row_dict['question']
            type_of_harm = row_dict['type_of_harm']
            original_preferred = row_dict['preferred_output']
            
            # 新しいchosen応答を生成
            new_chosen = self.generate_chosen_response(question, type_of_harm)
            
            if new_chosen is None:
                logger.warning(f"行 {idx}: chosen応答の生成に失敗")
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
            logger.error(f"行 {idx}の処理中にエラー: {e}")
            return None

    def process_batch_parallel(self, batch_data: List[Tuple[int, Dict]]) -> List[Optional[Dict]]:
        """バッチ処理（並列実行用）"""
        results = []
        for row_data in batch_data:
            result = self.process_row_parallel(row_data)
            results.append(result)
        return results

    def generate_dataset(self, use_parallel: bool = True):
        """メイン処理: データセット生成"""
        logger.info("DPOデータセットV2生成を開始します")
        
        if use_parallel:
            logger.info(f"並列処理モード: {self.max_workers}ワーカー")
            return self._generate_dataset_parallel()
        else:
            logger.info("シーケンシャル処理モード")
            return self._generate_dataset_sequential()

    def _generate_dataset_sequential(self):
        """シーケンシャル処理によるデータセット生成"""
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
            
            # 中間保存用のカウンター
            save_interval = 50  # 50件ごとに中間保存
            
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
                
                # 進捗ログと中間保存（50行ごと）
                if (successful_count + failed_count) % save_interval == 0:
                    current_progress = successful_count + failed_count
                    logger.info(f"進捗: {current_progress}/{len(filtered_df)}行完了 "
                              f"(成功: {successful_count}, 失敗: {failed_count})")
                    
                    # 中間結果をJSONで出力
                    self._save_intermediate_results(processed_data, current_progress)
                    
                    # 最新の5件をサンプルとして表示
                    if processed_data:
                        self._display_sample_data(processed_data[-5:], current_progress)
                
                # 100行ごとの詳細ログ
                if (successful_count + failed_count) % 100 == 0:
                    logger.info(f"詳細進捗: {successful_count + failed_count}/{len(filtered_df)}行完了 "
                              f"(成功: {successful_count}, 失敗: {failed_count})")
                    
                    # 統計情報を表示
                    self._display_statistics(processed_data, filtered_df, current_progress)
            
            # 最終結果の保存
            self._save_final_results(processed_data, filtered_df, successful_count, failed_count)
            
        except Exception as e:
            logger.error(f"データセット生成中にエラーが発生しました: {e}")
            if self.use_wandb:
                wandb.finish()
            raise

    def _generate_dataset_parallel(self):
        """並列処理によるデータセット生成"""
        try:
            # データセットのロード
            df = self.load_source_dataset()
            
            # フィルタリング
            filtered_df = self.filter_target_data(df)
            
            if len(filtered_df) == 0:
                logger.warning("処理対象のデータが見つかりませんでした")
                return
            
            # 並列処理用のデータ形式に変換
            rows_data = [(idx, row.to_dict()) for idx, row in filtered_df.iterrows()]
            
            # バッチサイズの計算
            batch_size = max(1, len(rows_data) // (self.max_workers * 4))  # ワーカー数の4倍でバッチ分割
            batches = [rows_data[i:i + batch_size] for i in range(0, len(rows_data), batch_size)]
            
            logger.info(f"並列処理設定: {len(batches)}バッチ、バッチサイズ: {batch_size}")
            
            # 処理結果を格納
            processed_data = []
            successful_count = 0
            failed_count = 0
            
            # 並列処理の実行（ThreadPoolExecutorを使用）
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # バッチ処理を並列実行
                future_to_batch = {
                    executor.submit(self._process_batch_parallel, batch): batch 
                    for batch in batches
                }
                
                # 完了したバッチから結果を収集
                for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                                 total=len(batches), desc="並列処理中"):
                    batch = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        
                        # 結果を処理
                        for result in batch_results:
                            if result:
                                processed_data.append(result)
                                successful_count += 1
                            else:
                                failed_count += 1
                        
                        # 進捗ログ
                        current_progress = successful_count + failed_count
                        if current_progress % 100 == 0:
                            logger.info(f"並列処理進捗: {current_progress}/{len(filtered_df)}行完了 "
                                      f"(成功: {successful_count}, 失敗: {failed_count})")
                        
                        # 中間保存（200件ごと）
                        if current_progress % 200 == 0:
                            self._save_intermediate_results(processed_data, current_progress)
                            if processed_data:
                                self._display_sample_data(processed_data[-5:], current_progress)
                        
                    except Exception as e:
                        logger.error(f"バッチ処理中にエラー: {e}")
                        failed_count += len(batch)
            
            # 最終結果の保存
            self._save_final_results(processed_data, filtered_df, successful_count, failed_count)
            
        except Exception as e:
            logger.error(f"並列データセット生成中にエラーが発生しました: {e}")
            if self.use_wandb:
                wandb.finish()
            raise

    def _process_batch_parallel(self, batch_data: List[Tuple[int, Dict]]) -> List[Optional[Dict]]:
        """並列処理用のバッチ処理（ThreadPoolExecutor用）"""
        results = []
        for idx, row_dict in batch_data:
            try:
                question = row_dict['question']
                type_of_harm = row_dict['type_of_harm']
                original_preferred = row_dict['preferred_output']
                
                # 新しいchosen応答を生成
                new_chosen = self.generate_chosen_response(question, type_of_harm)
                
                if new_chosen is None:
                    results.append(None)
                    continue
                
                # 元のpreferred_outputをrejectedとして使用
                new_rejected = original_preferred
                
                results.append({
                    'prompt': question,
                    'chosen': new_chosen,
                    'rejected': new_rejected,
                    'original_type_of_harm': type_of_harm,
                    'original_preferred': original_preferred
                })
                
            except Exception as e:
                logger.error(f"並列処理で行 {idx}の処理中にエラー: {e}")
                results.append(None)
        
        return results



    def _save_final_results(self, processed_data: List[Dict], filtered_df: pd.DataFrame, 
                           successful_count: int, failed_count: int):
        """最終結果の保存"""
        # 最終結果の保存
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
        
        # 最終サンプルデータの表示
        if processed_data:
            logger.info("=== 最終生成データサンプル ===")
            self._display_sample_data(processed_data[-5:], len(filtered_df), is_final=True)
        
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

    def _save_intermediate_results(self, processed_data: List[Dict], current_progress: int):
        """中間結果を保存"""
        if not processed_data:
            return
            
        # 中間保存ファイル名
        intermediate_file = self.output_dir / f"intermediate_results_{current_progress}.jsonl"
        
        # 中間結果を保存
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            for item in processed_data:
                output_item = {
                    'prompt': item['prompt'],
                    'chosen': item['chosen'],
                    'rejected': item['rejected']
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
        
        logger.info(f"中間結果を保存しました: {intermediate_file}")

    def _display_sample_data(self, sample_data: List[Dict], current_progress: int, is_final: bool = False):
        """サンプルデータを表示"""
        if not sample_data:
            return
            
        prefix = "最終" if is_final else f"進捗{current_progress}時点"
        logger.info(f"=== {prefix}サンプルデータ ===")
        
        for i, item in enumerate(sample_data, 1):
            logger.info(f"--- サンプル{i} ---")
            logger.info(f"質問: {item['prompt'][:100]}{'...' if len(item['prompt']) > 100 else ''}")
            logger.info(f"Chosen: {item['chosen'][:150]}{'...' if len(item['chosen']) > 150 else ''}")
            logger.info(f"Rejected: {item['rejected'][:150]}{'...' if len(item['rejected']) > 150 else ''}")
            logger.info(f"カテゴリ: {item['original_type_of_harm']}")
            logger.info("")

    def _display_statistics(self, processed_data: List[Dict], filtered_df: pd.DataFrame, current_progress: int):
        """統計情報を表示"""
        if not processed_data:
            return
            
        # カテゴリ別の統計
        category_stats = {}
        for item in processed_data:
            category = item['original_type_of_harm']
            if category not in category_stats:
                category_stats[category] = 0
            category_stats[category] += 1
        
        logger.info("=== カテゴリ別統計 ===")
        for category, count in category_stats.items():
            logger.info(f"  {category}: {count}件")
        
        # 応答長の統計
        chosen_lengths = [len(item['chosen']) for item in processed_data]
        rejected_lengths = [len(item['rejected']) for item in processed_data]
        
        logger.info("=== 応答長統計 ===")
        logger.info(f"  Chosen平均長: {sum(chosen_lengths)/len(chosen_lengths):.1f}文字")
        logger.info(f"  Rejected平均長: {sum(rejected_lengths)/len(rejected_lengths):.1f}文字")
        logger.info(f"  進捗: {current_progress}/{len(filtered_df)} ({current_progress/len(filtered_df)*100:.1f}%)")
        logger.info("")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DPOデータセットV2生成スクリプト')
    parser.add_argument('--sequential', action='store_true', 
                       help='シーケンシャル処理モード（並列処理を無効化）')
    parser.add_argument('--workers', type=int, default=None,
                       help='並列処理のワーカー数（デフォルト: CPU数、最大8）')
    parser.add_argument('--test', action='store_true',
                       help='テストモード（小規模データで動作確認）')
    
    args = parser.parse_args()
    
    try:
        # ワーカー数の設定
        if args.workers:
            max_workers = min(args.workers, 8)  # 最大8に制限
        else:
            max_workers = None  # デフォルト値を使用
        
        generator = DPOV2Generator(max_workers=max_workers)
        
        if args.test:
            logger.info("テストモードで実行します")
            # テスト用の小規模データ処理
            generator.generate_dataset(use_parallel=not args.sequential)
        else:
            # 本格実行
            generator.generate_dataset(use_parallel=not args.sequential)
        
        logger.info("DPOデータセットV2生成が正常に完了しました")
        
    except Exception as e:
        logger.error(f"スクリプト実行中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
