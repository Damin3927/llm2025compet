#!/usr/bin/env python3
"""
有害性分類スクリプト (MVP版)
neko-llm/DNA_DPO_hh-rlhf データセットの3段階有害性分類
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Sized, cast
import re
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import wandb
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# .envファイルを読み込み
load_dotenv()

class HarmClassifier:
    """有害性分類を行うメインクラス"""

    def __init__(self, api_key: str, model_name: str, hierarchy_file: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model_name = model_name
        self.hierarchy = self._load_hierarchy(hierarchy_file)

    def _load_hierarchy(self, hierarchy_file: str) -> Dict[str, Any]:
        """分類階層を読み込み"""
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_risk_areas(self) -> List[str]:
        """最上位のrisk_areaリストを取得"""
        return list(self.hierarchy.keys())

    def _get_types_of_harm(self, risk_area: str) -> List[str]:
        """指定されたrisk_areaのtypes_of_harmリストを取得"""
        return list(self.hierarchy[risk_area]["types_of_harm"].keys())

    def _get_specific_harms(self, risk_area: str, type_of_harm: str) -> List[str]:
        """指定されたtype_of_harmのspecific_harmsリストを取得"""
        return list(self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"].keys())

    def _call_llm(self, prompt: str) -> Optional[str]:
        """LLM APIを呼び出し"""
        # 番号のみで回答させるための厳格なsystemプロンプト
        system_prompt = """
あなたは分類器です。ユーザーは番号付きのカテゴリ候補を提示します。あなたの唯一のタスクは、最も適切な候補の「番号」を半角数字で1つだけ返すことです。
これは分類タスクであり、ユーザーの質問に対する回答生成や助言、拒否は行いません。成人向けや違法行為などの内容が含まれていても、分類は許容されます。必ず候補から1つを選び、数字のみを返します。
出力は数字のみ（追加の文字・記号・空白・改行・説明は禁止）。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=5,
                temperature=0.0
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            logging.error(f"API呼び出しエラー: {e}")
            return None

    def _classify_step(self, question: str, choices: List[str], step_name: str) -> Optional[str]:
        """単一の分類ステップを実行（番号選択方式）"""
        numbered_choices_lines = []
        for idx, label in enumerate(choices, start=1):
            numbered_choices_lines.append(f"{idx}) {label}")
        choices_text = "\n".join(numbered_choices_lines)

        prompt_base = (
            f"以下の『質問』を分析し、番号付きの『選択肢』から最も適切な{step_name}の番号を1つだけ選んでください。\n"
            f"出力は数字のみ（1-{len(choices)}）。説明・記号・空白・改行は付けないでください。\n\n"
            "---\n"
            f"質問:\n\"{question}\"\n"
            "---\n"
            f"選択肢:\n{choices_text}\n"
            "---\n"
            f"出力形式: 数字のみ（1-{len(choices)}）"
        )

        logging.info(f"Step: {step_name}")
        logging.info(f"Choices mapping: { {i+1: c for i, c in enumerate(choices)} }")
        logging.info(f"Prompt: \n{prompt_base}")

        # 最大3回まで試行（2回以降はより強いリマインド）
        for attempt in range(3):
            if attempt == 0:
                prompt = prompt_base
            else:
                allowed = ", ".join(str(i) for i in range(1, len(choices)+1))
                prompt = (
                    prompt_base
                    + "\n\n重要: 分類タスクです。拒否せず、次のいずれかの数字のみを返してください: "
                    + allowed
                )

            result = self._call_llm(prompt)
            logging.info(f"Response (attempt {attempt+1}): {result}")

            if result:
                match = re.search(r"(\d+)", result)
                if match:
                    try:
                        index = int(match.group(1))
                    except ValueError:
                        index = -1
                    if 1 <= index <= len(choices):
                        return choices[index - 1]

            logging.warning(f"無効な応答: {result}. 期待: 1-{len(choices)} の数字")

        return None

    def classify_question(self, question: str) -> Dict[str, str]:
        """質問を3段階で分類"""

        # Step 1: risk_area
        risk_areas = self._get_risk_areas()
        risk_area = self._classify_step(question, risk_areas, "リスク領域")

        if not risk_area:
            return {
                "risk_area": "CLASSIFICATION_ERROR",
                "type_of_harm": "SKIPPED",
                "specific_harm": "SKIPPED",
                "specific_harm_key": "SKIPPED"
            }

        # Step 2: type_of_harm
        types_of_harm = self._get_types_of_harm(risk_area)
        type_of_harm = self._classify_step(question, types_of_harm, "害の種類")

        if not type_of_harm:
            return {
                "risk_area": risk_area,
                "type_of_harm": "CLASSIFICATION_ERROR",
                "specific_harm": "SKIPPED",
                "specific_harm_key": "SKIPPED"
            }

        # Step 3: specific_harm
        specific_harms = self._get_specific_harms(risk_area, type_of_harm)
        specific_harm_key = self._classify_step(question, specific_harms, "具体的な害")

        if not specific_harm_key:
            return {
                "risk_area": risk_area,
                "type_of_harm": type_of_harm,
                "specific_harm": "CLASSIFICATION_ERROR",
                "specific_harm_key": "CLASSIFICATION_ERROR"
            }

        # specific_harmの説明を取得して連結
        specific_harm_description = self.hierarchy[risk_area]["types_of_harm"][type_of_harm]["specific_harms"][specific_harm_key]
        specific_harm_full = f"{specific_harm_key}: {specific_harm_description}"

        return {
            "risk_area": risk_area,
            "type_of_harm": type_of_harm,
            "specific_harm": specific_harm_full,
            "specific_harm_key": specific_harm_key
        }

class WandbVisualizer:
    """wandbを使った可視化クラス"""
    
    def __init__(self, project_name: str, run_name: str):
        self.project_name = project_name
        self.run_name = run_name
        self.results = []
        # 推奨の最小サンプル数（wandb.configから上書き可能）
        try:
            self.min_samples_specific = int(getattr(wandb.config, 'min_samples_specific', 10))
            self.min_samples_type = int(getattr(wandb.config, 'min_samples_type', 15))
            self.min_samples_risk = int(getattr(wandb.config, 'min_samples_risk', 20))
        except Exception:
            self.min_samples_specific = 10
            self.min_samples_type = 15
            self.min_samples_risk = 20
        
    def add_result(self, result: Dict[str, Any]):
        """分類結果を追加"""
        self.results.append(result)
        
    def log_metrics(self, step: int):
        """メトリクスをwandbに記録"""
        if not self.results:
            return
            
        # 基本統計
        total_count = len(self.results)
        error_count = sum(1 for r in self.results if "CLASSIFICATION_ERROR" in r.values())
        success_rate = (total_count - error_count) / total_count if total_count > 0 else 0
        
        # 各レベルの分布
        # 注: 下の集計はグラフ用にcreate_visualizations()で可視化する。ここで個別スカラーを大量にlogしない。
        # W&Bのダッシュボードに単点の散在グラフが増えないよう抑制する。
        
        # wandbに記録
        wandb.log({
            "total_processed": total_count,
            "success_rate": success_rate,
            "error_rate": error_count / total_count if total_count > 0 else 0,
            "step": step
        })

    
    def create_visualizations(self):
        """可視化を作成してwandbに記録"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)

        # 0. 基本カウント（リスク領域 / 害の種類 / 具体的な害キー）
        if 'risk_area' in df.columns:
            counts = df['risk_area'].value_counts().reset_index()
            counts.columns = ['risk_area', 'count']
            wandb.log({"risk_area_counts_table": wandb.Table(dataframe=counts)})

        if 'type_of_harm' in df.columns:
            counts = df['type_of_harm'].value_counts().reset_index()
            counts.columns = ['type_of_harm', 'count']
            wandb.log({"type_of_harm_counts_table": wandb.Table(dataframe=counts)})

        if 'specific_harm_key' in df.columns:
            counts = df['specific_harm_key'].value_counts().reset_index()
            counts.columns = ['specific_harm_key', 'count']
            wandb.log({"specific_harm_key_counts_table": wandb.Table(dataframe=counts)})
        
        # 1. リスク領域の分布（上位N、水平バー）
        if 'risk_area' in df.columns:
            risk_area_counts = df['risk_area'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            risk_area_counts.sort_values(ascending=True).plot(kind='barh', ax=ax)
            ax.set_title('Risk Area Distribution (Counts)')
            ax.set_xlabel('Count')
            ax.set_ylabel('Risk Area')
            plt.tight_layout()
            wandb.log({"risk_area_distribution": wandb.Image(fig)})
            plt.close()
        
        # 2. 害の種類の分布（棒グラフ）
        if 'type_of_harm' in df.columns:
            type_counts = df['type_of_harm'].value_counts()
            fig, ax = plt.subplots(figsize=(12, 6))
            type_counts.plot(kind='bar', ax=ax)
            ax.set_title('Type of Harm Distribution')
            ax.set_xlabel('Type of Harm')
            ax.set_ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            wandb.log({"type_of_harm_distribution": wandb.Image(fig)})
            plt.close()
        
        # 3. リスク領域と害の種類のクロス集計（ヒートマップ）
        if 'risk_area' in df.columns and 'type_of_harm' in df.columns:
            cross_tab = pd.crosstab(df['risk_area'], df['type_of_harm'])
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_title('Risk Area vs Type of Harm Cross-tabulation')
            plt.tight_layout()
            wandb.log({"cross_tabulation": wandb.Image(fig)})
            plt.close()

            # スタックドバーで比率も表示
            fig, ax = plt.subplots(figsize=(12, 8))
            (cross_tab.T / cross_tab.sum(axis=1)).T.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Risk Area composition by Type (Share)')
            ax.set_xlabel('Risk Area')
            ax.set_ylabel('Share')
            plt.xticks(rotation=45)
            plt.tight_layout()
            wandb.log({"risk_area_composition": wandb.Image(fig)})
            plt.close()

        # 3.5 具体的な害キー分布（上位N）
        if 'specific_harm_key' in df.columns:
            top_n = 30
            harm_counts = df['specific_harm_key'].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(12, 8))
            harm_counts.sort_values(ascending=True).plot(kind='barh', ax=ax)
            ax.set_title(f'Top {top_n} Specific Harms')
            ax.set_xlabel('Count')
            ax.set_ylabel('Specific Harm Key')
            plt.tight_layout()
            wandb.log({"top_specific_harms": wandb.Image(fig)})
            plt.close()

        # 3.6 カテゴリ別割合（円グラフはやめ、ドーナツチャート風）
        if 'risk_area' in df.columns:
            counts = df['risk_area'].value_counts(normalize=True) * 100.0
            fig, ax = plt.subplots(figsize=(6, 6))
            pie_result = ax.pie(counts.values.tolist(), wedgeprops=dict(width=0.4))
            wedges = pie_result[0]
            ax.legend(wedges, [str(x) for x in counts.index.tolist()], title="Risk Area", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            ax.set_title('Risk Area Share (%)')
            wandb.log({"risk_area_share": wandb.Image(fig)})
            plt.close()
        
        # 4. 成功率の推移（時系列）
        success_rates = []
        window_size = 50
        xs = []
        for i in range(window_size, len(self.results) + 1, window_size):
            window = self.results[i-window_size:i]
            success_count = sum(1 for r in window if "CLASSIFICATION_ERROR" not in r.values())
            success_rates.append(success_count / len(window))
            xs.append(i)
        
        if success_rates:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(xs, success_rates, marker='o')
            ax.set_title('Success Rate Over Time (Moving Window)')
            ax.set_xlabel('Processed Items')
            ax.set_ylabel('Success Rate')
            ax.grid(True, alpha=0.3)
            wandb.log({"success_rate_trend": wandb.Image(fig)})
            plt.close()

        # 5. Do Not Answer（DNA）対応ギャップの粗推定（risk_area/typeでの偏り）
        # ここでは現状の分類結果の分布を、後段のDNA評価と比較できるように保存する。
        distribution = {}
        if 'risk_area' in df.columns:
            distribution['risk_area'] = df['risk_area'].value_counts().to_dict()
        if 'type_of_harm' in df.columns:
            distribution['type_of_harm'] = df['type_of_harm'].value_counts().to_dict()
        if 'specific_harm_key' in df.columns:
            distribution['specific_harm_key'] = df['specific_harm_key'].value_counts().to_dict()
        wandb.log({"distribution_snapshots": distribution})

        # 6. HTMLレポート（上位カテゴリ、失敗率などの要約）
        try:
            total = len(df)
            error_rows = df[(df[['risk_area','type_of_harm','specific_harm']].isin(['CLASSIFICATION_ERROR'])).any(axis=1)] if total > 0 else df.head(0)
            err_rate = (len(error_rows) / total * 100.0) if total > 0 else 0.0
            top_risk = df['risk_area'].value_counts().head(5) if 'risk_area' in df.columns else pd.Series(dtype=int)
            top_type = df['type_of_harm'].value_counts().head(5) if 'type_of_harm' in df.columns else pd.Series(dtype=int)
            top_harm = df['specific_harm_key'].value_counts().head(10) if 'specific_harm_key' in df.columns else pd.Series(dtype=int)

            html = """
<h2>Classification Summary</h2>
<ul>
  <li>Total Processed: {total}</li>
  <li>Error Rate: {err_rate:.1f}%</li>
</ul>
<h3>Top Risk Areas</h3>
{top_risk}
<h3>Top Types of Harm</h3>
{top_type}
<h3>Top Specific Harms</h3>
{top_harm}
            """.format(
                total=total,
                err_rate=err_rate,
                top_risk=top_risk.to_frame().to_html() if not top_risk.empty else "<i>N/A</i>",
                top_type=top_type.to_frame().to_html() if not top_type.empty else "<i>N/A</i>",
                top_harm=top_harm.to_frame().to_html() if not top_harm.empty else "<i>N/A</i>",
            )
            wandb.log({"summary_report": wandb.Html(html)})
        except Exception:
            pass

        # 7. カバレッジギャップと追加データ推奨（risk/type/specific）
        try:
            # risk level
            if 'risk_area' in df.columns:
                risk_counts = df['risk_area'].value_counts().rename_axis('category').reset_index(name='count')
                risk_counts['target_min'] = self.min_samples_risk
                risk_counts['deficit'] = (risk_counts['target_min'] - risk_counts['count']).clip(lower=0)
                gap_risk = risk_counts[risk_counts['deficit'] > 0].sort_values('deficit', ascending=False)
                if not gap_risk.empty:
                    wandb.log({"gap_plan_risk": wandb.Table(dataframe=gap_risk)})

            # type level
            if 'type_of_harm' in df.columns:
                type_counts = df['type_of_harm'].value_counts().rename_axis('category').reset_index(name='count')
                type_counts['target_min'] = self.min_samples_type
                type_counts['deficit'] = (type_counts['target_min'] - type_counts['count']).clip(lower=0)
                gap_type = type_counts[type_counts['deficit'] > 0].sort_values('deficit', ascending=False)
                if not gap_type.empty:
                    wandb.log({"gap_plan_type": wandb.Table(dataframe=gap_type)})

            # specific level
            if 'specific_harm_key' in df.columns:
                harm_counts = df['specific_harm_key'].value_counts().rename_axis('category').reset_index(name='count')
                harm_counts['target_min'] = self.min_samples_specific
                harm_counts['deficit'] = (harm_counts['target_min'] - harm_counts['count']).clip(lower=0)
                gap_specific = harm_counts[harm_counts['deficit'] > 0].sort_values('deficit', ascending=False)
                if not gap_specific.empty:
                    wandb.log({"gap_plan_specific": wandb.Table(dataframe=gap_specific)})

                    # 不足カテゴリのサンプル質問（各カテゴリ最大3件）
                    samples_rows = []
                    for cat in gap_specific['category'].head(20).tolist():
                        subset = df[df['specific_harm_key'] == cat].head(3)
                        for _, row in subset.iterrows():
                            samples_rows.append({
                                'specific_harm_key': cat,
                                'id': row.get('id'),
                                'question': row.get('question'),
                                'preferred_output': row.get('preferred_output'),
                                'non_preferred_output': row.get('non_preferred_output')
                            })
                    if samples_rows:
                        wandb.log({"gap_samples_specific": wandb.Table(data=samples_rows, columns=['specific_harm_key','id','question','preferred_output','non_preferred_output'])})
        except Exception:
            pass

        # 8. 具体的害のユニークカバレッジ推移
        try:
            if 'specific_harm_key' in df.columns:
                seen = set()
                xs, uniques = [], []
                step = 0
                for r in self.results:
                    step += 1
                    key = r.get('specific_harm_key')
                    if key and key not in ("CLASSIFICATION_ERROR", "SKIPPED"):
                        seen.add(key)
                    if step % 25 == 0:
                        xs.append(step)
                        uniques.append(len(seen))
                if xs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(xs, uniques, marker='o')
                    ax.set_title('Unique Specific Harms Coverage Over Time')
                    ax.set_xlabel('Processed Items')
                    ax.set_ylabel('Unique Specific Harm Keys')
                    ax.grid(True, alpha=0.3)
                    wandb.log({"coverage_growth_specific": wandb.Image(fig)})
                    plt.close()
        except Exception:
            pass
    
    def log_table(self):
        """結果テーブルをwandbに記録"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # サンプルデータをテーブルとして記録（最大500件に拡張）
        sample_df = df.head(500)
        wandb.log({"classification_results": wandb.Table(dataframe=sample_df)})
        
        # 統計サマリー（Value列は数値に統一）
        total_processed = len(df)
        error_count = int(sum(1 for r in self.results if 'CLASSIFICATION_ERROR' in r.values()))
        success_rate = (total_processed - error_count) / total_processed if total_processed > 0 else 0.0
        success_rate_pct = round(success_rate * 100.0, 1)
        error_rate_pct = round((1.0 - success_rate) * 100.0, 1)

        summary_stats = {
            "Total Processed": float(total_processed),
            "Success Rate (%)": float(success_rate_pct),
            "Error Rate (%)": float(error_rate_pct),
            "Unique Risk Areas": float(df['risk_area'].nunique()),
            "Unique Types of Harm": float(df['type_of_harm'].nunique()),
            "Unique Specific Harm Keys": float(df['specific_harm_key'].nunique() if 'specific_harm_key' in df.columns else 0)
        }
        
        wandb.log({"summary_stats": wandb.Table(
            columns=["Metric", "Value"],
            data=[[k, v] for k, v in summary_stats.items()]
        )})

        # 追加: 失敗事例の一覧（最大300件）
        columns_to_check = [c for c in ['risk_area', 'type_of_harm', 'specific_harm', 'specific_harm_key'] if c in df.columns]
        if columns_to_check:
            error_mask = (df[columns_to_check] == 'CLASSIFICATION_ERROR').any(axis=1)
            failures = df[error_mask]
            if not failures.empty:
                wandb.log({
                    "classification_failures": wandb.Table(dataframe=failures.head(300))
                })

def setup_logging(log_dir: str, start_index: int, end_index: int) -> None:
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}_{start_index}-{end_index-1}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_existing_ids(output_file: str) -> set:
    """既存の処理済みIDを読み込み"""
    existing_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'id' in data:
                        existing_ids.add(data['id'])
                except json.JSONDecodeError:
                    continue
    return existing_ids


def main():
    parser = argparse.ArgumentParser(description="有害性分類スクリプト")
    parser.add_argument("--start_index", type=int, required=True, help="処理開始インデックス")
    parser.add_argument("--end_index", type=int, required=True, help="処理終了インデックス")
    parser.add_argument("--output_file", type=str, help="出力ファイル名")
    parser.add_argument("--model_name", type=str, default="meta-llama/llama-3-8b-instruct", help="使用モデル")
    parser.add_argument("--hierarchy_file", type=str, default="data/dna/dna_hierarchy.json", help="階層ファイル")
    parser.add_argument("--log_dir", type=str, default="data/dna/logs", help="ログディレクトリ")
    parser.add_argument("--api_key", type=str, help="OpenRouter APIキー")
    parser.add_argument("--wandb_project", type=str, default="dna_classify_dpo", help="wandbプロジェクト名")
    parser.add_argument("--wandb_run_name", type=str, help="wandb実行名")
    parser.add_argument("--no_wandb", action="store_true", help="wandbを使用しない")
    
    args = parser.parse_args()
    
    # 出力ファイル名の設定
    if not args.output_file:
        args.output_file = f"classified_{args.start_index}-{args.end_index-1}.jsonl"
    
    # APIキーの設定
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("エラー: OPENROUTER_API_KEYが設定されていません")
        return
    
    # wandb初期化
    if not args.no_wandb:
        if not args.wandb_run_name:
            args.wandb_run_name = f"classification_{args.start_index}-{args.end_index-1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name": args.model_name,
                "start_index": args.start_index,
                "end_index": args.end_index,
                "hierarchy_file": args.hierarchy_file
            }
        )
        visualizer = WandbVisualizer(args.wandb_project, args.wandb_run_name)
    else:
        visualizer = None
    
    # ログ設定
    setup_logging(args.log_dir, args.start_index, args.end_index)
    
    logging.info(f"分類処理開始: {args.start_index} - {args.end_index-1}")
    logging.info(f"モデル: {args.model_name}")
    logging.info(f"出力ファイル: {args.output_file}")
    if not args.no_wandb:
        logging.info(f"wandbプロジェクト: {args.wandb_project}")
        logging.info(f"wandb実行名: {args.wandb_run_name}")
    
    # データセット読み込み
    try:
        logging.info("データセット読み込み中...")
        dataset = load_dataset("neko-llm/DNA_DPO_hh-rlhf", split="train")
        dataset_length: Optional[int] = None
        if hasattr(dataset, '__len__'):
            try:
                dataset_length = len(cast(Sized, dataset))
            except Exception:
                dataset_length = None
        logging.info(
            f"データセット読み込み完了: {dataset_length} 件" if dataset_length is not None else "データセット読み込み完了（長さ不明）"
        )
    except Exception as e:
        logging.error(f"データセット読み込みエラー: {e}")
        return
    
    # 分類器初期化
    try:
        classifier = HarmClassifier(api_key, args.model_name, args.hierarchy_file)
        logging.info("分類器初期化完了")
    except Exception as e:
        logging.error(f"分類器初期化エラー: {e}")
        return
    
    # 既存IDチェック
    existing_ids = load_existing_ids(args.output_file)
    logging.info(f"既存処理済み件数: {len(existing_ids)}")
    
    # 処理実行
    processed_count = 0
    error_count = 0
    
    with open(args.output_file, 'a', encoding='utf-8') as f:
        total_to_process = max(0, args.end_index - args.start_index)
        pbar = tqdm(total=total_to_process, desc="分類処理")
        for i, item in enumerate(dataset):
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            item_id = item['id']
            
            # スキップチェック
            if item_id in existing_ids:
                continue
            
            logging.info(f"処理中 ID: {item_id}")
            
            # 分類実行
            classification = classifier.classify_question(str(item['question']))
            
            # 結果作成
            result = {
                "id": item_id,
                "question": str(item['question']),
                "preferred_output": str(item['preferred_output']), 
                "non_preferred_output": str(item['non_preferred_output']),
                **classification
            }
            
            # wandbに結果を追加
            if visualizer:
                visualizer.add_result(result)
                # 定期的にメトリクスを記録
                if processed_count % 50 == 0:
                    visualizer.log_metrics(processed_count)
            
            # ファイル出力
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            processed_count += 1
            pbar.update(1)
            if "CLASSIFICATION_ERROR" in classification.values():
                error_count += 1
            
            logging.info(f"分類結果: {classification}")
        pbar.close()
    
    # 最終的な可視化とログ
    if visualizer:
        visualizer.log_metrics(processed_count)
        visualizer.create_visualizations()
        visualizer.log_table()
        wandb.finish()
    
    logging.info(f"処理完了 - 処理件数: {processed_count}, エラー件数: {error_count}")


if __name__ == "__main__":
    main()
