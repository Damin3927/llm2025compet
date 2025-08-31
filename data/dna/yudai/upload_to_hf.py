#!/usr/bin/env python3
"""
Hugging Faceデータセットアップロードスクリプト
thinkタグ付きのDNA_DPO_hh-rlhfデータセットをHugging Faceにアップロード

このスクリプトは、既存のデータセットを更新する形でthinkタグ付きデータをアップロードします。
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 依存関係のインポート
try:
    from datasets import Dataset, DatasetDict, Features, Value, Sequence
    from huggingface_hub import HfApi, login
    DATASETS_AVAILABLE = True
    HF_HUB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 必要なライブラリがインストールされていません: {e}")
    print("pip install datasets huggingface-hub でインストールしてください")
    DATASETS_AVAILABLE = False
    HF_HUB_AVAILABLE = False

# .envファイルの読み込み
load_dotenv()


class HuggingFaceUploader:
    """Hugging Faceデータセットアップロードを行うクラス"""
    
    def __init__(self, api_key: str, dataset_name: str):
        if not HF_HUB_AVAILABLE:
            raise ImportError("huggingface_hubライブラリが必要です")
        
        self.api_key = api_key
        self.dataset_name = dataset_name
        self.api = HfApi()
        
        # Hugging Faceにログイン
        try:
            login(token=api_key)
            logging.info("Hugging Faceにログインしました")
        except Exception as e:
            logging.error(f"Hugging Faceログインエラー: {e}")
            raise
    
    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """JSONLファイルを読み込み"""
        data = []
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        
        logging.info(f"ファイル読み込み中: {file_path}")
        
        with file_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    logging.warning(f"Line {line_num}: JSON解析エラー - {e}")
                    continue
        
        logging.info(f"読み込み完了: {len(data)}件")
        return data
    
    def _merge_think_tagged_files(self, file_patterns: List[str]) -> List[Dict[str, Any]]:
        """複数のthinkタグ付きファイルをマージ"""
        all_data = []
        
        for pattern in file_patterns:
            # パターンマッチングでファイルを検索
            if "*" in pattern:
                import glob
                matching_files = glob.glob(pattern)
            else:
                matching_files = [pattern] if Path(pattern).exists() else []
            
            for file_path in matching_files:
                try:
                    data = self._load_jsonl_file(file_path)
                    all_data.extend(data)
                    logging.info(f"ファイル {file_path} から {len(data)}件を追加")
                except Exception as e:
                    logging.error(f"ファイル {file_path} の読み込みエラー: {e}")
                    continue
        
        # IDでソート（IDが存在する場合）
        if all_data and 'id' in all_data[0]:
            try:
                all_data.sort(key=lambda x: x.get('id', 0))
                logging.info("データをIDでソートしました")
            except Exception as e:
                logging.warning(f"IDソートエラー: {e}")
        
        return all_data
    
    def _validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """データの検証と統計情報を返す"""
        if not data:
            return {"valid": False, "error": "データが空です"}
        
        validation = {
            "valid": True,
            "total_count": len(data),
            "fields": {},
            "think_tag_stats": {
                "preferred_with_think": 0,
                "preferred_without_think": 0,
                "non_preferred_with_think": 0,
                "non_preferred_without_think": 0
            },
            "errors": []
        }
        
        # フィールドの存在チェック
        required_fields = ["question", "preferred_output", "non_preferred_output"]
        for field in required_fields:
            validation["fields"][field] = 0
        
        # 各アイテムの検証
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                validation["errors"].append(f"Item {i}: 辞書形式ではありません")
                continue
            
            # 必須フィールドのチェック
            for field in required_fields:
                if field in item and item[field]:
                    validation["fields"][field] += 1
            
            # thinkタグのチェック
            preferred_output = item.get("preferred_output", "")
            non_preferred_output = item.get("non_preferred_output", "")
            
            if "<think>" in preferred_output and "</think>" in preferred_output:
                validation["think_tag_stats"]["preferred_with_think"] += 1
            else:
                validation["think_tag_stats"]["preferred_without_think"] += 1
            
            if "<think>" in non_preferred_output and "</think>" in non_preferred_output:
                validation["think_tag_stats"]["non_preferred_with_think"] += 1
            else:
                validation["think_tag_stats"]["non_preferred_without_think"] += 1
        
        # 検証結果の判定
        if validation["think_tag_stats"]["preferred_without_think"] > 0:
            validation["errors"].append("preferred_outputにthinkタグが不足しているアイテムがあります")
        
        if validation["fields"]["question"] < len(data) * 0.9:  # 90%以上
            validation["errors"].append("questionフィールドが不足しているアイテムがあります")
        
        if validation["errors"]:
            validation["valid"] = False
        
        return validation
    
    def _create_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """データからHugging Faceデータセットを作成"""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasetsライブラリが必要です")
        
        logging.info("データセット作成中...")
        
        # データセットの作成
        dataset = Dataset.from_list(data)
        
        logging.info(f"データセット作成完了: {len(dataset)}件")
        return dataset
    
    def _upload_dataset(self, dataset: Dataset, commit_message: str) -> bool:
        """データセットをHugging Faceにアップロード"""
        try:
            logging.info(f"データセットアップロード中: {self.dataset_name}")
            logging.info(f"コミットメッセージ: {commit_message}")
            
            # データセットをプッシュ
            dataset.push_to_hub(
                self.dataset_name,
                commit_message=commit_message,
                private=False  # パブリックデータセット
            )
            
            logging.info("データセットアップロード完了")
            return True
            
        except Exception as e:
            logging.error(f"データセットアップロードエラー: {e}")
            return False
    
    def upload_think_tagged_data(self, file_patterns: List[str], commit_message: str = None) -> bool:
        """thinkタグ付きデータをアップロード"""
        try:
            # ファイルのマージ
            logging.info("ファイルマージ開始...")
            data = self._merge_think_tagged_files(file_patterns)
            
            if not data:
                logging.error("マージされたデータが空です")
                return False
            
            # データの検証
            logging.info("データ検証中...")
            validation = self._validate_data(data)
            
            if not validation["valid"]:
                logging.error("データ検証に失敗しました:")
                for error in validation["errors"]:
                    logging.error(f"  - {error}")
                return False
            
            # 統計情報の表示
            logging.info("データ統計:")
            logging.info(f"  総件数: {validation['total_count']}")
            logging.info(f"  フィールド統計: {validation['fields']}")
            logging.info(f"  Thinkタグ統計: {validation['think_tag_stats']}")
            
            # データセットの作成
            dataset = self._create_dataset(data)
            
            # コミットメッセージの生成
            if not commit_message:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Update dataset with think tags - {timestamp}"
            
            # アップロード
            return self._upload_dataset(dataset, commit_message)
            
        except Exception as e:
            logging.error(f"アップロード処理エラー: {e}")
            return False


def setup_logging(log_level: str = "INFO") -> None:
    """ログ設定"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Faceデータセットアップロードスクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python upload_to_hf.py --files "think_tagged_*.jsonl"
  python upload_to_hf.py --files "think_tagged_0-9999.jsonl" "think_tagged_10000-19999.jsonl"
  python upload_to_hf.py --files "think_tagged_*.jsonl" --commit-message "Custom commit message"
        """
    )
    
    parser.add_argument("--files", nargs="+", required=True,
                       help="アップロードするファイルパターン（glob対応）")
    parser.add_argument("--dataset-name", type=str,
                       default="neko-llm/dna_dpo_hh-rlhf",
                       help="アップロード先のデータセット名")
    parser.add_argument("--commit-message", type=str,
                       help="コミットメッセージ（省略時は自動生成）")
    parser.add_argument("--api-key", type=str,
                       help="Hugging Face APIキー（環境変数より優先）")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="ログレベル")
    parser.add_argument("--dry-run", action="store_true",
                       help="実際のアップロードは行わず、検証のみ実行")
    
    args = parser.parse_args()
    
    # ログ設定
    setup_logging(args.log_level)
    
    # APIキーの設定
    api_key = args.api_key or os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key:
        logging.error("HUGGINGFACE_API_KEYが設定されていません")
        print("エラー: Hugging Face APIキーが設定されていません")
        print("以下のいずれかの方法で設定してください:")
        print("1. --api-key コマンドライン引数で指定")
        print("2. 環境変数 HUGGINGFACE_API_KEY で設定")
        print("3. .envファイルで設定")
        return 1
    
    # 依存関係のチェック
    if not DATASETS_AVAILABLE or not HF_HUB_AVAILABLE:
        logging.error("必要なライブラリがインストールされていません")
        print("エラー: 必要なライブラリがインストールされていません")
        print("pip install datasets huggingface-hub でインストールしてください")
        return 1
    
    # ファイルの存在チェック
    existing_files = []
    for pattern in args.files:
        if "*" in pattern:
            import glob
            matching_files = glob.glob(pattern)
        else:
            matching_files = [pattern] if Path(pattern).exists() else []
        
        if matching_files:
            existing_files.extend(matching_files)
        else:
            logging.warning(f"パターン '{pattern}' にマッチするファイルが見つかりません")
    
    if not existing_files:
        logging.error("アップロード対象のファイルが見つかりません")
        return 1
    
    logging.info(f"アップロード対象ファイル: {len(existing_files)}件")
    for file_path in existing_files:
        logging.info(f"  - {file_path}")
    
    # アップローダーの初期化
    try:
        uploader = HuggingFaceUploader(api_key, args.dataset_name)
        logging.info(f"アップローダー初期化完了: {args.dataset_name}")
    except Exception as e:
        logging.error(f"アップローダー初期化エラー: {e}")
        return 1
    
    # ドライランまたは実際のアップロード
    if args.dry_run:
        logging.info("=== ドライラン実行 ===")
        try:
            # ファイルのマージと検証のみ実行
            data = uploader._merge_think_tagged_files(args.files)
            validation = uploader._validate_data(data)
            
            logging.info("=== 検証結果 ===")
            logging.info(f"総件数: {validation['total_count']}")
            logging.info(f"フィールド統計: {validation['fields']}")
            logging.info(f"Thinkタグ統計: {validation['think_tag_stats']}")
            
            if validation["errors"]:
                logging.warning("検証エラー:")
                for error in validation["errors"]:
                    logging.warning(f"  - {error}")
            else:
                logging.info("検証成功: アップロード可能です")
            
        except Exception as e:
            logging.error(f"ドライランエラー: {e}")
            return 1
        
        logging.info("ドライラン完了")
        return 0
    
    # 実際のアップロード実行
    logging.info("=== アップロード開始 ===")
    
    success = uploader.upload_think_tagged_data(args.files, args.commit_message)
    
    if success:
        logging.info("アップロードが正常に完了しました")
        print(f"\n✅ アップロード完了!")
        print(f"データセット: https://huggingface.co/datasets/{args.dataset_name}")
        return 0
    else:
        logging.error("アップロードに失敗しました")
        print("\n❌ アップロード失敗")
        return 1


if __name__ == "__main__":
    exit(main())
