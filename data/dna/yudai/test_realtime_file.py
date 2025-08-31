#!/usr/bin/env python3
"""
リアルタイムファイル生成のテストスクリプト
"""

import os
import json
import time
from pathlib import Path

# 出力ディレクトリの作成 - 絶対パスを使用
output_dir = Path("/home/argo/llm2025compet/data/dna/processed")
output_dir.mkdir(parents=True, exist_ok=True)

def test_realtime_file():
    """リアルタイムファイルのテスト"""
    realtime_file = output_dir / "realtime_results.jsonl"
    
    print(f"現在の作業ディレクトリ: {os.getcwd()}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"リアルタイムファイル: {realtime_file}")
    print(f"ファイルが存在するか: {realtime_file.exists()}")
    
    # ファイルが存在しない場合は作成
    if not realtime_file.exists():
        try:
            with open(realtime_file, 'w', encoding='utf-8') as f:
                f.write("# DPOデータセットV2 リアルタイム結果\n")
                f.write(f"# 開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("# 形式: prompt, chosen, rejected, type_of_harm, timestamp, sequence_number\n")
                f.flush()
                os.fsync(f.fileno())
            
            print(f"リアルタイムファイルを初期化しました: {realtime_file}")
        except Exception as e:
            print(f"リアルタイムファイル初期化エラー: {e}")
    else:
        print(f"リアルタイムファイルが既に存在します: {realtime_file}")
    
    # テストデータを追加
    test_data = {
        'prompt': 'テスト質問です',
        'chosen': 'テスト回答です',
        'rejected': 'テスト拒否回答です',
        'type_of_harm': 'テストカテゴリ',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sequence_number': 1
    }
    
    try:
        with open(realtime_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_data, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        
        print(f"テストデータを追加しました: {test_data}")
    except Exception as e:
        print(f"テストデータ追加エラー: {e}")
    
    # ファイルの内容を確認
    if realtime_file.exists():
        print(f"ファイルサイズ: {realtime_file.stat().st_size} バイト")
        with open(realtime_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print("ファイル内容:")
            print(content)

if __name__ == "__main__":
    test_realtime_file()
