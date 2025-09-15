import os
import sys
from datasets import load_dataset, Dataset

# --- 設定 ---
TARGET_REPO_ID = "neko-llm/HLE_RL_OlympiadBench"
LOCAL_UPDATES_FILE = "output.jsonl"

# --- スクリプト本体 ---
def direct_update_dataset():
    """
    元のデータセットを直接ダウンロードし、ローカルの更新内容とマージして、
    同じリポジトリに再度アップロードする。
    """
    
    try:
        from huggingface_hub import HfFolder
        if not HfFolder.get_token():
            print("❌ Hugging Faceトークンが見つかりません。まず 'huggingface-cli login' を実行してください。", file=sys.stderr)
            sys.exit(1)
    except ImportError:
        print("❌ 'huggingface_hub'ライブラリが見つかりません。'pip install huggingface_hub' を実行してください。", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.exists(LOCAL_UPDATES_FILE):
        print(f"❌ ローカルファイル '{LOCAL_UPDATES_FILE}' が見つかりません。", file=sys.stderr)
        sys.exit(1)

    print(f"Step 1/4: 元のデータセット '{TARGET_REPO_ID}' を読み込んでいます...")
    try:
        original_dataset = load_dataset(TARGET_REPO_ID, split="train")
        print(f"✅ 元のデータセットには {len(original_dataset)} 件のデータがあります。")
    except Exception as e:
        print(f"❌ データセットの読み込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Step 2/4: ローカルの更新ファイル '{LOCAL_UPDATES_FILE}' を読み込んでいます...")
    updates_dataset = load_dataset("json", data_files=LOCAL_UPDATES_FILE, split="train")
    updates_dict = {item['id']: item for item in updates_dataset}
    print(f"✅ 更新ファイルには {len(updates_dict)} 件のデータがあります。")

    # ★★★ ここが修正箇所 ★★★
    def update_row(example):
        """データセットの各行を更新するための関数"""
        example_id = example['id']
        # 更新対象の行の場合のみ、ファイルの内容で更新する
        if example_id in updates_dict:
            update_data = updates_dict[example_id]
            example['output'] = update_data.get('output')
            example['answer_CoT'] = update_data.get('answer_CoT')
            example['is_correct'] = update_data.get('is_correct')
        # elseブロックを削除したため、更新対象でない行は何も変更されない
        return example

    print("Step 3/4: データセットをマージしています...")
    updated_dataset = original_dataset.map(update_row)
    print("✅ マージが完了しました。")
    
    print(f"Step 4/4: 更新されたデータセットを '{TARGET_REPO_ID}' に直接アップロードしています...")
    try:
        updated_dataset.push_to_hub(
            TARGET_REPO_ID,
            commit_message=f"Update dataset with new CoT results from {LOCAL_UPDATES_FILE}"
        )
        print("\n🎉 アップロードが成功しました！")
        print(f"こちらでデータセットを確認できます: https://huggingface.co/datasets/{TARGET_REPO_ID}")
    except Exception as e:
        print(f"\n❌ アップロード中にエラーが発生しました: {e}", file=sys.stderr)
        print("ヒント: あなたのアカウントが'neko-llm'組織のメンバーであり、書き込み権限を持っているか再確認してください。")

if __name__ == "__main__":
    direct_update_dataset()