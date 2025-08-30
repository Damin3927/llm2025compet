from datasets import load_dataset

# 元のデータセットのリポジトリID
source_repo_id = "tentatani/HLE_SFT_OlymMATH"

# アップロード先のリポジトリID
target_repo_id = "neko-llm/HLE_SFT_OlymMATH" 

# 1. 元のデータセットを読み込む
print(f"'{source_repo_id}' からデータセットを読み込んでいます...")
dataset = load_dataset(source_repo_id)

# 2. 読み込んだデータセットを新しい場所にプッシュする
print(f"データセットを '{target_repo_id}' にプッシュしています...")
# この処理により、neko-llmアカウント上に新しいデータセットリポジトリが作成されます
dataset.push_to_hub(target_repo_id)

print(f"\n✅ 複製が完了しました！")
print(f"https://huggingface.co/datasets/{target_repo_id} を確認してください。")
