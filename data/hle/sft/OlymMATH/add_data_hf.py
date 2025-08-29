from datasets import load_dataset, concatenate_datasets, DatasetDict

# Hugging FaceリポジトリID
repo_id = "tentatani/HLE_SFT_OlymMATH"
# ローカルにある「追加したい」新しいデータファイルへのパス
new_data_file_path = "/Users/tentatani/uv_project/evaluate_cot/dataset_8.jsonl" 

# 1. Hubから既存のデータセットを読み込む
print(f"'{repo_id}' から既存のデータを読み込んでいます...")
existing_dataset = load_dataset(repo_id)

# 2. ローカルから追加したい新しいデータを読み込む
print(f"'{new_data_file_path}' から新しいデータを読み込んでいます...")
new_data = load_dataset("json", data_files=new_data_file_path)

# 3. 既存のデータと新しいデータを結合する
print("データを結合しています...")
# 'train' スプリット同士を結合
combined_dataset = concatenate_datasets([existing_dataset['train'], new_data['train']])

# 結合したものを再度 'train' スプリットとしてDatasetDictに格納する
final_dataset = DatasetDict({
    'train': combined_dataset
})

# 4. 結合後の完全なデータセットをHubにプッシュ（上書き）する
print(f"結合後のデータを '{repo_id}' にプッシュしています...")
final_dataset.push_to_hub(repo_id)

print("\n🚀 データ追加処理が完了しました！")
print(f"元のデータ数: {len(existing_dataset['train'])}")
print(f"追加したデータ数: {len(new_data['train'])}")
print(f"合計データ数: {len(final_dataset['train'])}")
