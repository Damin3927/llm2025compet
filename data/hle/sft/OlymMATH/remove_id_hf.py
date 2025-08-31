from datasets import load_dataset

# Hugging FaceリポジトリID
repo_id = "tentatani/HLE_SFT_OlymMATH" 
# 削除したいデータのIDリスト
ids_to_remove = [81,82,83,84,85,87,88,89] 

# 1. Hugging Face Hubからデータセットを読み込む
print(f"'{repo_id}' からデータセットを読み込んでいます...")
dataset = load_dataset(repo_id)

# 2. 'train'スプリットのデータをフィルタリングする
print(f"ID {ids_to_remove} のデータを削除しています...")
# `filter`関数を使い、削除したいIDリストに含まれていないデータだけを残す
filtered_dataset = dataset.filter(lambda example: example['id'] not in ids_to_remove)

# 3. フィルタリング後のデータセットをHubにプッシュ（上書き）する
print(f"更新されたデータセットを '{repo_id}' にプッシュしています...")
filtered_dataset.push_to_hub(repo_id)

print("\n✅ 処理が完了しました！")
print(f"元のデータ数: {len(dataset['train'])}")
print(f"削除後のデータ数: {len(filtered_dataset['train'])}")
