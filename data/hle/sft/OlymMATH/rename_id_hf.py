from datasets import load_dataset

# Hugging FaceリポジトリID
repo_id = "tentatani/HLE_SFT_OlymMATH"

# 1. Hubから現在のデータセットを読み込む
print(f"'{repo_id}' からデータセットを読み込んでいます...")
dataset = load_dataset(repo_id)

# 2. map() を使ってIDを0からの連番に振り直す
print("IDを0からの連番に振り直しています...")
# with_indices=Trueで、各データのインデックス(i)を取得できるようにする
# example（元のデータ）と i（インデックス）を受け取り、
# exampleの'id'をiで上書きした新しいexampleを返す
updated_dataset = dataset.map(
    lambda example, i: {'id': i}, 
    with_indices=True
)

# 3. IDを振り直したデータセットをHubにプッシュ（上書き）する
print(f"更新されたデータセットを '{repo_id}' にプッシュしています...")
updated_dataset.push_to_hub(repo_id)

print("\n✅ IDの振り直しが完了しました！")
print("Hugging Faceのページをリロードして、IDが0から始まっていることを確認してください。")
