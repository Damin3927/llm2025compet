import os
import sys
import json
import argparse
from huggingface_hub import HfApi, HfFolder, upload_file, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

def upload_and_merge_results(local_filepath, repo_id):
    """
    ローカルの結果ファイルをHugging Face Hubにマージしてアップロードする
    """
    print(f"\n🚀 Preparing to upload and merge results to Hugging Face Hub repository: {repo_id}...")
    
    # Hugging Faceにログインしているか確認
    if not HfFolder.get_token():
        print("❌ Hugging Face token not found. Please run 'huggingface-cli login' first.", file=sys.stderr)
        sys.exit(1)

    # ローカルファイルが存在するか確認
    if not os.path.exists(local_filepath):
        print(f"❌ Local results file '{local_filepath}' not found. Nothing to upload.", file=sys.stderr)
        sys.exit(1)

    remote_data = {}
    try:
        # リポジトリから既存のファイルをダウンロード
        downloaded_path = hf_hub_download(repo_id=repo_id, filename="results.jsonl", repo_type="dataset")
        print(f"✅ Downloaded existing data from '{repo_id}'.")
        with open(downloaded_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    remote_data[item['id']] = item
                except (json.JSONDecodeError, KeyError):
                    continue
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print("🟡 No existing 'results.jsonl' found on the Hub. A new file will be created.")
        else:
            print(f"❌ Error downloading from the Hub: {e}. Aborting upload.", file=sys.stderr)
            return
    except Exception as e:
        print(f"❌ An unexpected error occurred during download: {e}. Aborting upload.", file=sys.stderr)
        return

    # ローカルのデータを読み込み、リモートデータとマージ
    update_count, new_count = 0, 0
    with open(local_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                item_id = item['id']
                if item_id in remote_data:
                    update_count += 1
                else:
                    new_count += 1
                remote_data[item_id] = item
            except (json.JSONDecodeError, KeyError):
                continue

    if not remote_data:
        print("🟡 No data to upload.")
        return
        
    print(f"🔄 Merging results: {new_count} new entries, {update_count} updated entries.")
    
    merged_filepath = "merged_results_for_upload.jsonl"
    sorted_items = sorted(remote_data.values(), key=lambda x: x['id'])
    with open(merged_filepath, 'w', encoding='utf-8') as f:
        for item in sorted_items:
            f.write(json.dumps(item) + '\n')

    try:
        api = HfApi()
        # リポジトリをPublicで作成（既に存在する場合は何もしない）
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        
        print(f"☁️ Uploading merged file to '{repo_id}'...")
        upload_file(
            path_or_fileobj=merged_filepath,
            path_in_repo="results.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Upload successful! View your merged dataset at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face Hub: {e}", file=sys.stderr)
    finally:
        # 一時ファイルを削除
        if os.path.exists(merged_filepath):
            os.remove(merged_filepath)

def main():
    parser = argparse.ArgumentParser(description='Upload and merge results.jsonl to a Hugging Face Hub dataset.')
    parser.add_argument('--local_file', type=str, default='results.jsonl', help='Path to the local results file.')
    parser.add_argument('--hf_repo', type=str, default='neko-llm/HLE_RL_OlympiadBench', help='Hugging Face repository ID to upload to.')
    args = parser.parse_args()
    
    upload_and_merge_results(local_filepath=args.local_file, repo_id=args.hf_repo)

if __name__ == "__main__":
    main()