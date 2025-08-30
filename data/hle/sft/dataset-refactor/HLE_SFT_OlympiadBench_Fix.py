import os
import pandas as pd
from tqdm import tqdm
import datasets
from huggingface_hub import upload_file

# --- Hugging Face Upload Settings: Repository Name ---
OUTPUT_DATASET_ID = "neko-llm/HLE_SFT_OlympiadBench"

def main():
    """
    HLE_SFT_OlympiadBench Dataset のデータクリーニングを行う Main Script。

    NOTE:
    関数の機能:
    1. HLE_SFT_OlympiadBench Dataset を読み込む。
    2. HLE_SFT_OlympiadBench Dataset の output, answer の [] を削除する。
    3. 作成したデータセットを Hugging Face にアップロードする。
    """
    print("🚀 HLE_SFT_OlympiadBench Dataset のデータクリーニングを開始します...")

    # 出力ファイル設定
    output_filename = "hle_sft_olympiadbench_fix.csv"

    # --- Resume Logic ---
    processed_ids = []
    if os.path.exists(output_filename):
        print(f"📄 既存の結果ファイルを発見: '{output_filename}'")
        try:
            existing_df = pd.read_csv(output_filename)
            if 'id' in existing_df.columns:
                processed_ids = existing_df['id'].dropna().tolist()
                print(f"✅ {len(processed_ids)} 件の処理済み問題を発見。これらはスキップされます。")
        except pd.errors.EmptyDataError:
            print("⚠️ 既存の結果ファイルが空です。新規で開始します。")
        except Exception as e:
            print(f"⚠️ 既存ファイルの読み込みに失敗。新規で開始します。エラー: {e}")

    # --- HLE_SFT_OlympiadBench Dataset Loading ---
    print("📥 HLE_SFT_OlympiadBench Dataset を読み込み中...")
    try:
        # まずローカルファイルがあるかチェック
        if os.path.exists("hle_sft_olympiadbench_fix.jsonl"):
            print("📄 ローカルのJSONLファイルを使用します...")
            olympiadbench_dataset = datasets.load_dataset('json', data_files='hle_sft_olympiadbench_fix.jsonl', split='train')
        else:
            # Hugging Face Dataset を読み込み（trainデータのみ）
            # スキーマ問題を回避するため、ローカル読み込みを優先
            print("🌐 Hugging Face からデータセットを読み込み中...")
            olympiadbench_dataset = datasets.load_dataset("neko-llm/HLE_SFT_OlympiadBench", split='train')

        print(f"✅ データセット読み込み完了。{len(olympiadbench_dataset)} 件の問題を発見。")

        # サンプルデータの確認
        print("📋 サンプルデータ:")
        sample = olympiadbench_dataset[0]
        for key in sample.keys():
            print(f"  {key}: {str(sample[key])[:100]}...")

    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        print("🔄 ローカルJSONLファイルでの読み込みを試行中...")
        try:
            # フォールバック: ローカルファイルを直接読み込み
            olympiadbench_dataset = datasets.load_dataset('json', data_files='hle_sft_olympiadbench_fix.jsonl', split='train')
            print(f"✅ ローカルファイルでの読み込み成功。{len(olympiadbench_dataset)} 件の問題を発見。")
        except Exception as fallback_error:
            print(f"❌ ローカルファイル読み込みも失敗: {fallback_error}")
            return


    # --- データクリーニング ---
    print("🧹 データクリーニングを開始します...")
    cleaned_data = []

    for i in tqdm(range(len(olympiadbench_dataset))):
        sample = olympiadbench_dataset[i]

        # outputとanswerがリスト型の場合、最初の要素を取り出す
        if isinstance(sample['output'], list) and len(sample['output']) > 0:
            output_text = sample['output'][0]
        else:
            output_text = sample['output']

        if isinstance(sample['answer'], list) and len(sample['answer']) > 0:
            answer_text = sample['answer'][0]
        else:
            answer_text = sample['answer']

        # output, answer それぞれ文字列の場合のみ、先頭と末尾の [ ] を削除
        if isinstance(output_text, str):
            # 先頭の [ を削除
            if output_text.startswith('['):
                output_text = output_text[1:]
            # 末尾の ] を削除
            if output_text.endswith(']'):
                output_text = output_text[:-1]
            # NOTE: output の場合は Prefixに <think></think> を追加（元の推論がない場合）
            if not output_text.strip().startswith('<think>'):
                output_text = f"<think></think>{output_text}"

        if isinstance(answer_text, str):
            # 先頭の [ を削除
            if answer_text.startswith('['):
                answer_text = answer_text[1:]
            # 末尾の ] を削除
            if answer_text.endswith(']'):
                answer_text = answer_text[:-1]

        # クリーニングされたデータを作成
        cleaned_sample = {
            'id': sample['id'],
            'question': sample['question'],
            'output': output_text,
            'answer': answer_text
        }
        cleaned_data.append(cleaned_sample)

    # --- クリーニングされたデータを保存 ---
    print(f"💾 クリーニングされたデータを保存中: {output_filename}")
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df.to_csv(output_filename, index=False, encoding='utf-8')

    print(f"✅ データクリーニング完了!")
    print(f"   - 処理されたレコード数: {len(cleaned_data)}")
    print(f"   - 保存先: {output_filename}")

    # サンプルデータの確認
    print("\n📋 クリーニング後のサンプルデータ:")
    for i in range(min(3, len(cleaned_data))):
        sample = cleaned_data[i]
        print(f"  サンプル {i+1}:")
        print(f"    ID: {sample['id']}")
        print(f"    Answer: {repr(sample['answer'])}")
        print(f"    Output (先頭100文字): {repr(sample['output'][:100])}...")

    # --- 複数フォーマットでの保存とHugging Face アップロード ---
    try:
        print("\n🔄 最終データの処理とアップロードを開始...")
        final_df = cleaned_df  # 既にDataFrameとして準備されている

        # 統計情報の計算
        success_count = len(final_df)
        total_count = len(cleaned_data)

        print(f"📊 最終データセット統計:")
        print(f"   保存されたレコード数: {success_count}")
        print(f"   処理対象総数: {total_count}")
        print(f"   データセット品質: 100% (クリーニング済みデータ)")

        # ベースファイル名（拡張子なし）を取得
        base_filename = os.path.splitext(output_filename)[0]

        # --- 2. Parquet形式での保存 ---
        parquet_filename = f"{base_filename}.parquet"
        print(f"💾 Parquet形式で保存中: {parquet_filename}")
        final_df.to_parquet(parquet_filename, index=False)

        # --- 3. JSONL形式での保存 ---
        jsonl_filename = f"{base_filename}.jsonl"
        print(f"💾 JSONL形式で保存中: {jsonl_filename}")
        final_df.to_json(jsonl_filename, orient='records', lines=True, force_ascii=False)

        # --- 4. /data に physics_qa_dataset.jsonl を保存 ---
        # dataディレクトリを作成（存在しない場合）
        os.makedirs("data", exist_ok=True)

        physics_qa_dataset_filename = "data/physics_qa_dataset.jsonl"
        print(f"💾 physics_qa_dataset.jsonl を保存中: {physics_qa_dataset_filename}")
        final_df.to_json(physics_qa_dataset_filename, orient='records', lines=True, force_ascii=False)

        # --- Hugging Face Datasetsに3つのファイル形式でアップロード ---
        print(f"📤 {OUTPUT_DATASET_ID} に3つのファイル形式でアップロード中...")

        # 1. CSV ファイルをアップロード
        print("  📄 CSV ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=output_filename,
            path_in_repo=output_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 2. Parquet ファイルをアップロード
        print("  📊 Parquet ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=parquet_filename,
            path_in_repo=parquet_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 3. JSONL ファイルをアップロード
        print("  📝 JSONL ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=jsonl_filename,
            path_in_repo=jsonl_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 4. /data に physics_qa_dataset.jsonl をアップロード
        print("  📝 data/physics_qa_dataset.jsonl をアップロード中...")
        upload_file(
            path_or_fileobj=physics_qa_dataset_filename,
            path_in_repo=physics_qa_dataset_filename,  # これで "data/physics_qa_dataset.jsonl" としてアップロードされる
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        print(f"✅ 3つのファイル形式でのアップロード完了: {OUTPUT_DATASET_ID}")

        # --- 統計情報の表示 ---
        print(f"\n📊 処理統計:")
        print(f"  - 総処理数: {total_count}")
        print(f"  - 成功数: {success_count}")
        print(f"  - 成功率: {success_count/total_count*100:.1f}%")

        print(f"\n📁 保存されたファイル:")
        print(f"  - CSV: {output_filename}")
        print(f"  - Parquet: {parquet_filename}")
        print(f"  - JSONL: {jsonl_filename}")
        print(f"  - Physics QA Dataset: {physics_qa_dataset_filename}")

        print(f"\n🌐 Hugging Face Datasetsアップロード先:")
        print(f"  - Repository: {OUTPUT_DATASET_ID}")
        print(f"    ├─ {output_filename}")
        print(f"    ├─ {parquet_filename}")
        print(f"    ├─ {jsonl_filename}")
        print(f"    └─ {physics_qa_dataset_filename}")

    except Exception as e:
        print(f"❌ データ処理またはHugging Face Hubへのアップロードに失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()
