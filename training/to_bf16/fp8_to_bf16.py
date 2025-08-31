import os
import subprocess
from huggingface_hub import HfApi, snapshot_download

# --- 設定項目 ---

# 1. あなたのHugging Faceユーザー名
# 例: "my-cool-username"
HF_USERNAME = "neko-llm" 

# 2. ダウンロードする元のFP8モデル
FP8_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"

# 3. アップロードする先のBF16モデルのリポジトリ名
BF16_REPO_ID = f"{HF_USERNAME}/{FP8_MODEL_ID.split('/')[-1]}-bf16"

# 4. ローカルに保存する際のディレクトリ名
BASE_DIR = "./models"
FP8_LOCAL_PATH = os.path.join(BASE_DIR, "fp8_model")
BF16_LOCAL_PATH = os.path.join(BASE_DIR, "bf16_model")

# --- メイン処理 ---

def main():
    """
    モデルのダウンロード、BF16への変換、Hugging Face Hubへのアップロードを実行します。
    """
    # Hugging Face APIクライアントを初期化
    api = HfApi()

    # --- 1. モデルのダウンロード ---
    print(f"Downloading model '{FP8_MODEL_ID}' to '{FP8_LOCAL_PATH}'...")
    snapshot_download(
        repo_id=FP8_MODEL_ID,
        local_dir=FP8_LOCAL_PATH,
        local_dir_use_symlinks=False, # Windowsとの互換性のためFalseを推奨
    )
    print("✅ Download complete.")

    # --- 2. FP8からBF16への変換 ---
    # `dequantize.py` を外部プロセスとして実行
    print(f"\nConverting model from FP8 to BF16...")
    print(f"Input: {FP8_LOCAL_PATH}")
    print(f"Output: {BF16_LOCAL_PATH}")
    
    # dequantize.pyが存在することを確認
    if not os.path.exists("llm2025compet/training/to_bf16/dequantize.py"):
        print("❌ Error: 'dequantize.py' not found in the current directory.")
        print("Please save the conversion script provided as 'dequantize.py'.")
        return

    try:
        command = [
            "python", "llm2025compet/training/to_bf16/dequantize.py",
            "--input-fp8-hf-path", FP8_LOCAL_PATH,
            "--output-bf16-hf-path", BF16_LOCAL_PATH,
        ]
        subprocess.run(command, check=True)
        print("✅ Conversion complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}")
        print("Please ensure that 'kernel.weight_dequant' is available and all required libraries (torch, etc.) are installed.")
        return
    except FileNotFoundError:
        print("❌ Error: 'python' command not found. Make sure Python is installed and in your PATH.")
        return

    # --- 3. Hugging Face Hubへのアップロード ---
    print(f"\nUploading converted model to '{BF16_REPO_ID}'...")

    # アップロード先のリポジトリを作成（存在しない場合）
    api.create_repo(
        repo_id=BF16_REPO_ID,
        repo_type="model",
        exist_ok=True,
    )

    # 変換されたモデルファイル（BF16）をアップロード
    api.upload_folder(
        folder_path=BF16_LOCAL_PATH,
        repo_id=BF16_REPO_ID,
        commit_message=f"Upload BF16 version of {FP8_MODEL_ID}",
    )
    
    print("\n🎉 All processes finished successfully!")
    print(f"You can find your model at: https://huggingface.co/{BF16_REPO_ID}")


if __name__ == "__main__":
    # Hugging Face Hubへのログインを促す
    # 事前にターミナルで `huggingface-cli login` を実行しておくのが最も確実です。
    main()
    # if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
    #     print("⚠️ Hugging Face token not found.")
    #     print("Please log in using 'huggingface-cli login' in your terminal.")
    # else:
    #     main()

# 
# srun --partition P02 --nodes=1 --nodelist osk-gpu[56] --gpus-per-node=1 --time 2:00:00 --mem=0 --cpus-per-task=160 --pty bash -i
# python llm2025compet/training/to_bf16/fp8_to_bf16.py