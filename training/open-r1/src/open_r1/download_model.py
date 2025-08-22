from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="モデルのダウンロードスクリプト")
    parser.add_argument("-n", "--name", default=None, type=str)

    return parser.parse_args()

def main():
    args = get_args()

    # 環境変数(モデルの保存先)を表示
    model_path = os.environ["HF_HOME"]
    print(f"download to {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.name)
    model = AutoModelForCausalLM.from_pretrained(args.name)

    print(f"download finished!")

if __name__ == "__main__":
    main()