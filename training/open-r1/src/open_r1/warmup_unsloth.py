# warmup_unsloth.py
# Unslothのキャッシュを事前に生成するためのスクリプト
print("Unslothの初期化を開始します。これによりキャッシュファイルが生成されます...")
try:
    from unsloth import FastLanguageModel
    print("Unslothの初期化が正常に完了しました。")
except Exception as e:
    print(f"初期化中にエラーが発生しました: {e}")