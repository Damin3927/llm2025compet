print("Unslothの初期化を開始します。これによりキャッシュファイルが生成されます...")
try:
    from unsloth import FastLanguageModel
    print("Unslothの初期化が正常に完了しました。")
except Exception as e:
    print("初期化中にエラーが発生しました:")
    print(e)