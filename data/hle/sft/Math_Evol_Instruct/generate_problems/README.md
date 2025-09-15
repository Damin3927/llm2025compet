# Math Problem Evolution Scripts

## 概要

このリポジトリには、OpenRouter利用して、既存の数学問題をより挑戦的な問題へと自動で上方修正するためのPythonスクリプトが含まれています。
* **対象データセット**: https://huggingface.co/datasets/Hothan/OlympiadBench
* **利用API**: deepseek/deepseek-r1-0528:free
* **注**: OpenRouterの無料枠がどの程度あるのか、掴めきれてない状態（08/02現在）です。公式ドキュメントによると1日50リクエストなので、40問に設定していますが、それ以上実施できたこともありました。また、1問の生成に平均して5分ほどかかります。

---

## セットアップ

スクリプトを実行する前に、以下の準備を完了させてください。

1.  **必要なPythonライブラリのインストール**
    ```bash
    pip install datasets pandas requests tqdm huggingface_hub pyarrow
    ```

2.  **OpenRouter APIキーの設定**
    OpenRouterで取得したAPIキーを、環境変数として設定してください。
    * **Windowsの場合:**
      ```bash
      set OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
      ```
    * **macOS / Linuxの場合:**
      ```bash
      export OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
      ```

3.  **Hugging Faceへログイン**
    ```bash
    huggingface-cli login
    ```
    トークンを入力してください。
    **注**: 全問処理版 (`evolve_math_problems_all.py`) でHugging Faceへのアップロードを行う場合は、**`write`権限**を持つトークンが必要です。

---

## スクリプトの実行

目的に応じて、2つのスクリプトを使い分けます。

### 方法1：全問題を処理し、Hugging Faceにアップロードする（8/3現在修正中のため無視してください）

この方法は、データセットのすべての問題を対象に処理を実行します。

* **スクリプト名:** `evolve_math_problems_all.py` (全問処理用のスクリプト)
* **主な機能:**
    * **レジューム機能**: 処理が中断された場合、次回の実行時に自動で途中から再開します。
    * **逐次保存**: 5問処理するごとに、`evolved_math_problems.csv`に結果を追記保存します。
    * **自動アップロード**: 全処理完了後、最終的なCSVファイルをHugging Face Hubにアップロードします。

#### 実行コマンド
```bash
python evolve_math_problems_all.py
```

### 方法2：指定範囲の問題のみを処理する（CSV生成のみ）

この方法は、データセットの一部だけを処理し、結果をローカルのCSVファイルに保存したい場合に便利です。Hugging Faceへのアップロードは行いません。

* **スクリプト名:** `evolve_math_problems_divided_for_[OE_TO or TP_TO]_maths_en_COMP.py` (範囲指定用のスクリプト)

実行前の設定
スクリプトファイルを開き、main関数冒頭にある以下の箇所を修正してください。

```Python
# --- ユーザー設定 (User Configuration) ---
# ★★★ ここで開始問題番号と処理数を指定してください ★★★
start_from_problem_number = 401  # 例: 401番目の問題から開始
num_to_process = 40            # 例: 40問を処理
# ★★★ 設定はここまで ★★★
```

#### 実行コマンド
```bash
python evolve_math_problems_divided_for_[OE_TO or TP_TO]_maths_en_COMP.py
```