# ReasonMD データセット変換

## 概要

このモジュールは、ReasonMD医療データセットを処理するための2段階パイプラインを提供します：
1. **選択**: ストリーミングデータセットから高品質なサンプルを選択するために、半ガウス分布を使用した長さベースのサンプリング
2. **変換**: 選択されたデータをVLLMで抽出された回答を含む構造化されたSFT（教師あり微調整）形式に変換

このパイプラインは、医療推論の質問を処理し、思考の連鎖出力を抽出し、VLLMを使用して冗長な推論テキストから簡潔な最終回答を生成します。

## データセット処理パイプライン
[mermaid_flowchart](https://claude.ai/public/artifacts/e6ca12dc-b734-4421-a326-b00004a70aa7)
### パイプラインのステップごとの説明

1. **HuggingFaceデータセットの取得**
   - `lingshu-medical-mllm/ReasonMed`データセットをHuggingFace Hubからストリーミングで取得します。

2. **ステップ1: データ選択（reasonmd_selector.py）**
   - データセットをストリーミングしながら処理を開始します。

3. **長さ統計の収集**
   - 最初の5000サンプルについて、出力テキストの長さ分布を収集します。

4. **動的ビンの作成**
   - 収集した長さ分布に基づき、サンプルの長さごとに動的なビン（区間）を作成します。

5. **半ガウス重みの適用**
   - 長い出力ほど選ばれやすくなるよう、半ガウス分布に基づく重みを各ビンに割り当てます。

6. **リザーバーサンプリング**
   - ストリーミングデータから効率的にサンプルを選択するため、リザーバーサンプリングを実施します。

7. **各アイテムの処理**
   - 各サンプルを長さバケットに割り当て、確率的にサンプリングします。

8. **選択されたアイテムの保存**
   - 選ばれたサンプルを`reasonmd_selected.json`として保存します。

9. **ステップ2: データ変換（convert_reasonmd.py）**
   - 選択済みデータ、または`CoT_Medicine`データセットを読み込みます。

10. **各行の処理**
    - 各データ行について以下を実施します。

11. **フィールドの抽出**
    - `instruction`を「質問」として、`output`を「推論」として抽出します。

12. **検証**
    - 抽出した内容が有効かどうかを検証します。無効な場合はスキップし、警告をログに記録します。

13. **VLLMによる処理**
    - 有効なデータについてはVLLM（Qwen/Qwen3-32Bモデル）で推論から回答を抽出します。

14. **回答の解析と検証**
    - 抽出した回答の形式が正しいか検証します。無効な場合はスキップします。

15. **構造化エントリの作成**
    - 有効な場合は、推論と回答を含む構造化エントリを作成します。

16. **出力のフォーマット**
    - `<think>推論</think>`タグで推論を囲み、その後に回答を記載する形式で出力を整形します。

17. **結果への追加**
    - 作成したエントリを結果リストに追加します。

18. **全データ処理の繰り返し**
    - さらに行があれば繰り返し、なければ次へ進みます。

19. **JSONファイルへの保存**
    - 最終的な結果を`reasonmd_cot.json`として保存します。

20. **警告の記録**
    - 無効な行があった場合は警告としてログに記録します。

このように、データセットの選択から変換・保存までを段階的に処理します。

## 前提条件

### システム要件
- Python 3.7以上
- CUDA対応GPU（VLLMモデル推論用）
- 十分なGPUメモリ（32Bパラメータモデル用）
- インターネット接続（データセットとモデルのダウンロード用）

### 依存関係
```bash
pip install datasets vllm tqdm transformers torch
```

### 設定
- Hugging Faceトークンを`../../keys.json`に保存：
```json
{
  "llm": "your-huggingface-token-here"
}
```

## データセット情報

### プライマリデータセット（選択用）
- **ソース**: Hugging Face Hub - `lingshu-medical-mllm/ReasonMed`
- **モード**: ストリーミング（完全ダウンロード不要）
- **内容**: 大規模医療推論データセット

### 代替データセット（直接変換用）
- **ソース**: Hugging Face Hub - `neko-llm/CoT_Medicine`
- **設定**: `reasonmed_raw_selected`
- **分割**: `train`

### データフィールド
- **instruction**: 医療の質問またはプロンプト
- **output**: 回答付きの詳細な推論プロセス
- **input**: 追加コンテキスト（存在する場合）

## 処理コンポーネント

### ステップ1: データ選択（`reasonmd_selector.py`）

#### 1.1 長さベースのサンプリング
- **統計収集**: 長さ分布のために最初の5000サンプルを分析
- **動的ビニング**: バランスの取れたサンプリングのために分位数ベースのビンを作成
- **半ガウス重み付け**: より長い出力を優先する重みを適用
- **リザーバーサンプリング**: ストリーミングデータから効率的にサンプルを選択

#### 1.2 選択パラメータ
- `target_samples`: 選択するサンプル数（デフォルト: 500）
- `sample_size_for_stats`: 統計用サンプル（デフォルト: 5000）
- `num_bins`: 長さビンの数（デフォルト: 6）
- `curve_sharpness`: 半ガウス曲線の鋭さ（デフォルト: 3.0）

### ステップ2: データ変換（`convert_reasonmd.py`）

#### 2.1 データ読み込み
- `reasonmd_selected.json`から選択されたデータを読み込み
- 代替: CoT_Medicineデータセットから直接読み込み

#### 2.2 回答抽出（`extract_answer_from_output`）
冗長な推論から簡潔な回答を知的に抽出するためにVLLMを使用：

- **入力**: 質問 + 完全な推論出力
- **モデル**: 設定可能（デフォルト: Qwen/Qwen3-32B）
- **出力**: 単一文字（A-D）または特定の医学用語
- **認識されるパターン**:
  - 直接回答: "A", "B", "C", "D"
  - フォーマット済み回答: "Answer: D", "The answer is B"
  - 埋め込み回答: より長いテキストから抽出

#### 2.3 行変換（`transform_row`）
各データセット行を処理：

1. **検証**: 必須フィールドをチェック
2. **抽出**: 質問と推論出力を取得
3. **回答生成**: VLLMを使用して最終回答を抽出
4. **形式検証**: 回答が品質基準を満たすことを確認
5. **構造作成**: thinkタグでフォーマットされたエントリを構築

#### 2.4 回答検証（`basic_answer_format_check`）
抽出された回答の品質管理：

- **有効な形式**:
  - 単一文字（A-D）
  - 医学用語（2-300文字）
  
- **無効なインジケータ**:
  - 空または短すぎる回答
  - "unable to extract", "cannot determine"などのフレーズ
  - 過度に長い応答

## 出力形式

### JSON構造
```json
[
  {
    "id": "reasonmed_0",
    "question": "45歳の患者が...",
    "output": "<think>このケースをステップバイステップで分析させてください...</think>\n\nB",
    "answer": "B"
  },
  {
    "id": "reasonmed_1",
    "question": "どの薬物が禁忌ですか...",
    "output": "<think>禁忌を確認しています...</think>\n\nメトホルミン",
    "answer": "メトホルミン"
  }
]
```

### フィールド
- **id**: 一意の識別子（形式: `reasonmed_{index}`）
- **question**: データセットからの元の医療質問
- **output**: thinkタグを含む組み合わせた推論と回答
- **answer**: 抽出された最終回答（簡潔）

## 使用方法

### 完全な2段階ワークフロー

```bash
# ステップ1: ストリーミングデータセットから高品質なサンプルを選択
python reasonmd_selector.py --target_samples 1000 --num_bins 6

# ステップ2: 選択されたデータをVLLMでSFT形式に変換
python convert_reasonmd.py
```

### ステップ1: データ選択

```bash
# 基本的な選択（デフォルトで500サンプル）
python reasonmd_selector.py

# カスタム選択パラメータ
python reasonmd_selector.py \
    --target_samples 1000 \
    --sample_size_for_stats 5000 \
    --num_bins 8 \
    --curve_sharpness 4.0 \
    --output custom_selected.json

# 非常に長い出力を優先
python reasonmd_selector.py --curve_sharpness 5.0 --num_bins 10
```

### ステップ2: データ変換

```bash
# 選択されたデータを変換（デフォルトでreasonmd_selected.jsonを使用）
python convert_reasonmd.py

# カスタムモデルを使用
python convert_reasonmd.py --model "meta-llama/Llama-3-70B"

# テストモード（最初の5エントリ）
python convert_reasonmd.py --test-mode --debug

# マルチGPU処理
python convert_reasonmd.py --tp 2

# 完全なオプション
python convert_reasonmd.py \
    --model "Qwen/Qwen3-32B" \
    --tp 1 \
    --test-mode \
    --debug
```

### 代替: CoT_Medicineからの直接変換
選択をスキップしてCoT_Medicineデータセットから直接変換したい場合：

```bash
# これは選択されたデータの代わりにneko-llm/CoT_Medicineから読み込みます
python convert_reasonmd.py
```

## 設定パラメータ

### VLLM設定
- **Temperature**: 0.1（一貫した出力のために低く設定）
- **Max Tokens**: 50（短い回答に最適化）
- **Top-p**: 0.9
- **Stop Tokens**: `["\n", ".", "```"]`
- **Max Model Length**: 8192トークン

### 処理オプション
- `--model`: 回答抽出用のVLLMモデル（デフォルト: Qwen/Qwen3-32B）
- `--tp`: マルチGPU用のテンソル並列サイズ（デフォルト: 1）
- `--test-mode`: テスト用に5エントリのみ処理
- `--debug`: 詳細なデバッグ出力を有効化

## 出力ファイル

### ステップ1: 選択出力
- **デフォルト**: `reasonmd_selected.json`
- **カスタム**: `--output`パラメータで指定
- **内容**: 元の構造を持つ選択された生データセットアイテム

### ステップ2: 変換出力

#### プロダクションモード
- **パス**: `~/explore/data/hle/sft/medical/results/reasonmd_cot.json`
- **内容**: SFT形式で正常に処理されたすべてのエントリ

#### テストモード
- **パス**: `~/explore/data/hle/sft/medical/results/reasonmd_cot_test.json`
- **内容**: 検証用の最初の5つの処理済みエントリ

## エラー処理

### スキップ条件
以下の場合、行がスキップされます：
1. `instruction`フィールドが欠落または空
2. `output`フィールドが欠落または空
3. 回答抽出の失敗
4. 無効な回答形式

### 警告
スクリプトは以下の警告をログに記録します：
- 理由付きでスキップされた行
- 検証の失敗
- 処理エラー

## テスト

### ユニットテスト
```bash
# 変換テストを実行
python test_reasonmd_conversion.py

# シェルスクリプトでクイックテスト
./test_convert_reasonmd.sh
```

### 検証ステップ
1. サンプル出力の形式準拠をチェック
2. 回答抽出の精度を検証
3. thinkタグのフォーマットを確認
4. JSON構造を検証

## プロジェクト構造

```
reasonMD/
├── convert_reasonmd.py           # メイン変換スクリプト
├── test_reasonmd_conversion.py   # ユニットテスト
├── test_convert_reasonmd.sh      # テストシェルスクリプト
├── reasonmd_selector.py          # データセット選択ユーティリティ
├── product_requirement_document.md # 詳細仕様
├── README.md                     # 英語版README
└── README_ja.md                  # このファイル（日本語版）
```

## パフォーマンスの考慮事項

### GPUメモリ
- モデルの読み込みには大量のVRAMが必要（モデルによって異なる）
- マルチGPU分散には`--tp`パラメータを使用
- リソースが限られている場合は小さいモデルを検討

### 処理時間
- 完全なデータセット: GPUによって数時間
- テストモード: 約1-2分
- ボトルネック: 回答抽出のためのVLLM推論

## トラブルシューティング

### よくある問題

1. **CUDAメモリ不足**
   ```bash
   # テンソル並列を使用
   python convert_reasonmd.py --tp 2
   
   # または小さいモデルを使用
   python convert_reasonmd.py --model "Qwen/Qwen2-7B"
   ```

2. **HFトークンの欠落**
   - `../../keys.json`に`keys.json`が存在することを確認
   - トークンがデータセットアクセス権限を持っていることを確認

3. **VLLMインストール**
   ```bash
   pip install vllm --upgrade
   ```

4. **データセットアクセスの問題**
   - インターネット接続を確認
   - Hugging Faceトークンの有効性を確認
   - アカウントでデータセットにアクセス可能であることを確認

## 開発ノート

### コンバーターの拡張
処理ロジックを変更するには：

1. **カスタム回答抽出**: `extract_answer_from_output()`を変更
2. **検証ルール**: `basic_answer_format_check()`を更新
3. **出力形式**: `transform_row()`で構造を調整
4. **モデル設定**: `main()`でVLLMパラメータを変更

### 新機能の追加
- 効率化のためのバッチ処理
- 複数のデータセット設定のサポート
- 回答抽出用のカスタムプロンプトテンプレート
- 処理結果の統計分析

## ライセンス

この変換ツールは研究および教育目的で提供されています。データのライセンス情報については、元の[CoT_Medicineデータセット](https://huggingface.co/datasets/neko-llm/CoT_Medicine)を参照してください。