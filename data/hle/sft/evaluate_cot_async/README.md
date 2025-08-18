# CoT評価・再生成システム

トレーニングデータにおけるChain of Thought（CoT）の品質を評価し、改善するための包括的なシステムです。このモジュールは、CoTサンプルの教育的価値を評価し、評価結果に基づいて高品質なCoTを再生成するための柔軟で再利用可能な方法を提供します。

## 機能

### 評価機能
- **クラスベース設計**: 他のスクリプトへのインポートと使用が簡単
- **柔軟な設定**: APIキー、モデル、パラメータをカスタマイズ可能
- **複数の評価モード**: 単一アイテム評価とバッチデータセット処理
- **マルチモデルサポート**: より堅牢な評価のために複数のモデルを使用
- **進捗追跡**: 内蔵のプログレスバーと詳細な統計情報
- **複数フォーマット対応**: JSONL、JSON、Hugging Faceデータセット
- **再開機能**: 既に評価済みのアイテムをスキップ

### 再生成機能
- **非同期マルチモデル再生成**: 複数のモデルで同時にCoTを再生成
- **自動ベスト選択**: 評価スコアに基づいて最良の結果を自動選択
- **柔軟なモデル設定**: 各モデルに独自のパラメータを設定可能
- **バッチ処理**: データセット全体の再生成をサポート
- **後方互換性**: 既存のコードとの互換性を維持

## インストール

必要な依存関係をインストールしてください：

```bash
pip install openai tqdm python-dotenv
# オプション: Hugging Faceデータセットサポート用
pip install datasets
```

APIキーを設定：

```bash
# 環境変数を設定
export OPENROUTER_API_KEY="your_api_key_here"

# または.envファイルを作成
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

## クイックスタート

### 基本的な使用方法

#### 評価

```python
from cot_evaluator import CoTEvaluationProcessor

# プロセッサーを初期化
processor = CoTEvaluationProcessor()

# 単一アイテムを評価
item = {
    "question": "What is 2 + 2?",
    "output": "<think>I need to add 2 and 2. 2 + 2 = 4.</think>",
    "answer": "4"
}

result = processor.evaluate_single_item(item)
print(f"グレード: {result['grade']}, スコア: {result['score']:.2f}/10")

# データセットを評価
stats = processor.evaluate_dataset(
    dataset_path="your_dataset.jsonl",
    output_file="evaluated_dataset.jsonl"
)
print(f"{stats['evaluated_count']}個のアイテムを評価しました")
```

#### 再生成

```python
from cot_regenerator import CoTRegenerator
import asyncio

# 単一モデルで再生成
regenerator = CoTRegenerator()
new_cot = regenerator.regenerate_single(
    question="What is 15% of 240?",
    answer="36",
    previous_cot="15% of 240 = 240 × 0.15 = 36",
    evaluation_details={
        "grade": "C",
        "weaknesses": ["説明不足", "検証なし"],
        "improvement_suggestions": ["パーセントの概念を説明", "結果を検証"]
    }
)

# 複数モデルで非同期再生成（最良の結果を自動選択）
regenerator = CoTRegenerator(
    models=["model1", "model2", "model3"]
)

best_result = asyncio.run(regenerator.regenerate_multi_async(
    question, answer, previous_cot, evaluation_details
))

print(f"最良のモデル: {best_result['best_model']}")
print(f"最良のCoT: {best_result['best_cot']}")
```

### 高度な設定

```python
# プロセッサーをカスタマイズ
processor = CoTEvaluationProcessor(
    api_key="your_custom_api_key",
    model="deepseek/deepseek-r1-0528:free",
    temperature=0.1,
    max_tokens=4000
)

# 評価に複数のモデルを使用
stats = processor.evaluate_dataset(
    dataset_path="dataset.jsonl",
    evaluator_models="model1,model2,model3",
    eval_concurrency=3,
    output_format="json",
    skip_existing=False
)
```

## APIリファレンス

### CoTEvaluationProcessor

CoT評価のメインクラス。

### CoTRegenerator（新規追加）

評価結果に基づいてCoTを再生成するクラス。

#### コンストラクタ

```python
CoTRegenerator(
    models: Optional[Union[str, List[str], List[ModelConfig]]] = None,
    api_keys: Optional[Dict[str, str]] = None,
    default_api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: Optional[str] = None
)
```

**パラメータ:**
- `models`: 単一モデル名、モデル名のリスト、またはModelConfigオブジェクトのリスト
- `api_keys`: APIキーの辞書（例: {"openrouter": "key", "anthropic": "key"}）
- `default_api_base`: デフォルトのAPIベースURL
- `system_prompt`: 再生成用のカスタムシステムプロンプト

#### メソッド

##### regenerate_single()

単一モデルでCoTを再生成。

```python
regenerate_single(
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any],
    model_config: Optional[ModelConfig] = None
) -> Optional[str]
```

**パラメータ:**
- `question`: 問題/質問
- `answer`: 期待される答え
- `previous_cot`: 前回のCoT推論
- `evaluation_details`: 評価器からの評価詳細
- `model_config`: 使用する特定のモデル設定（オプション）

**戻り値:**
- 再生成されたCoTテキスト（`<think>`タグなし）または失敗時はNone

##### regenerate_multi_async()

複数モデルで非同期にCoTを再生成し、最良の結果を選択。

```python
async regenerate_multi_async(
    question: str,
    answer: str,
    previous_cot: str,
    evaluation_details: Dict[str, Any],
    return_all: bool = False
) -> Union[Dict[str, Any], List[Dict[str, Any]]]
```

**パラメータ:**
- `question`: 問題/質問
- `answer`: 期待される答え
- `previous_cot`: 前回のCoT推論
- `evaluation_details`: 評価器からの評価詳細
- `return_all`: Trueの場合すべての結果を返す、Falseの場合最良のみ返す

**戻り値:**
- `return_all`がFalseの場合: 最良の結果とメタデータを含む辞書
- `return_all`がTrueの場合: すべての結果とメタデータのリスト

##### regenerate_dataset()

データセット全体のCoTを再生成。

```python
regenerate_dataset(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    grade_threshold: str = "C",
    specific_ids: Optional[List[str]] = None,
    use_async: bool = True
) -> Dict[str, Any]
```

**パラメータ:**
- `dataset_path`: 入力JSONLデータセットへのパス
- `output_path`: 出力パス（デフォルトは入力パス）
- `grade_threshold`: このグレード以下のアイテムを再生成
- `specific_ids`: 再生成する特定のIDのリスト（オプション）
- `use_async`: 非同期マルチモデル再生成を使用するか

**戻り値:**
- 再生成統計を含む辞書

### ModelConfig（新規追加）

特定モデルの設定クラス。

```python
ModelConfig(
    model_name: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 40000,
    retry_attempts: int = 3
)
```

**パラメータ:**
- `model_name`: モデル名
- `api_base_url`: APIベースURL
- `api_key`: APIキー（提供されない場合は環境変数を使用）
- `temperature`: 生成温度
- `max_tokens`: 最大トークン数
- `retry_attempts`: リトライ回数

### CoTEvaluationProcessor詳細

#### コンストラクタ

```python
CoTEvaluationProcessor(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8000
)
```

**パラメータ:**
- `api_key`: OpenAI APIキー（提供されない場合は環境変数を使用）
- `base_url`: APIベースURL（デフォルトはOpenRouter）
- `model`: 評価に使用するモデル名
- `temperature`: 生成温度（一貫した評価のために0.0）
- `max_tokens`: 評価レスポンスの最大トークン数

#### メソッド

##### evaluate_single_item()

単一のCoTアイテムを評価。

```python
evaluate_single_item(
    item: Dict,
    model_names: Optional[List[str]] = None,
    concurrency: int = 4
) -> Dict
```

**パラメータ:**
- `item`: `question`、`output`、`answer`フィールドを含む辞書
- `model_names`: マルチモデル評価用のモデル名リスト
- `concurrency`: 同時評価数

**戻り値:**
- グレード、スコア、詳細な評価を含む辞書

##### evaluate_dataset()

データセット全体を評価。

```python
evaluate_dataset(
    dataset_path: str,
    output_file: Optional[str] = None,
    ids: Optional[str] = None,
    evaluator_models: Optional[str] = None,
    eval_concurrency: int = 4,
    output_format: str = 'jsonl',
    skip_existing: bool = True
) -> Dict[str, Any]
```

**パラメータ:**
- `dataset_path`: データセットファイルまたはHugging Faceデータセットへのパス
- `output_file`: 出力ファイルパス（指定されない場合は入力を上書き）
- `ids`: 評価するIDのカンマ区切りリスト
- `evaluator_models`: モデル名のカンマ区切りリスト
- `eval_concurrency`: 同時評価数
- `output_format`: 出力フォーマット（'jsonl'または'json'）
- `skip_existing`: 既に評価済みのアイテムをスキップするか

**戻り値:**
- 評価統計を含む辞書

## データフォーマット

### 入力フォーマット

各アイテムは以下の構造を持つ必要があります：

```json
{
    "id": "unique_id",
    "question": "問題または質問",
    "output": "<think>思考の連鎖による推論</think>",
    "answer": "期待される答え"
}
```

### 出力フォーマット

評価結果は`metadata.cot_history`フィールドに保存されます：

```json
{
    "metadata": {
        "cot_history": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "output": "<think>...</think>",
                "evaluation": {
                    "grade": "A",
                    "score": 8.5,
                    "timestamp": "2024-01-01T12:00:00",
                    "passed_requirements": {
                        "independence": true,
                        "logical_completeness": true,
                        "correctness": true,
                        "answer_reached": true
                    },
                    "learning_value_scores": {
                        "method_explanation": 8,
                        "step_by_step": 9,
                        "verification": 7,
                        "common_mistakes": 8,
                        "domain_insight": 9,
                        "metacognitive": 8
                    },
                    "strengths": ["明確なステップバイステップの推論", "良い説明"],
                    "weaknesses": ["より多くの検証を提供できる"],
                    "improvement_suggestions": ["代替解法を追加"]
                }
            }
        ]
    }
}
```

## 評価基準

評価は以下に基づいてCoTの品質を評価します：

### 必須要件（いずれかが失敗した場合はグレードD）
1. **独立性**: 外部参照なし
2. **論理的完全性**: 飛躍のない接続された推論
3. **正確性**: 方法の適切な適用
4. **答えへの到達**: 正しい/妥当な答えに到達

### 学習価値（0-10スケール）
1. **方法選択の説明**: なぜこの方法を選んだか
2. **ステップバイステップの導出**: 明確で詳細なステップ
3. **検証とチェック**: 結果を検証
4. **一般的な間違いの処理**: エラーが起きやすい領域を指摘
5. **ドメインの洞察**: 意味と解釈を説明
6. **メタ認知的要素**: 推論プロセスを示す

### グレーディングスケール
- **A (8.0+)**: 優秀なトレーニングデータ
- **B (6.0-7.9)**: 良好なトレーニングデータ
- **C (4.0-5.9)**: 許容可能なトレーニングデータ
- **D (<4.0)**: 不良なトレーニングデータ

## 例

以下の完全な例についてはサンプルファイルを参照してください：

### 評価の例（`example_usage.py`）
- 単一アイテム評価
- データセット評価
- マルチモデル評価
- カスタム設定

### 再生成の例（`regenerate_example.py`）
- 単一モデル再生成
- カスタムモデル設定
- 非同期マルチモデル再生成
- 全結果の取得と比較
- データセット全体の再生成
- 既存コードとの互換性

## コマンドライン使用方法

元のコマンドラインインターフェースも引き続き利用可能です：

```bash
python cot_evaluator.py --dataset your_dataset.jsonl --output-file evaluated.jsonl
```

## 他のスクリプトとの統合

### 評価と再生成の統合例

```python
# データ処理パイプラインで
from cot_evaluator import CoTEvaluationProcessor
from cot_regenerator import CoTRegenerator
import asyncio

async def process_training_data():
    evaluator = CoTEvaluationProcessor()
    regenerator = CoTRegenerator(
        models=["model1", "model2", "model3"]
    )
    
    # データをロード
    data = load_your_data()
    
    for item in data:
        # CoTの品質を評価
        if needs_evaluation(item):
            result = evaluator.evaluate_single_item(item)
            
            if result['grade'] in ['A', 'B']:
                # 高品質なのでそのままトレーニングに使用
                add_to_training_set(item)
            elif result['grade'] == 'C':
                # 改善の余地あり - 再生成を試みる
                evaluation_details = {
                    "grade": result['grade'],
                    "strengths": result['evaluation']['strengths'],
                    "weaknesses": result['evaluation']['weaknesses'],
                    "improvement_suggestions": result['evaluation']['improvement_suggestions'],
                    "learning_value_scores": result['evaluation']['learning_value_scores']
                }
                
                # マルチモデルで再生成し最良を選択
                best = await regenerator.regenerate_multi_async(
                    item['question'],
                    item['answer'],
                    item['output'],
                    evaluation_details
                )
                
                if best and best['best_grade'] > result['grade']:
                    # 改善成功
                    item['output'] = f"<think>{best['best_cot']}</think>{item['answer']}"
                    add_to_training_set(item)
                else:
                    # 改善失敗 - 元のまま使用
                    add_to_training_set(item)
            else:
                # グレードD - 拒否
                reject_item(item)

# 実行
asyncio.run(process_training_data())
```

### 既存コードからのインポート

```python
# 他のフォルダから使用する場合
import sys
sys.path.append('/path/to/evaluate_cot_multiagent')

from evaluate_cot_multiagent import CoTRegenerator, CoTEvaluator

# または直接インポート
from evaluate_cot_multiagent.cot_regenerator import CoTRegenerator
from evaluate_cot_multiagent.cot_evaluator import CoTEvaluator
```

## エラーハンドリング

プロセッサーには堅牢なエラーハンドリングが含まれています：
- リトライロジックを含むAPIの失敗
- 無効なJSONレスポンス
- 必須フィールドの欠落
- ファイルI/Oエラー

評価の成功率とエラー数については、返された統計情報を確認してください。

## パフォーマンスのヒント

- 中断された評価を再開するには`skip_existing=True`を使用
- APIレート制限に基づいて`eval_concurrency`を調整
- より堅牢な評価のために複数のモデルを使用（ただしコストが増加）
- 大規模なデータセットのバッチ処理を検討