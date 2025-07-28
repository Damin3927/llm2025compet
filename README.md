# Team Neko 2025 LLM Competition

## ドキュメント

- [Notion](https://www.notion.so/Team-22a9dd6b4cc28015850dc9a4d2314393)

## ローカル環境構築

Nvidia GPU を持っていらっしゃらない方は、docker で開発するメリットはあまりないと思われます。docker の中で開発することもできます。

Nvidia GPU を持っている方は、host にインストールされている Nvidia ドライバのバージョンの関係で、docker の中でないとうまく動かないことがあるかもしれません。(未検証)

### Docker を使わずに開発する

```bash
$ curl -LsSf https://astral.sh/uv/0.8.0/install.sh | sh
$ echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
$ uv sync
```

### Docker を使って開発する

```bash
$ make build
$ # 以下のコマンドで、コンテナの中に入ることができる。コンテナの中で、uv や python3 等のコマンドを実行することができる。
$ make run
```

上記の `make run` か、もしくは `make up` でコンテナを裏側で立ち上げ続けることができる。VSCode 互換のエディタであれば、`Attach to Running Container...` で `/llm2025compet-app-1` を選ぶことで、コンテナ内で作業することができる。

#### Windowsの場合
```bash
$ # Imageの作成
$ build-win.bat
$ # 以下のコマンドで、コンテナの中に入ることができる。コンテナの中で、uv や python3 等のコマンドを実行することができる。
$ up-win.bat
```
CUDAのバージョンが12.8未満だと、イメージと一致せずエラーが出る。その時は、CUDAのアップデートか、イメージのダウングレードで対応できる。
イメージのダウングレードはbuild-win.batにおいてnvidia_cuda_version="12.X.1-devel-ubuntu24.04"
とすれば良い

#### devcontainerの場合
0. **build-win.batでImageを作成**
1. vscodeの拡張機能"Dev Containers"をダウンロード
2. 画面左下端のボタンをクリック
3. Reopen in container(コンテナーで再度開く)をクリック

devcontainerでも同様に、CUDAのバージョンによってエラーが出るため注意する。対処法は上記と同様である。

## GPU ノードでの環境構築

こちらをご参照ください。

[Notion](https://www.notion.so/2379dd6b4cc2803aac5eddbf87c7a436)

## Team 間の IO インターフェース

それぞれ HuggingFace を経由することを想定しています。

- データ(HLE/DNA) <-> 学習: HuggingFace Datasets
  - データチームが push した Dataset を、学習チームが pull して学習に使用する
- 学習 <-> 推論: HuggingFace Models
  - 学習チームが push したモデルを、推論チームが pull して推論に使用する

## データ処理ツール

### Difficulty Scorer

質問回答データの難易度を複数の指標で評価し、スコア化するツールです。

**場所**: `data/hle/sft/difficulty_scorer.py`

**機能**:
- 複数の小型LLMを使用した難易度評価
- 3つの指標による総合スコア:
  1. 金回答の平均対数確率
  2. アンサンブル正解率
  3. IRT難易度パラメータ β
- ストリーミング処理でメモリ効率的
- GPU/CPU両対応、OOM対策済み

**使用例**:
```bash
# 基本的な使用
python difficulty_scorer.py \
  --input "dataset-name" \
  --output "difficulty_scores.json" \
  --max_samples 10000

# HuggingFaceプライベートデータセット
python difficulty_scorer.py \
  --input "private/dataset" \
  --dataset_spec "config:split" \
  --question_field "problem" \
  --answer_field "solution" \
  --hf_token $HF_TOKEN \
  --max_sequence_length 1024 \
  --use_float32 \
  --output "scores.json"

# SLURMでの実行
sbatch run_difficulty_scorer.sh
```

**出力形式**:
```json
[
  {
    "id": "item_1",
    "avg_logprob": -2.1,
    "ensemble_acc": 0.75,
    "irt_beta": 0.3,
    "difficulty_z": 1.2
  }
]
```

### Length Selector

回答長に基づいてデータを選択するツールです。半ガウシアン分布で最長回答を最も多く、最短回答を最も少なく選択します。

**場所**: `data/hle/sft/length_selector.py`

**機能**:
- 動的ビン作成（データ分布に基づく）
- 半ガウシアン分布による重み付け
- オープンエンド方式（最短・最長の外れ値も含む）
- ストリーミング処理対応
- リザーバーサンプリング

**使用例**:
```bash
# 基本的な使用
python length_selector.py \
  --input "dataset-name" \
  --total_samples 5000 \
  --output "selected_data.json"

# 詳細設定
python length_selector.py \
  --input "neko-llm/SFT_OpenMathReasoning" \
  --dataset_spec "cot" \
  --answer_field "generated_solution" \
  --total_samples 10000 \
  --num_bins 8 \
  --curve_sharpness 3.0 \
  --sample_size_for_stats 2000 \
  --shuffle \
  --output "math_selected.json"

# SLURMでの実行
sbatch run_length_selector.sh
# または
./universal_length_selector.sh "dataset-name" 5000
```

**パラメータ**:
- `--curve_sharpness`: 分布の鋭さ（高いほど長い回答に集中）
  - 1.0: 緩やかな分布
  - 2.0: 標準（デフォルト）
  - 4.0: 鋭い分布
- `--num_bins`: ビン数（デフォルト: 6）
- `--sample_size_for_stats`: ビン作成用サンプル数（デフォルト: 1000）

**分布例**:
```
Bin 0 (shortest): 100 samples (10%)   ← 最少
Bin 1: 200 samples (20%)
Bin 2: 300 samples (30%)
Bin 3: 400 samples (40%)
Bin 4: 500 samples (50%)
Bin 5 (longest): 600 samples (60%)    ← 最多
```

### 実行スクリプト

両ツールには複数の実行スクリプトが用意されています：

**基本スクリプト**: `run_*.sh`
- 設定をスクリプト内で編集
- シンプルな実行方法

**ユニバーサルスクリプト**: `universal_*.sh`
- コマンドライン引数対応
- ローカル/SLURM自動検出
- タイムスタンプ付きログ

**動的投入スクリプト**: `submit_*.sh`
- パラメータ指定で動的にジョブ作成
- 一時ファイル自動管理

## License

TBD
