# DPO データ生成 - 本田大明チーム（P02）

さくらDCサーバでのLlama-3.1-405BモデルによるDPOデータセット生成

## さくらDCサーバ仕様

### チーム割り当て
- **チーム**: 本田大明（P02）
- **パーティション**: `--partition=P02`
- **利用ノード**: `osk-gpu54` (他にosk-gpu55, osk-gpu56も利用可能)
- **ディスク領域**: `/home/Competition2025/P02/shareP02/` (チーム共有30TB)

### ハードウェア仕様
- **CPU**: INTEL XEON PLATINUM 8580 x 2 (120コア、240スレッド)
- **メモリ**: 約1.5TB
- **GPU**: H100 x 8基 (640GB GPU メモリ)

## 実行方法

### 1. バッチ型実行（推奨）
```bash
# ジョブスクリプトを編集（必要に応じて）
vim run_dpo_slurm.sh

# ジョブを投入
sbatch run_dpo_slurm.sh
```

### 2. 対話型実行
```bash
# 対話型セッションを開始
bash run_dpo_interactive.sh
```

### 3. ローカル実行（テスト用）
```bash
# 標準実行スクリプト
bash run_dpo_generation.sh
```

## モニタリング

### ジョブ状況確認
```bash
# 自分のジョブ一覧
squeue -u $(whoami)

# 特定ジョブの詳細
squeue -j <JOB_ID>

# パーティション全体の状況
squeue -p P02
```

### GPU使用状況確認
```bash
# 対話型セッション内で
watch -n 1 nvidia-smi

# リアルタイム監視
nvidia-smi dmon -s pucvmet -d 5
```

### ログファイル確認
```bash
# 出力ログ
tail -f /home/Competition2025/P02/shareP02/logs/dpo_gen_405b-<JOB_ID>.out

# エラーログ
tail -f /home/Competition2025/P02/shareP02/logs/dpo_gen_405b-<JOB_ID>.err

# GPU監視ログ
tail -f /home/Competition2025/P02/shareP02/logs/gpu_monitor_<JOB_ID>.log
```

## ジョブ管理

### ジョブキャンセル
```bash
# 自分のジョブをキャンセル
scancel <JOB_ID>

# チーム用スクリプトでキャンセル（他メンバーのジョブも可）
bash /home/Competition2025/P02/shareP02/scripts/scancel.sh <JOB_ID>

# 特定ノードの全ジョブをキャンセル
squeue -w osk-gpu54 -h -o "%i" | xargs -r -n1 /home/Competition2025/P02/shareP02/scripts/scancel.sh
```

### 自動ジョブの処理
```bash
# kan.hataの自動ジョブを確認
squeue -u kan.hata -p P02

# 必要に応じてキャンセル
bash /home/Competition2025/P02/shareP02/scripts/scancel.sh <AUTO_JOB_ID>
```

## ファイル構成

```
data/dna/
├── gen_dpo_node.py           # メインのPythonスクリプト
├── questions.json           # 入力データ
├── run_dpo_slurm.sh         # SLURMバッチジョブスクリプト
├── run_dpo_interactive.sh   # 対話型実行スクリプト
├── run_dpo_generation.sh    # 標準実行スクリプト
└── README_P02.md           # このファイル
```

## 設定カスタマイズ

### SLURMスクリプトの設定
```bash
# run_dpo_slurm.sh の主要設定
#SBATCH --partition=P02        # パーティション
#SBATCH --nodelist=osk-gpu54   # 使用ノード
#SBATCH --gres=gpu:8           # GPU数
#SBATCH --time=168:00:00       # 実行時間（最大7日）
```

### Python環境設定
```bash
# uv仮想環境を使用する場合
source .venv/bin/activate

# 環境変数
export HF_HOME="/home/Competition2025/P02/shareP02/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/Competition2025/P02/shareP02/.cache/transformers"
```

### Hugging Face認証
```bash
# ログイン
huggingface-cli login

# または環境変数で設定
export HUGGINGFACE_HUB_TOKEN="your_token_here"
```

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - バッチサイズを2に下げる
   - 逐次読み込みモードに変更（`load_both_models=False`）

2. **GPU未認識**
   - `module load cuda/12.4` を確認
   - `CUDA_VISIBLE_DEVICES` の設定を確認

3. **ファイルアクセス権限エラー**
   - `/home/Competition2025/P02/shareP02/logs/` ディレクトリの作成
   - 適切な権限設定

4. **Hugging Face認証エラー**
   - `huggingface-cli whoami` で状況確認
   - Llamaモデルのライセンス承認

### デバッグ用コマンド
```bash
# ノード情報確認
sinfo -p P02

# GPU使用可能状況
sinfo -p P02 -o "%N %G %T"

# ディスク使用量
df -h /home/Competition2025/P02/

# プロセス確認
ps aux | grep python
```

## 期待パフォーマンス

### H100 x 8環境での予想値
- **バッチサイズ**: 4質問同時処理
- **処理速度**: ~10-20秒/バッチ
- **スループット**: ~0.2-0.4 質問/秒
- **メモリ使用量**: 405GB (GPU), ~100GB (システム)

### 生成データ
- **出力ファイル**: `dpo_dataset_405b.jsonl`
- **フォーマット**: 1行1JSON（preferred/non-preferred応答ペア）
- **再開機能**: 中断時も処理済みIDから継続

## サポート

- **チーム共有領域**: `/home/Competition2025/P02/shareP02/`
- **ログディレクトリ**: `/home/Competition2025/P02/shareP02/logs/`
- **参考資料**: さくらDCサーバ利用手順書.md