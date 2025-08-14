#!/usr/bin/env bash
set -euo pipefail

# 現在のディレクトリを動的に取得（スクリプトの場所から3階層上）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SESSION="think_tags"
WORKERS=5
TOTAL=49027

# Weights & Biases設定
WANDB_PROJECT="dpo-add-think-tags"
WANDB_ENABLED=true

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux が見つかりません。sudo apt-get install -y tmux などでインストールしてください" >&2
  exit 1
fi

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  echo "仮想環境 .venv が見つかりません: $PROJECT_ROOT/.venv/bin/activate" >&2
  exit 1
fi

# wandbが利用可能かチェック
if [ "$WANDB_ENABLED" = true ]; then
  if ! "$PROJECT_ROOT/.venv/bin/python" -c "import wandb" 2>/dev/null; then
    echo "Warning: wandbがインストールされていません。--wandbオプションは無効になります。"
    echo "pip install wandb でインストールしてください。"
    WANDB_ENABLED=false
  fi
fi

chunk_size=$(( (TOTAL + WORKERS - 1) / WORKERS ))

make_cmd() {
  local start=$1
  local end=$2
  local worker_id=$3
  
  if [ "$WANDB_ENABLED" = true ]; then
    cat <<'EOS'
bash -lc '
set -euo pipefail
PROJECT_ROOT="__PROJECT_ROOT__"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
python data/dna/add_think_tags.py \
  --start_index __START__ \
  --end_index __END__ \
  --output_file "data/dna/think_tagged___START_____END_MINUS_1__.jsonl" \
  --wandb \
  --wandb_project __WANDB_PROJECT__ \
  --wandb_run_name "worker___WORKER_ID_____START_____END_MINUS_1__"
'
EOS
  else
    cat <<'EOS'
bash -lc '
set -euo pipefail
PROJECT_ROOT="__PROJECT_ROOT__"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
python data/dna/add_think_tags.py \
  --start_index __START__ \
  --end_index __END__ \
  --output_file "data/dna/think_tagged___START_____END_MINUS_1__.jsonl"
'
EOS
  fi
}

echo "=== Think Tags 並行処理セットアップ ==="
echo "プロジェクト: $PROJECT_ROOT"
echo "セッション名: $SESSION"
echo "ワーカー数: $WORKERS"
echo "総件数: $TOTAL"
echo "チャンクサイズ: $chunk_size"
echo "W&B有効: $WANDB_ENABLED"
if [ "$WANDB_ENABLED" = true ]; then
  echo "W&Bプロジェクト: $WANDB_PROJECT"
fi
echo ""

# 既存セッションがあれば終了
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "既存セッション '$SESSION' を終了します..."
  tmux kill-session -t "$SESSION"
fi

# 最初のペイン
created=false
for ((i=0; i<WORKERS; i++)); do
  start=$(( i * chunk_size ))
  end=$(( start + chunk_size ))
  if [ "$start" -ge "$TOTAL" ]; then
    break
  fi
  if [ "$end" -gt "$TOTAL" ]; then
    end=$TOTAL
  fi
  
  end_minus_1=$(( end - 1 ))
  
  echo "ワーカー $i: $start - $end_minus_1 (${end_minus_1}件)"
  
  # コマンド生成
  cmd=$(make_cmd "$start" "$end" "$i")
  cmd=${cmd/__PROJECT_ROOT__/$PROJECT_ROOT}
  cmd=${cmd/__START__/$start}
  cmd=${cmd/__END__/$end}
  cmd=${cmd/__END_MINUS_1__/$end_minus_1}
  cmd=${cmd/__WORKER_ID__/$i}
  cmd=${cmd/__WANDB_PROJECT__/$WANDB_PROJECT}
  
  if [ "$created" = false ]; then
    echo "セッション '$SESSION' を作成中..."
    tmux new-session -d -s "$SESSION" "$cmd"
    created=true
  else
    echo "ペイン $i を追加中..."
    tmux split-window -t "$SESSION":0 -v "$cmd"
  fi
done

# レイアウト整形
echo "レイアウトを整形中..."
tmux select-layout -t "$SESSION":0 tiled >/dev/null

echo ""
echo "=== セットアップ完了 ==="
echo "tmux セッション '$SESSION' を開始しました。"
echo ""
echo "使用方法:"
echo "  - アタッチ: tmux attach -t '$SESSION'"
echo "  - セッション一覧: tmux list-sessions"
echo "  - セッション終了: tmux kill-session -t '$SESSION'"
echo ""
echo "各ワーカーの出力ファイル:"
for ((i=0; i<WORKERS; i++)); do
  start=$(( i * chunk_size ))
  end=$(( start + chunk_size ))
  if [ "$start" -ge "$TOTAL" ]; then
    break
  fi
  if [ "$end" -gt "$TOTAL" ]; then
    end=$TOTAL
  fi
  end_minus_1=$(( end - 1 ))
  echo "  ワーカー $i: data/dna/think_tagged_${start}_${end_minus_1}.jsonl"
done

if [ "$WANDB_ENABLED" = true ]; then
  echo ""
  echo "Weights & Biases:"
  echo "  プロジェクト: $WANDB_PROJECT"
  echo "  ダッシュボード: https://wandb.ai/[ユーザー名]/$WANDB_PROJECT"
fi

echo ""
echo "アタッチします..."
tmux attach -t "$SESSION"


