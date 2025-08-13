#!/usr/bin/env bash
set -euo pipefail

# =======================
# Settings (edit freely)
# =======================
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-float16}"
MAX_LEN="${MAX_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"               # 使うVRAM割合
GEN_CFG="${GEN_CFG:-vllm}"                 # "vllm" で統一サンプリング、空ならモデル推奨を使用
LOG_DIR="${LOG_DIR:-./logs}"
PID_FILE="${PID_FILE:-./vllm.pid}"

# Optional caches（速度安定用）
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# OpenAI互換クライアント用（アプリ側が参照）
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://osk-gpu54:${PORT}/v1}"

mkdir -p "${LOG_DIR}"

# =======================
# Helpers
# =======================
healthcheck() {
  curl -s "http://osk-gpu54:${PORT}/health" | grep -q "OK"
}

already_running() {
  # 既にポートが使用中か確認（vLLM以外が使ってる可能性もあるので注意）
  ss -ltn | awk '{print $4}' | grep -q ":${PORT}$"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

# =======================
# Stop if running
# =======================
if [[ -f "${PID_FILE}" ]] && ps -p "$(cat ${PID_FILE})" > /dev/null 2>&1; then
  echo "[INFO] vLLM already running (PID=$(cat ${PID_FILE}))."
  echo "       Stop it with: kill \$(cat ${PID_FILE}) && rm ${PID_FILE}"
  exit 0
fi

if already_running; then
  die "Port ${PORT} is already in use. Change PORT or stop the process using it."
fi

# =======================
# Build args
# =======================
EXTRA_ARGS=()
EXTRA_ARGS+=(--model "${MODEL}")
EXTRA_ARGS+=(--host "${HOST}")
EXTRA_ARGS+=(--port "${PORT}")
EXTRA_ARGS+=(--dtype "${DTYPE}")
EXTRA_ARGS+=(--max-model-len "${MAX_LEN}")
EXTRA_ARGS+=(--gpu-memory-utilization "${GPU_UTIL}")
EXTRA_ARGS+=(--max-log-len 2048)   # ログ肥大抑制（任意）

# generation-config を強制したい場合だけ付与
if [[ -n "${GEN_CFG}" ]]; then
  EXTRA_ARGS+=(--generation-config "${GEN_CFG}")
fi

# 量子化を使うなら（対応モデルのみ）例:
# EXTRA_ARGS+=(--quantization awq)

# =======================
# Launch
# =======================
LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Starting vLLM: ${MODEL} on ${HOST}:${PORT}"
echo "[INFO] Logs: ${LOG_FILE}"
echo "[INFO] OPENAI_BASE_URL=${OPENAI_BASE_URL}"

# conda を使っているなら（必要に応じて）:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate llmbench

nohup python -m vllm.entrypoints.openai.api_server \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"

# =======================
# Wait for health
# =======================
echo -n "[INFO] Waiting for server health"
for _ in $(seq 1 60); do
  if healthcheck; then
    echo ""
    echo "[OK] vLLM is healthy at http://osk-gpu54:${PORT}"
    echo "[OK] Swagger UI:            http://osk-gpu54:${PORT}/docs"
    echo "[OK] OpenAI base url:       ${OPENAI_BASE_URL}"
    exit 0
  fi
  echo -n "."
  sleep 1
done

echo ""
echo "[WARN] Health check did not pass in 60s. Tail logs:"
tail -n 100 "${LOG_FILE}" || true
exit 1
