#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="company-ai-agent"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in \
    "/Users/uenheng/miniforge3/bin/conda" \
    "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
    "/opt/homebrew/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  printf '未找到 Conda，请先运行 scripts/setup-local.sh。\n' >&2
  exit 1
fi

if ! curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]] && [[ -d /Applications/Ollama.app ]]; then
    printf '正在启动 Ollama...\n'
    open -a Ollama
  fi
  for _ in {1..30}; do
    if curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  printf 'Ollama 未启动，请先打开 Ollama 应用。\n' >&2
  exit 1
fi

LAN_IP=""
if command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
elif command -v hostname >/dev/null 2>&1; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi

printf '聊天页：http://127.0.0.1:8000/\n'
printf '管理页：http://127.0.0.1:8000/admin\n'
if [[ -n "$LAN_IP" ]]; then
  printf '局域网页：http://%s:8000/\n' "$LAN_IP"
fi
printf '按 Ctrl+C 停止服务。\n\n'

cd "$PROJECT_DIR"
eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$ENV_NAME"
exec python -m uvicorn company_agent.main:app --host 0.0.0.0 --port 8000
