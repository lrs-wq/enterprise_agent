#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="/Users/uenheng/Library/Application Support/GogelilaAI/company-ai-agent"

mkdir -p "$RUNTIME_DIR" "$RUNTIME_DIR/data"

for directory in src static templates policies; do
  rsync -a --delete "$PROJECT_DIR/$directory/" "$RUNTIME_DIR/$directory/"
done

install -m 600 "$PROJECT_DIR/.env" "$RUNTIME_DIR/.env"

if [[ ! -f "$RUNTIME_DIR/data/knowledge.db" ]]; then
  rsync -a "$PROJECT_DIR/data/" "$RUNTIME_DIR/data/"
fi

printf '公网运行副本已同步到：%s\n' "$RUNTIME_DIR"
