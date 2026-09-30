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
  printf '未找到 Conda。请先安装 Miniforge，再重新运行此脚本。\n' >&2
  exit 1
fi

cd "$PROJECT_DIR"
if ! "$CONDA_BIN" run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  printf '正在创建 Conda 环境 %s...\n' "$ENV_NAME"
  "$CONDA_BIN" env create -f environment.yml
else
  printf 'Conda 环境 %s 已存在，将同步项目依赖。\n' "$ENV_NAME"
fi

"$CONDA_BIN" run -n "$ENV_NAME" python -m pip install -e '.[dev]'

if [[ ! -f .env ]]; then
  cp .env.example .env
  ADMIN_TOKEN_VALUE="$("$CONDA_BIN" run -n "$ENV_NAME" python -c 'import secrets; print(secrets.token_hex(18))')"
  sed -i.bak "s/^AI_AGENT_ADMIN_TOKEN=.*/AI_AGENT_ADMIN_TOKEN=${ADMIN_TOKEN_VALUE}/" .env
  rm -f .env.bak
  printf '已创建 .env，并生成本机管理员令牌。\n'
fi

printf '\n环境安装完成。\n'
printf '环境名称：%s\n' "$ENV_NAME"
printf 'Python：'
"$CONDA_BIN" run -n "$ENV_NAME" python --version
printf '下一步：bash scripts/run-local.sh\n'
