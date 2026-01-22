#!/usr/bin/env bash
# 部署脚本：安装依赖并提示后续步骤
set -euo pipefail

# 定位仓库根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 检查 Python 版本
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ first." >&2
  exit 1
fi

# 安装依赖
python3 -m pip install --upgrade pip
python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

# 输出提示
echo ""
echo "Deployment complete."
echo "Next steps:"
echo "  1) Prepare config.yaml (see repo root)"
echo "  2) Build index: python 01_index.py"
echo "  3) Chat: python 02_chat.py"
echo "  4) Stability eval: python 04_stability.py"
