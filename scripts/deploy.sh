#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.10+ first." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo ""
echo "Deployment complete. Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Next steps:"
echo "  1) Prepare config.yaml (see repo root)"
echo "  2) Build index: python 01_index.py"
echo "  3) Chat: python 02_chat.py"
