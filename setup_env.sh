#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"
python "${REPO_ROOT}/yoloe/install_yoloe.py" --skip-core-deps "$@"
