#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Virtualenv non trovato in $VENV_PY"
  echo "Crea il venv con: python3 -m venv .venv"
  exit 1
fi

"$VENV_PY" "$ROOT/models/cross_validation.py"
