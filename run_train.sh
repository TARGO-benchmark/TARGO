#!/usr/bin/env bash
# Train TARGO-Net. All arguments are passed to train_targo.py.
# Set TARGO_PYTHON to choose the interpreter (default: python3 on PATH).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TARGO_PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "No Python interpreter found. Set TARGO_PYTHON=/path/to/python." >&2
  exit 127
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
cd "$ROOT"
exec "$PYTHON_BIN" "$ROOT/train_targo.py" "$@"
