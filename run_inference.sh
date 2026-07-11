#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${TARGO_PYTHON:-/data3/ran/miniconda3/envs/targo/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "No runnable Python found. Set TARGO_PYTHON=/path/to/python." >&2
    exit 127
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
exec "$PYTHON_BIN" "$ROOT/inference_targo.py" "$@"
