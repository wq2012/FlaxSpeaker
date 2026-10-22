#!/bin/bash
set -o errexit

if command -v pytype >/dev/null 2>&1; then
  pytype .
elif [ -n "${PYTYPE_BIN:-}" ] && [ -x "$PYTYPE_BIN" ]; then
  PATH="$(dirname "$PYTYPE_BIN"):$PATH" "$PYTYPE_BIN" .
fi
flake8 .
pytest tests.py
