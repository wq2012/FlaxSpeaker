#!/bin/bash
set -o errexit

if command -v pytype >/dev/null 2>&1; then
  pytype .
elif [ -x /tmp/flax_py310_venv/bin/pytype ]; then
  PATH="/tmp/flax_py310_venv/bin:$PATH" /tmp/flax_py310_venv/bin/pytype .
fi
flake8 .
pytest tests.py
