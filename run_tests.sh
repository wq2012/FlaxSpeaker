#!/bin/bash
set -o errexit

pytype .
flake8 .
pytest tests.py
