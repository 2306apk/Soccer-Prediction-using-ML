#!/usr/bin/env bash
set -euo pipefail

# Detect the available Python interpreter
if command -v python >/dev/null 2>&1; then
    PYTHON=(python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
elif command -v py >/dev/null 2>&1; then
    PYTHON=(py -3)
else
    echo "ERROR: Python was not found."
    echo "Install Python 3 and ensure it is available in your terminal PATH."
    exit 1
fi

echo "Using Python: ${PYTHON[*]}"

"${PYTHON[@]}" -m pip install -r requirements.txt

"${PYTHON[@]}" -m src.data_preprocessing --fetch --build
"${PYTHON[@]}" -m src.improved_features
"${PYTHON[@]}" -m src.leakage_checks
"${PYTHON[@]}" -m src.feature_analysis
"${PYTHON[@]}" -m pytest -q