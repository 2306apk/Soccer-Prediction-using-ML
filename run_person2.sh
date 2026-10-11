#!/usr/bin/env bash
set -euo pipefail

# Person 2: Complete ML Model Training and Evaluation Pipeline Runner

# Check for virtual environment first
if [ -f ".venv/bin/python" ]; then
    PYTHON=(.venv/bin/python)
elif command -v python >/dev/null 2>&1; then
    PYTHON=(python)
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=(python3)
else
    echo "ERROR: Python was not found."
    exit 1
fi

echo "================================================================="
echo "Running Person 2 Pipeline with: ${PYTHON[*]}"
echo "================================================================="

echo "Step 1: Running Model Training Component Overview..."
"${PYTHON[@]}" -m src.model_training

echo "Step 2: Running Protocol A (E0: Paper Reproduction)..."
"${PYTHON[@]}" scripts/run_paper_reproduction.py

echo "Step 3: Running Protocol B (E1-E5: Chronological Experiments & Model Serialization)..."
"${PYTHON[@]}" scripts/run_model_experiments.py

echo "Step 4: Running Production Inference Demo..."
"${PYTHON[@]}" scripts/demo_inference.py --fixture_index 0

echo "Step 5: Executing Complete Automated Test Suite..."
"${PYTHON[@]}" -m pytest -v

echo "================================================================="
echo "Person 2 Pipeline Finished Successfully!"
echo "Master Comparison Table: results/tables/model_comparison.csv"
echo "Serialized Model:        results/models/epl_model.joblib"
echo "Evaluation Figures:      results/figures/"
echo "================================================================="
