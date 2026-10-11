"""
demo_inference.py  --  Person 2, Interactive & CLI demo inference script

Demonstrates loading the serialized production model (epl_model.joblib),
running inference on held-out match fixtures from the 2018/19 test season,
displaying predicted outcomes (H / D / A) with soft-voting predicted class probabilities
and tactical match context.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data_preprocessing import read_processed
from src.model_inference import EPLMatchPredictor

MODELS_DIR = ROOT / "results" / "models"


def main():
    parser = argparse.ArgumentParser(description="Run EPL match outcome prediction demo.")
    parser.add_argument(
        "--fixture_index",
        type=int,
        default=0,
        help="Index of test fixture to predict (0 to 10)",
    )
    args = parser.parse_args()

    model_file = MODELS_DIR / "epl_model.joblib"
    meta_file = MODELS_DIR / "model_metadata.json"

    if not model_file.exists() or not meta_file.exists():
        print(f"Error: Trained model not found in {MODELS_DIR}.")
        print("Please run `python scripts/run_model_experiments.py` first to train and serialize the model.")
        sys.exit(1)

    print("=" * 75)
    print("EPL MATCH OUTCOME PREDICTION — PRODUCTION INFERENCE DEMO")
    print("=" * 75)

    predictor = EPLMatchPredictor.load(model_file, meta_file)
    print(f"Loaded Champion Model: {predictor.model_name}")
    print(f"Feature Track:         {predictor.feature_group} ({len(predictor.feature_columns)} features)")
    print(f"Training Period:       {predictor.metadata.get('training_period', 'Historical seasons')}")
    print(f"Target Classes:        {predictor.classes}")
    print("-" * 75)

    # Load test fixtures
    df = read_processed("features_engineered.csv")
    test_matches = df[df["season"] == "1819"].sort_values(["date", "fixture_id"]).reset_index(drop=True)

    if test_matches.empty:
        print("No test fixtures found for season 2018/19.")
        sys.exit(1)

    idx = min(max(0, args.fixture_index), len(test_matches) - 1)
    match_row = test_matches.iloc[[idx]]

    fixture_id = match_row["fixture_id"].values[0]
    date_str = pd.to_datetime(match_row["date"].values[0]).strftime("%Y-%m-%d")
    home_team = match_row["home_team"].values[0]
    away_team = match_row["away_team"].values[0]
    actual_result = match_row["result"].values[0]
    home_goals = match_row["home_goals"].values[0]
    away_goals = match_row["away_goals"].values[0]

    # Pre-match context features
    h_elo = match_row["h_elo"].values[0] if "h_elo" in match_row else None
    a_elo = match_row["a_elo"].values[0] if "a_elo" in match_row else None
    elo_diff = match_row["elo_diff"].values[0] if "elo_diff" in match_row else None

    print(f"Fixture:       {home_team} vs {away_team}")
    print(f"Match Date:    {date_str} (Fixture #{fixture_id})")
    if h_elo and a_elo:
        print(f"Pre-Match Elo: {home_team} ({h_elo:.1f}) vs {away_team} ({a_elo:.1f}) [Diff: {elo_diff:+.1f}]")
    print("-" * 75)

    # Execute Prediction
    predictions = predictor.predict(match_row)
    pred_outcome = predictions["predicted_outcome"].values[0]
    pred_full = predictions["predicted_label_full"].values[0]

    prob_h = predictions["prob_H"].values[0] if "prob_H" in predictions else None
    prob_d = predictions["prob_D"].values[0] if "prob_D" in predictions else None
    prob_a = predictions["prob_A"].values[0] if "prob_A" in predictions else None

    print("PREDICTION RESULT:")
    print(f"  Predicted Outcome:   {pred_outcome} ({pred_full})")
    if prob_h is not None:
        print(f"  Probability Home:    {prob_h * 100:.1f}%")
        print(f"  Probability Draw:    {prob_d * 100:.1f}%")
        print(f"  Probability Away:    {prob_a * 100:.1f}%")
    print("-" * 75)
    print(f"ACTUAL MATCH OUTCOME:  {actual_result} ({home_team} {home_goals} - {away_goals} {away_team})")
    correct = (pred_outcome == actual_result)
    print(f"STATUS:                {'✓ Correct Prediction' if correct else '✗ Misclassification'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
