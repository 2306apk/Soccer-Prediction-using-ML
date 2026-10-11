"""
test_model_inference.py  --  Tests for model serialization and inference interface.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.model_inference import EPLMatchPredictor, save_trained_model
from src.model_training import build_model_pipeline


def test_model_serialization_and_roundtrip(tmp_path: Path):
    """Verify that a fitted pipeline can be serialized and reloaded with exact predictions."""
    rng = np.random.default_rng(42)
    feature_cols = ["feat_a", "feat_b", "feat_c"]
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=feature_cols)
    y = pd.Series(rng.choice(["H", "D", "A"], size=40))

    pipeline = build_model_pipeline("logistic_regression", {"C": 1.0}, random_state=42)
    pipeline.fit(X, y)

    model_file, meta_file = save_trained_model(
        pipeline=pipeline,
        feature_columns=feature_cols,
        model_name="TestLogisticRegression",
        feature_group="test_group",
        training_period="Test Train",
        validation_period="Test Val",
        test_period="Test Test",
        hyperparameters={"C": 1.0},
        validation_metrics={"accuracy": 0.5},
        test_metrics={"accuracy": 0.5},
        output_dir=tmp_path,
    )

    assert model_file.exists()
    assert meta_file.exists()

    predictor = EPLMatchPredictor.load(model_file, meta_file)
    assert predictor.feature_columns == feature_cols

    # Test single prediction
    sample_dict = {"feat_a": 0.5, "feat_b": -0.2, "feat_c": 1.1}
    pred_res = predictor.predict_single(sample_dict)
    assert "predicted_outcome" in pred_res
    assert pred_res["predicted_outcome"] in ["H", "D", "A"]
    assert "prob_H" in pred_res


def test_inference_feature_reordering_and_missing_handling(tmp_path: Path):
    """Verify that predictor handles column reordering and rejects missing required columns."""
    feature_cols = ["f1", "f2", "f3"]
    X = pd.DataFrame(np.ones((20, 3)), columns=feature_cols)
    y = pd.Series((["H", "D", "A"] * 7)[:20])

    pipe = build_model_pipeline("poly_svm", random_state=42)
    pipe.fit(X, y)

    model_file, meta_file = save_trained_model(
        pipeline=pipe,
        feature_columns=feature_cols,
        model_name="TestSVM",
        feature_group="test_group",
        training_period="Train",
        validation_period="Val",
        test_period="Test",
        hyperparameters={},
        validation_metrics={},
        test_metrics={},
        output_dir=tmp_path,
    )

    predictor = EPLMatchPredictor.load(model_file, meta_file)

    # Scrambled column order should be aligned properly
    df_scrambled = pd.DataFrame({"f3": [1.0], "f1": [2.0], "f2": [3.0]})
    pred = predictor.predict(df_scrambled)
    assert len(pred) == 1

    # Missing required column must raise ValueError
    df_incomplete = pd.DataFrame({"f1": [2.0], "f2": [3.0]})
    with pytest.raises(ValueError, match="missing 1 required feature"):
        predictor.predict(df_incomplete)
