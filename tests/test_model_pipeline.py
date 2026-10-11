"""
test_model_pipeline.py  --  Tests for Person 2's model pipelines, metrics, and leakage checks.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_preprocessing import PROCESSED_DIR
from src.model_evaluation import (
    CLASS_ORDER,
    compute_confusion_matrix,
    compute_metrics,
    confusion_matrix_to_dataframe,
)
from src.model_training import (
    build_model_pipeline,
    load_dataset_and_feature_groups,
    prepare_xy,
)

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "features_engineered.csv").exists(),
    reason="run `bash run_person1.sh` first",
)


def test_load_dataset_and_feature_groups():
    df, groups = load_dataset_and_feature_groups()
    assert len(df) == 3790
    assert "historical_only" in groups
    assert "historical_only_extended" in groups
    assert len(groups["historical_only"]) == 47
    assert len(groups["historical_only_extended"]) == 101


def test_prepare_xy_leakage_protection():
    df, groups = load_dataset_and_feature_groups()
    # Attempting to include any target-derived or ID column must immediately raise ValueError
    for bad_col in ["result", "label", "home_goals", "away_goals", "match_id", "fixture_id"]:
        bad_features = groups["paper_F1"] + [bad_col]
        with pytest.raises(ValueError, match="Target leakage detected"):
            prepare_xy(df, bad_features)


def test_prepare_xy_valid_target_labels():
    df, groups = load_dataset_and_feature_groups()
    X, y = prepare_xy(df, groups["paper_F1"], target_col="result")
    assert set(y.unique()).issubset(set(CLASS_ORDER))
    assert len(X) == len(y) == 3790


def test_pipeline_fitting_with_missing_values():
    """Verify that model pipelines gracefully handle missing values via inside-fold median imputation."""
    # Synthetic small dataset with NaNs
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 5
    X_synthetic = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Inject missing values
    X_synthetic.iloc[0, 0] = np.nan
    X_synthetic.iloc[5, 2] = np.nan
    X_synthetic.iloc[10, 4] = np.nan

    y_synthetic = pd.Series(rng.choice(["H", "D", "A"], size=n_samples))

    for model_name in ["logistic_regression", "poly_svm", "random_forest", "hist_gradient_boosting"]:
        pipe = build_model_pipeline(model_name, random_state=42)
        assert isinstance(pipe, Pipeline)
        pipe.fit(X_synthetic, y_synthetic)
        preds = pipe.predict(X_synthetic)
        assert len(preds) == n_samples
        assert set(preds).issubset({"H", "D", "A"})


def test_metric_generation_and_order():
    """Verify metrics calculation and 3x3 confusion matrix with row=actual, col=predicted."""
    y_true = pd.Series(["H", "H", "D", "D", "A", "A"])
    y_pred = pd.Series(["H", "D", "D", "A", "A", "H"])

    metrics = compute_metrics(y_true, y_pred, prefix="test")
    assert "test_accuracy" in metrics
    assert "test_macro_f1" in metrics
    assert "test_draw_recall" in metrics
    assert "test_draw_precision" in metrics

    # Accuracy is 3 / 6 = 0.5
    assert metrics["test_accuracy"] == 0.5

    cm = compute_confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    assert cm.shape == (3, 3)
    # CLASS_ORDER: H, D, A
    # Actual H (2 items): 1 predicted H, 1 predicted D, 0 predicted A -> [1, 1, 0]
    assert (cm[0] == [1, 1, 0]).all()
    # Actual D (2 items): 0 predicted H, 1 predicted D, 1 predicted A -> [0, 1, 1]
    assert (cm[1] == [0, 1, 1]).all()
    # Actual A (2 items): 1 predicted H, 0 predicted D, 1 predicted A -> [1, 0, 1]
    assert (cm[2] == [1, 0, 1]).all()

    cm_df = confusion_matrix_to_dataframe(cm, labels=CLASS_ORDER)
    assert list(cm_df.index) == ["Actual_H", "Actual_D", "Actual_A"]
    assert list(cm_df.columns) == ["Pred_H", "Pred_D", "Pred_A"]


def test_reproducibility():
    """Verify that pipelines initialized with the same random_state yield identical predictions."""
    df, groups = load_dataset_and_feature_groups()
    X = df[groups["paper_F1"]].head(100)
    y = df["result"].head(100)

    pipe1 = build_model_pipeline("random_forest", random_state=123)
    pipe2 = build_model_pipeline("random_forest", random_state=123)

    pipe1.fit(X, y)
    pipe2.fit(X, y)

    preds1 = pipe1.predict(X)
    preds2 = pipe2.predict(X)
    assert (preds1 == preds2).all()
