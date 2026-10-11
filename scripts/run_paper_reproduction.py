"""
run_paper_reproduction.py  --  Person 2, Experiment E0: Paper-style Reproduction

Reproduces the evaluation protocol of:
"Gaining a Statistical Edge in Soccer Prediction using Machine Learning:
 Role of Meta Statistics in Match Prediction"

Paper Protocol (Protocol A):
- Random 80:20 train/test split
- Feature sets:
    * paper_F1: 2 features (overall squad ratings)
    * paper_F2: 10 features (overall + position-wise ratings)
    * paper_F3: 280 features (detailed match player statistics)
- Model families:
    * Polynomial-kernel SVM (degree 5 and degree 3)
    * RBF-kernel SVM
    * Multinomial Logistic Regression (SoftMax)
    * Gaussian Discriminant Analysis (LinearDiscriminantAnalysis & QuadraticDiscriminantAnalysis)
    * One-hidden-layer Neural Network (MLPClassifier)

Reference from paper:
  Paper reports 58.89% test accuracy on F1 with degree-5 polynomial SVM.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data_preprocessing import PROCESSED_DIR, TABLES_DIR, read_processed
from src.model_evaluation import (
    CLASS_ORDER,
    compute_confusion_matrix,
    compute_metrics,
    confusion_matrix_to_dataframe,
    plot_metric_comparison,
)
from src.model_training import build_model_pipeline, get_paper_style_split, prepare_xy

FIG_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "results" / "models"


def run_paper_reproduction(random_state: int = 42) -> pd.DataFrame:
    print("=" * 80)
    print("EXPERIMENT E0: PAPER-STYLE REPRODUCTION (PROTOCOL A - RANDOM 80:20 SPLIT)")
    print("=" * 80)

    # 1. Load engineered dataset and paper F3 dataset
    df_eng = read_processed("features_engineered.csv")
    with open(PROCESSED_DIR / "feature_groups.json", "r", encoding="utf-8") as f:
        groups = json.load(f)

    # paper_f3.csv alignment
    f3_path = PROCESSED_DIR / "paper_f3.csv"
    has_f3 = f3_path.exists()
    if has_f3:
        df_f3 = pd.read_csv(f3_path)
        f3_feature_cols = [
            c for c in df_f3.columns
            if c not in {"paper_row", "fixture_id", "date", "label", "result"}
        ]
        # Align targets
        label_map = {-1: "A", 0: "D", 1: "H"}
        df_f3["result"] = df_f3["label"].map(label_map)
    else:
        df_f3 = None
        f3_feature_cols = []

    # 2. Split indices (random 80:20 stratified)
    train_idx, test_idx = get_paper_style_split(df_eng, test_size=0.20, random_state=random_state)
    print(f"Total dataset: {len(df_eng)} matches")
    print(f"Train split: {len(train_idx)} matches (80%) | Test split: {len(test_idx)} matches (20%)")
    print("-" * 80)

    # 3. Define feature sets and models to evaluate
    feature_sets = {
        "paper_F1": groups["paper_F1"],
        "paper_F2": groups["paper_F2"],
    }
    if has_f3:
        feature_sets["paper_F3"] = f3_feature_cols

    models_to_test = [
        ("Poly SVM (deg 5)", "poly_svm", {"degree": 5, "C": 1.0, "coef0": 1.0}),
        ("Poly SVM (deg 3)", "poly_svm", {"degree": 3, "C": 1.0, "coef0": 1.0}),
        ("RBF SVM", "rbf_svm", {"C": 1.0, "gamma": "scale"}),
        ("Multinomial Logistic Reg", "logistic_regression", {"C": 1.0, "max_iter": 1000}),
        ("GDA (LDA)", "lda", {"solver": "svd"}),
        ("GDA (QDA)", "qda", {"reg_param": 0.1}),
        ("Neural Network (MLP 1-layer)", "mlp", {"hidden_layer_sizes": (64,), "max_iter": 400}),
    ]

    results_list: List[Dict[str, Any]] = []
    cm_records: List[pd.DataFrame] = []

    for fset_name, fcols in feature_sets.items():
        print(f"\nEvaluating Feature Set: {fset_name} ({len(fcols)} features)")
        source_df = df_f3 if fset_name == "paper_F3" else df_eng
        X, y = prepare_xy(source_df, fcols, target_col="result")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        for display_name, m_type, m_params in models_to_test:
            # Skip QDA on F3 if dimensionality exceeds sample limits or causes singularity
            if fset_name == "paper_F3" and m_type == "qda":
                continue

            t0 = time.time()
            try:
                pipeline = build_model_pipeline(m_type, hyperparameters=m_params, random_state=random_state)
                pipeline.fit(X_train, y_train)
                elapsed = time.time() - t0

                train_preds = pipeline.predict(X_train)
                test_preds = pipeline.predict(X_test)

                tr_metrics = compute_metrics(y_train, train_preds, prefix="training")
                te_metrics = compute_metrics(y_test, test_preds, prefix="test")

                # Confusion matrix
                cm = compute_confusion_matrix(y_test, test_preds, labels=CLASS_ORDER)
                cm_df = confusion_matrix_to_dataframe(cm, labels=CLASS_ORDER)
                cm_df["model_name"] = f"{fset_name}_{display_name}"
                cm_records.append(cm_df)

                rec = {
                    "experiment_id": "E0_paper_reproduction",
                    "evaluation_protocol": "Protocol_A_random_80_20",
                    "feature_group": fset_name,
                    "model_name": display_name,
                    "training_period": "Random 80% split (3,032 matches)",
                    "validation_period": "None (Random test split evaluation)",
                    "test_period": "Random 20% split (758 matches)",
                    "selected_hyperparameters": str(m_params),
                    "training_accuracy": tr_metrics["training_accuracy"],
                    "validation_accuracy": np.nan,
                    "test_accuracy": te_metrics["test_accuracy"],
                    "test_macro_f1": te_metrics["test_macro_f1"],
                    "test_weighted_f1": te_metrics["test_weighted_f1"],
                    "home_precision": te_metrics["test_home_precision"],
                    "home_recall": te_metrics["test_home_recall"],
                    "home_f1": te_metrics["test_home_f1"],
                    "draw_precision": te_metrics["test_draw_precision"],
                    "draw_recall": te_metrics["test_draw_recall"],
                    "draw_f1": te_metrics["test_draw_f1"],
                    "away_precision": te_metrics["test_away_precision"],
                    "away_recall": te_metrics["test_away_recall"],
                    "away_f1": te_metrics["test_away_f1"],
                    "training_time_seconds": round(elapsed, 2),
                }
                results_list.append(rec)
                print(
                    f"  {display_name:<28} | Train Acc: {tr_metrics['training_accuracy']:.4f} | "
                    f"Test Acc: {te_metrics['test_accuracy']:.4f} | Macro-F1: {te_metrics['test_macro_f1']:.4f} | "
                    f"Draw Recall: {te_metrics['test_draw_recall']:.4f} | Time: {elapsed:.2f}s"
                )

            except Exception as e:
                print(f"  FAILED {display_name} on {fset_name}: {e}")
                results_list.append({
                    "experiment_id": "E0_paper_reproduction",
                    "evaluation_protocol": "Protocol_A_random_80_20",
                    "feature_group": fset_name,
                    "model_name": display_name,
                    "training_period": "Random 80%",
                    "validation_period": "None",
                    "test_period": "Random 20%",
                    "selected_hyperparameters": str(m_params),
                    "training_accuracy": np.nan,
                    "validation_accuracy": np.nan,
                    "test_accuracy": np.nan,
                    "test_macro_f1": np.nan,
                    "test_weighted_f1": np.nan,
                    "home_precision": np.nan,
                    "home_recall": np.nan,
                    "home_f1": np.nan,
                    "draw_precision": np.nan,
                    "draw_recall": np.nan,
                    "draw_f1": np.nan,
                    "away_precision": np.nan,
                    "away_recall": np.nan,
                    "away_f1": np.nan,
                    "training_time_seconds": round(time.time() - t0, 2),
                })

    df_results = pd.DataFrame(results_list)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(TABLES_DIR / "paper_reproduction_results.csv", index=False)

    if cm_records:
        df_cms = pd.concat(cm_records)
        df_cms.to_csv(TABLES_DIR / "e0_confusion_matrices.csv")

    # Plot comparison of E0
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    f1_results = df_results[df_results["feature_group"] == "paper_F1"].copy()
    if not f1_results.empty:
        plot_metric_comparison(
            df=f1_results,
            metric_col="test_accuracy",
            title="E0 Paper Reproduction: Test Accuracy on F1 (Ref: 58.89%)",
            output_path=FIG_DIR / "e0_paper_reproduction_accuracy.png",
            ylabel="Test Accuracy",
            group_col="model_name",
            baseline_val=0.5889,
        )

    print("\n" + "=" * 80)
    print("E0 SUMMARY: PAPER REFERENCE COMPARISON")
    print("=" * 80)
    f1_poly5 = df_results[
        (df_results["feature_group"] == "paper_F1") & (df_results["model_name"] == "Poly SVM (deg 5)")
    ]
    if not f1_poly5.empty:
        obs_acc = f1_poly5["test_accuracy"].values[0]
        print(f"Paper Reported F1 + Poly SVM (deg 5): 58.89% (0.5889)")
        print(f"Our Observed F1 + Poly SVM (deg 5):   {obs_acc * 100:.2f}% ({obs_acc:.4f})")
        print(f"Delta from paper reference:           {(obs_acc - 0.5889) * 100:+.2f} percentage points")
    print(f"Outputs written to:")
    print(f"  - {TABLES_DIR / 'paper_reproduction_results.csv'}")
    print(f"  - {FIG_DIR / 'e0_paper_reproduction_accuracy.png'}")
    print("=" * 80)

    return df_results


if __name__ == "__main__":
    run_paper_reproduction()
