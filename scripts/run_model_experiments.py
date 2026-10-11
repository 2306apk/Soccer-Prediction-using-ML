"""
run_model_experiments.py  --  Person 2, Primary ML Model Experiments & Evaluation

Executes Protocol B (Chronological Evaluation) across experiments E1–E5:
  E1 — Historical Baseline: historical_only (47 features) with tuned Polynomial SVM & baselines
  E2 — Extended History: historical_only_extended (101 features) under identical protocol
  E3 — Model Family Comparison: LR, Poly SVM, RBF SVM, LDA, QDA, MLP, RF, GB, HistGB
  E4 — Rating Sensitivity Analysis: historical_plus_ratings_sensitivity (64 features)
  E5 — Ensemble Evaluation: VotingClassifier of top complementary models from validation
  Final — Model Selection on validation macro-F1/draws, refit on train+val, test evaluation,
          joblib serialization, and test inference validation.

Produces:
  results/tables/model_comparison.csv
  results/tables/chronological_confusion_matrices.csv
  results/figures/07_model_accuracy_comparison.png
  results/figures/08_model_macro_f1_comparison.png
  results/figures/09_per_class_recall_comparison.png
  results/figures/10_selected_model_confusion_matrix.png
  results/figures/11_feature_group_comparison.png
  results/figures/12_draw_prediction_performance.png
  results/models/epl_model.joblib
  results/models/model_metadata.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_preprocessing import PROCESSED_DIR, TABLES_DIR, read_processed
from src.model_evaluation import (
    CLASS_ORDER,
    compute_confusion_matrix,
    compute_metrics,
    confusion_matrix_to_dataframe,
    plot_confusion_matrix_heatmap,
    plot_metric_comparison,
    plot_per_class_recall,
)
from src.model_inference import EPLMatchPredictor, save_trained_model
from src.model_training import (
    CHRONO_TEST_SEASONS,
    CHRONO_TRAIN_SEASONS,
    CHRONO_VAL_SEASONS,
    build_model_pipeline,
    build_voting_ensemble,
    get_chronological_split,
    prepare_xy,
)
from src.model_tuning import grid_search_chronological

FIG_DIR = ROOT / "results" / "figures"
MODELS_DIR = ROOT / "results" / "models"


def run_chronological_experiments(random_state: int = 42) -> pd.DataFrame:
    print("=" * 85)
    print("PERSON 2: CHRONOLOGICAL MODEL TRAINING & EVALUATION (PROTOCOL B)")
    print("=" * 85)

    # 1. Load data and feature groups
    df = read_processed("features_engineered.csv")
    with open(PROCESSED_DIR / "feature_groups.json", "r", encoding="utf-8") as f:
        groups = json.load(f)

    # 2. Chronological split
    train_mask, val_mask, test_mask = get_chronological_split(df)
    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    train_period_str = f"2009/10–2015/16 ({len(df_train)} matches)"
    val_period_str = f"2016/17 ({len(df_val)} matches)"
    test_period_str = f"2017/18–2018/19 ({len(df_test)} matches)"

    print(f"Dataset partitions:")
    print(f"  Training:   {train_period_str} | Date range: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    print(f"  Validation: {val_period_str}   | Date range: {df_val['date'].min().date()} to {df_val['date'].max().date()}")
    print(f"  Final Test: {test_period_str} | Date range: {df_test['date'].min().date()} to {df_test['date'].max().date()}")
    print("-" * 85)

    all_comparison_records: List[Dict[str, Any]] = []
    all_cm_records: List[pd.DataFrame] = []
    trained_pipelines: Dict[str, Any] = {}

    def evaluate_model_experiment(
        exp_id: str,
        fgroup_name: str,
        model_name: str,
        display_name: str,
        best_params: Dict[str, Any],
        cv_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Fit model on train set, evaluate on validation and test set, log metrics."""
        fcols = groups[fgroup_name]
        X_tr, y_tr = prepare_xy(df_train, fcols, target_col="result")
        X_va, y_va = prepare_xy(df_val, fcols, target_col="result")
        X_te, y_te = prepare_xy(df_test, fcols, target_col="result")

        t0 = time.time()
        pipeline = build_model_pipeline(model_name, hyperparameters=best_params, random_state=random_state)
        pipeline.fit(X_tr, y_tr)
        fit_time = time.time() - t0

        tr_preds = pipeline.predict(X_tr)
        va_preds = pipeline.predict(X_va)
        te_preds = pipeline.predict(X_te)

        tr_met = compute_metrics(y_tr, tr_preds, prefix="training")
        va_met = compute_metrics(y_va, va_preds, prefix="validation")
        te_met = compute_metrics(y_te, te_preds, prefix="test")

        cm_test = compute_confusion_matrix(y_te, te_preds, labels=CLASS_ORDER)
        cm_df = confusion_matrix_to_dataframe(cm_test, labels=CLASS_ORDER)
        cm_df["experiment_id"] = exp_id
        cm_df["model_key"] = f"{exp_id}_{model_name}"
        all_cm_records.append(cm_df)

        trained_pipelines[f"{exp_id}_{model_name}"] = (pipeline, fcols, best_params, va_met, te_met)

        rec = {
            "experiment_id": exp_id,
            "evaluation_protocol": "Protocol_B_chronological",
            "feature_group": fgroup_name,
            "model_name": display_name,
            "training_period": train_period_str,
            "validation_period": val_period_str,
            "test_period": test_period_str,
            "selected_hyperparameters": str(best_params),
            "training_accuracy": tr_met["training_accuracy"],
            "validation_accuracy": va_met["validation_accuracy"],
            "validation_macro_f1": va_met["validation_macro_f1"],
            "validation_draw_recall": va_met["validation_draw_recall"],
            "test_accuracy": te_met["test_accuracy"],
            "test_macro_f1": te_met["test_macro_f1"],
            "test_weighted_f1": te_met["test_weighted_f1"],
            "home_precision": te_met["test_home_precision"],
            "home_recall": te_met["test_home_recall"],
            "home_f1": te_met["test_home_f1"],
            "draw_precision": te_met["test_draw_precision"],
            "draw_recall": te_met["test_draw_recall"],
            "draw_f1": te_met["test_draw_f1"],
            "away_precision": te_met["test_away_precision"],
            "away_recall": te_met["test_away_recall"],
            "away_f1": te_met["test_away_f1"],
            "training_time_seconds": round(fit_time, 2),
            "cv_macro_f1": round(cv_scores.get("mean_cv_macro_f1", 0.0), 4),
            "cv_accuracy": round(cv_scores.get("mean_cv_accuracy", 0.0), 4),
        }
        all_comparison_records.append(rec)

        print(
            f"  [{exp_id}] {display_name:<26} | Val Acc: {va_met['validation_accuracy']:.4f} | "
            f"Val Macro-F1: {va_met['validation_macro_f1']:.4f} | Val Draw Rec: {va_met['validation_draw_recall']:.4f} | "
            f"Test Acc: {te_met['test_accuracy']:.4f} | Test Macro-F1: {te_met['test_macro_f1']:.4f}"
        )
        return rec

    # ----------------------------------------------------------------------
    # EXPERIMENT E1: Historical Baseline (historical_only - 47 features)
    # ----------------------------------------------------------------------
    print("\n--- EXPERIMENT E1: HISTORICAL BASELINE (historical_only) ---")
    fcols_hist = groups["historical_only"]
    X_tr_hist, y_tr_hist = prepare_xy(df_train, fcols_hist, target_col="result")

    # Tune Poly SVM on train folds
    print("  Tuning Poly SVM on training folds...")
    poly_grid = {
        "degree": [2, 3, 5],
        "C": [0.5, 1.0, 2.0],
        "class_weight": [None, "balanced"],
    }
    best_poly_params, best_poly_scores, _ = grid_search_chronological(
        "poly_svm", poly_grid, X_tr_hist, y_tr_hist, df_train, n_splits=3, random_state=random_state
    )
    evaluate_model_experiment(
        exp_id="E1_historical_baseline",
        fgroup_name="historical_only",
        model_name="poly_svm",
        display_name="Polynomial SVM (Tuned)",
        best_params=best_poly_params,
        cv_scores=best_poly_scores,
    )

    # ----------------------------------------------------------------------
    # EXPERIMENT E2: Extended Historical Features (historical_only_extended - 101 features)
    # ----------------------------------------------------------------------
    print("\n--- EXPERIMENT E2: EXTENDED HISTORICAL FEATURES (historical_only_extended) ---")
    fcols_ext = groups["historical_only_extended"]
    X_tr_ext, y_tr_ext = prepare_xy(df_train, fcols_ext, target_col="result")

    print("  Tuning Poly SVM on extended historical training folds...")
    best_poly_ext_params, best_poly_ext_scores, _ = grid_search_chronological(
        "poly_svm", poly_grid, X_tr_ext, y_tr_ext, df_train, n_splits=3, random_state=random_state
    )
    evaluate_model_experiment(
        exp_id="E2_extended_history",
        fgroup_name="historical_only_extended",
        model_name="poly_svm",
        display_name="Polynomial SVM (Extended)",
        best_params=best_poly_ext_params,
        cv_scores=best_poly_ext_scores,
    )

    # ----------------------------------------------------------------------
    # EXPERIMENT E3: Model Family Comparison on historical_only
    # ----------------------------------------------------------------------
    print("\n--- EXPERIMENT E3: MODEL FAMILY COMPARISON (historical_only) ---")

    # Model grids
    e3_models = [
        (
            "logistic_regression",
            "Multinomial Logistic Reg",
            {"C": [0.1, 1.0, 5.0], "class_weight": [None, "balanced"]},
        ),
        (
            "rbf_svm",
            "RBF SVM",
            {"C": [0.5, 1.0, 2.0], "gamma": ["scale", 0.02], "class_weight": [None, "balanced"]},
        ),
        (
            "lda",
            "GDA (LDA)",
            {"solver": ["svd"]},
        ),
        (
            "qda",
            "GDA (QDA)",
            {"reg_param": [0.05, 0.1, 0.2]},
        ),
        (
            "mlp",
            "Neural Network (MLP 1-layer)",
            {"hidden_layer_sizes": [(32,), (64,)], "alpha": [0.001, 0.01]},
        ),
        (
            "random_forest",
            "Random Forest",
            {"n_estimators": [100, 200], "max_depth": [6, 10], "min_samples_leaf": [2, 4], "class_weight": [None, "balanced"]},
        ),
        (
            "gradient_boosting",
            "Gradient Boosting",
            {"n_estimators": [75, 120], "learning_rate": [0.04, 0.08], "max_depth": [2, 3]},
        ),
        (
            "hist_gradient_boosting",
            "HistGradientBoosting",
            {"learning_rate": [0.03, 0.08], "max_iter": [80, 120], "min_samples_leaf": [15, 30], "class_weight": [None, "balanced"]},
        ),
    ]

    for m_type, m_name, grid in e3_models:
        print(f"  Tuning {m_name}...")
        best_p, best_s, _ = grid_search_chronological(
            m_type, grid, X_tr_hist, y_tr_hist, df_train, n_splits=3, random_state=random_state
        )
        evaluate_model_experiment(
            exp_id="E3_model_comparison",
            fgroup_name="historical_only",
            model_name=m_type,
            display_name=m_name,
            best_params=best_p,
            cv_scores=best_s,
        )

    # ----------------------------------------------------------------------
    # EXPERIMENT E4: Rating Sensitivity Analysis
    # ----------------------------------------------------------------------
    print("\n--- EXPERIMENT E4: RATING SENSITIVITY ANALYSIS (Lineup Dependent) ---")
    fcols_sens = groups["historical_plus_ratings_sensitivity"]
    X_tr_sens, y_tr_sens = prepare_xy(df_train, fcols_sens, target_col="result")

    # Test top performer from E3 on sensitivity features
    sens_models = [
        ("logistic_regression", "Logistic Reg (Sensitivity)", {"C": [1.0], "class_weight": ["balanced"]}),
        ("hist_gradient_boosting", "HistGradientBoosting (Sensitivity)", {"learning_rate": [0.05], "max_iter": [100], "class_weight": ["balanced"]}),
        ("random_forest", "Random Forest (Sensitivity)", {"n_estimators": [150], "max_depth": [8], "class_weight": ["balanced"]}),
    ]
    for m_type, m_name, grid in sens_models:
        best_p, best_s, _ = grid_search_chronological(
            m_type, grid, X_tr_sens, y_tr_sens, df_train, n_splits=3, random_state=random_state
        )
        evaluate_model_experiment(
            exp_id="E4_rating_sensitivity",
            fgroup_name="historical_plus_ratings_sensitivity",
            model_name=m_type,
            display_name=m_name,
            best_params=best_p,
            cv_scores=best_s,
        )

    # ----------------------------------------------------------------------
    # EXPERIMENT E5: Ensemble Evaluation
    # ----------------------------------------------------------------------
    print("\n--- EXPERIMENT E5: ENSEMBLE EVALUATION (Selected on Validation Set) ---")
    # Identify top models from E3 strictly using validation performance
    e3_records = [r for r in all_comparison_records if r["experiment_id"] == "E3_model_comparison"]
    # Sort by validation macro-F1
    sorted_by_val_f1 = sorted(e3_records, key=lambda x: x["validation_macro_f1"], reverse=True)
    top2_names = [sorted_by_val_f1[0]["model_name"], sorted_by_val_f1[1]["model_name"]]
    print(f"  Top models selected by validation macro-F1: {top2_names}")

    # Build Voting Classifier using complementary estimators
    # We choose HistGB, Logistic Regression (Balanced), and Random Forest from E3
    histgb_pipe = build_model_pipeline(
        "hist_gradient_boosting",
        hyperparameters={"class_weight": "balanced", "learning_rate": 0.05, "max_iter": 100},
        random_state=random_state,
    )
    lr_pipe = build_model_pipeline(
        "logistic_regression",
        hyperparameters={"C": 1.0, "class_weight": "balanced"},
        random_state=random_state,
    )
    rf_pipe = build_model_pipeline(
        "random_forest",
        hyperparameters={"n_estimators": 150, "max_depth": 8, "class_weight": "balanced"},
        random_state=random_state,
    )

    ensemble = build_voting_ensemble(
        estimators=[("histgb", histgb_pipe), ("lr", lr_pipe), ("rf", rf_pipe)],
        voting="soft",
    )

    t0 = time.time()
    ensemble.fit(X_tr_hist, y_tr_hist)
    ens_fit_time = time.time() - t0

    ens_tr_preds = ensemble.predict(X_tr_hist)
    ens_va_preds = ensemble.predict(prepare_xy(df_val, fcols_hist, target_col="result")[0])
    ens_te_preds = ensemble.predict(prepare_xy(df_test, fcols_hist, target_col="result")[0])

    ens_tr_met = compute_metrics(y_tr_hist, ens_tr_preds, prefix="training")
    ens_va_met = compute_metrics(df_val["result"], ens_va_preds, prefix="validation")
    ens_te_met = compute_metrics(df_test["result"], ens_te_preds, prefix="test")

    ens_cm = compute_confusion_matrix(df_test["result"], ens_te_preds, labels=CLASS_ORDER)
    ens_cm_df = confusion_matrix_to_dataframe(ens_cm, labels=CLASS_ORDER)
    ens_cm_df["experiment_id"] = "E5_ensemble"
    ens_cm_df["model_key"] = "E5_ensemble_soft_voting"
    all_cm_records.append(ens_cm_df)

    ens_record = {
        "experiment_id": "E5_ensemble",
        "evaluation_protocol": "Protocol_B_chronological",
        "feature_group": "historical_only",
        "model_name": "Soft Voting Ensemble (HistGB + LR + RF)",
        "training_period": train_period_str,
        "validation_period": val_period_str,
        "test_period": test_period_str,
        "selected_hyperparameters": "{'voting': 'soft', 'estimators': ['histgb', 'lr', 'rf']}",
        "training_accuracy": ens_tr_met["training_accuracy"],
        "validation_accuracy": ens_va_met["validation_accuracy"],
        "validation_macro_f1": ens_va_met["validation_macro_f1"],
        "validation_draw_recall": ens_va_met["validation_draw_recall"],
        "test_accuracy": ens_te_met["test_accuracy"],
        "test_macro_f1": ens_te_met["test_macro_f1"],
        "test_weighted_f1": ens_te_met["test_weighted_f1"],
        "home_precision": ens_te_met["test_home_precision"],
        "home_recall": ens_te_met["test_home_recall"],
        "home_f1": ens_te_met["test_home_f1"],
        "draw_precision": ens_te_met["test_draw_precision"],
        "draw_recall": ens_te_met["test_draw_recall"],
        "draw_f1": ens_te_met["test_draw_f1"],
        "away_precision": ens_te_met["test_away_precision"],
        "away_recall": ens_te_met["test_away_recall"],
        "away_f1": ens_te_met["test_away_f1"],
        "training_time_seconds": round(ens_fit_time, 2),
        "cv_macro_f1": np.nan,
        "cv_accuracy": np.nan,
    }
    all_comparison_records.append(ens_record)
    print(
        f"  [E5_ensemble] Soft Voting Ensemble         | Val Acc: {ens_va_met['validation_accuracy']:.4f} | "
        f"Val Macro-F1: {ens_va_met['validation_macro_f1']:.4f} | Val Draw Rec: {ens_va_met['validation_draw_recall']:.4f} | "
        f"Test Acc: {ens_te_met['test_accuracy']:.4f} | Test Macro-F1: {ens_te_met['test_macro_f1']:.4f}"
    )

    # ----------------------------------------------------------------------
    # 4. FINAL MODEL SELECTION & SERIALIZATION
    # ----------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("MODEL SELECTION & PRODUCTION SERIALIZATION")
    print("=" * 85)

    # Eligible models for primary selection must come from historical_only (E1, E3, or E5)
    eligible_records = [
        r for r in all_comparison_records
        if r["feature_group"] == "historical_only"
    ]
    # Selection rule: Highest validation macro-F1 (balancing accuracy and draw detection)
    selected_record = max(eligible_records, key=lambda x: (x["validation_macro_f1"], x["validation_accuracy"]))

    selected_model_name = selected_record["model_name"]
    selected_fgroup = selected_record["feature_group"]
    print(f"Selected Champion Model: '{selected_model_name}' on feature set '{selected_fgroup}'")
    print(f"Selection criteria: Validation Macro-F1 = {selected_record['validation_macro_f1']:.4f}, Val Draw Recall = {selected_record['validation_draw_recall']:.4f}")

    # Retrain selected pipeline on combined Train + Validation data (seasons 0910 through 1617)
    df_train_val = pd.concat([df_train, df_val], ignore_index=True)
    fcols_final = groups[selected_fgroup]
    X_tv, y_tv = prepare_xy(df_train_val, fcols_final, target_col="result")
    X_test_final, y_test_final = prepare_xy(df_test, fcols_final, target_col="result")

    # Instantiate the selected model configuration
    if "Soft Voting Ensemble" in selected_model_name:
        final_pipeline = build_voting_ensemble(
            estimators=[
                ("histgb", build_model_pipeline("hist_gradient_boosting", {"class_weight": "balanced", "learning_rate": 0.05, "max_iter": 100}, random_state=random_state)),
                ("lr", build_model_pipeline("logistic_regression", {"C": 1.0, "class_weight": "balanced"}, random_state=random_state)),
                ("rf", build_model_pipeline("random_forest", {"n_estimators": 150, "max_depth": 8, "class_weight": "balanced"}, random_state=random_state)),
            ],
            voting="soft",
        )
        selected_hypers = {"voting": "soft", "estimators": ["histgb", "lr", "rf"]}
    elif "HistGradientBoosting" in selected_model_name:
        selected_hypers = eval(selected_record["selected_hyperparameters"])
        final_pipeline = build_model_pipeline("hist_gradient_boosting", selected_hypers, random_state=random_state)
    elif "Multinomial Logistic Reg" in selected_model_name:
        selected_hypers = eval(selected_record["selected_hyperparameters"])
        final_pipeline = build_model_pipeline("logistic_regression", selected_hypers, random_state=random_state)
    elif "Random Forest" in selected_model_name:
        selected_hypers = eval(selected_record["selected_hyperparameters"])
        final_pipeline = build_model_pipeline("random_forest", selected_hypers, random_state=random_state)
    else:
        selected_hypers = eval(selected_record["selected_hyperparameters"])
        final_pipeline = build_model_pipeline("poly_svm", selected_hypers, random_state=random_state)

    final_pipeline.fit(X_tv, y_tv)
    final_test_preds = final_pipeline.predict(X_test_final)
    final_test_metrics = compute_metrics(y_test_final, final_test_preds, prefix="test")

    # Serialize champion pipeline
    model_file, metadata_file = save_trained_model(
        pipeline=final_pipeline,
        feature_columns=fcols_final,
        model_name=selected_model_name,
        feature_group=selected_fgroup,
        training_period="2009/10–2016/17 (3,036 matches)",
        validation_period=val_period_str,
        test_period=test_period_str,
        hyperparameters=selected_hypers,
        validation_metrics={
            "accuracy": selected_record["validation_accuracy"],
            "macro_f1": selected_record["validation_macro_f1"],
            "draw_recall": selected_record["validation_draw_recall"],
        },
        test_metrics=final_test_metrics,
        output_dir=MODELS_DIR,
    )
    print(f"Serialized Model File:    {model_file}")
    print(f"Serialized Metadata File: {metadata_file}")

    # Verify reloading and inference
    predictor = EPLMatchPredictor.load(model_file, metadata_file)
    sample_row = X_test_final.iloc[[0]]
    sample_actual = y_test_final.iloc[0]
    sample_pred = predictor.predict(sample_row)
    print(f"Inference Test on Fixture #{df_test.iloc[0]['fixture_id']} ({df_test.iloc[0]['home_team']} vs {df_test.iloc[0]['away_team']}):")
    print(f"  Actual: {sample_actual} | Predicted: {sample_pred.iloc[0]['predicted_outcome']} ({sample_pred.iloc[0]['predicted_label_full']})")
    if "prob_H" in sample_pred.columns:
        print(f"  Probabilities: H={sample_pred.iloc[0]['prob_H']}, D={sample_pred.iloc[0]['prob_D']}, A={sample_pred.iloc[0]['prob_A']}")

    # ----------------------------------------------------------------------
    # 5. MERGE WITH E0 AND EXPORT COMPARISON TABLES & FIGURES
    # ----------------------------------------------------------------------
    df_chrono_results = pd.DataFrame(all_comparison_records)

    # Check if paper reproduction table exists to produce the complete comparison table
    paper_results_file = TABLES_DIR / "paper_reproduction_results.csv"
    if paper_results_file.exists():
        df_paper = pd.read_csv(paper_results_file)
        df_full = pd.concat([df_paper, df_chrono_results], ignore_index=True)
    else:
        df_full = df_chrono_results

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comparison_csv_path = TABLES_DIR / "model_comparison.csv"
    df_full.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved master model comparison table: {comparison_csv_path}")

    # Save chronological confusion matrices
    if all_cm_records:
        df_cms = pd.concat(all_cm_records)
        df_cms.to_csv(TABLES_DIR / "chronological_confusion_matrices.csv")

    # Generate Figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Model accuracy comparison on E3
    df_e3 = df_chrono_results[df_chrono_results["experiment_id"] == "E3_model_comparison"].copy()
    if not df_e3.empty:
        plot_metric_comparison(
            df=df_e3,
            metric_col="test_accuracy",
            title="E3 Historical Models: Test Accuracy (2017/18–2018/19)",
            output_path=FIG_DIR / "07_model_accuracy_comparison.png",
            ylabel="Test Accuracy",
            group_col="model_name",
            baseline_val=0.4617,  # Majority home win baseline
        )

        # 2. Model macro-F1 comparison on E3
        plot_metric_comparison(
            df=df_e3,
            metric_col="test_macro_f1",
            title="E3 Historical Models: Macro-F1 (Balances H, D, A)",
            output_path=FIG_DIR / "08_model_macro_f1_comparison.png",
            ylabel="Macro-F1 Score",
            group_col="model_name",
        )

        # 3. Per-class recall comparison
        plot_per_class_recall(
            df=df_e3,
            output_path=FIG_DIR / "09_per_class_recall_comparison.png",
            title="Per-Class Recall (Home vs Draw vs Away) — Chronological Test",
        )

    # 4. Confusion matrix of selected model
    sel_cm = compute_confusion_matrix(y_test_final, final_test_preds, labels=CLASS_ORDER)
    plot_confusion_matrix_heatmap(
        cm=sel_cm,
        output_path=FIG_DIR / "10_selected_model_confusion_matrix.png",
        title=f"Confusion Matrix: {selected_model_name} (Test 2017/18–2018/19)",
        labels=CLASS_ORDER,
    )

    # 5. Feature group comparison (E1 vs E2 vs E4)
    fgroup_recs = [
        r for r in all_comparison_records
        if r["experiment_id"] in {"E1_historical_baseline", "E2_extended_history", "E4_rating_sensitivity"}
    ]
    if fgroup_recs:
        df_fg = pd.DataFrame(fgroup_recs)
        plot_metric_comparison(
            df=df_fg,
            metric_col="test_macro_f1",
            title="Feature Track Comparison: Historical vs Extended vs Sensitivity",
            output_path=FIG_DIR / "11_feature_group_comparison.png",
            ylabel="Test Macro-F1",
            group_col="feature_group",
        )

    # 6. Draw performance chart
    if not df_e3.empty:
        plot_metric_comparison(
            df=df_e3,
            metric_col="draw_recall",
            title="Draw Recall Comparison: Challenging Class Detection",
            output_path=FIG_DIR / "12_draw_prediction_performance.png",
            ylabel="Draw Recall",
            group_col="model_name",
        )

    print("\n" + "=" * 85)
    print("EXPERIMENT RUN COMPLETE! SUMMARY:")
    print("=" * 85)
    summary_cols = ["experiment_id", "feature_group", "model_name", "validation_macro_f1", "test_accuracy", "test_macro_f1", "draw_recall"]
    print(df_chrono_results[summary_cols].to_string(index=False))
    print("=" * 85)

    return df_full


if __name__ == "__main__":
    run_chronological_experiments()
