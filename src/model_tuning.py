"""
model_tuning.py  --  Person 2, Hyperparameter tuning and validation

Performs leak-free model tuning strictly on the training partition:
- Uses expanding-window chronological CV across training seasons for Protocol B
- Avoids evaluating on the final test partition during tuning
- Supports grid evaluation with tracking of:
    * Mean CV accuracy
    * Mean CV macro-F1
    * Mean CV draw recall
    * Training time per configuration
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.model_evaluation import CLASS_ORDER
from src.model_training import build_model_pipeline, get_chronological_cv


def evaluate_config_on_cv(
    model_name: str,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_train_meta: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    """Evaluate a single hyperparameter configuration across expanding chronological folds."""
    cv_generator = get_chronological_cv(df_train_meta, n_splits=n_splits)

    fold_accs = []
    fold_macro_f1s = []
    fold_draw_recalls = []

    t0 = time.time()
    for train_idx, val_idx in cv_generator:
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        pipe = build_model_pipeline(model_name, hyperparameters=params, random_state=random_state)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)

        fold_accs.append(accuracy_score(y_va, preds))
        fold_macro_f1s.append(f1_score(y_va, preds, labels=CLASS_ORDER, average="macro", zero_division=0))
        # Draw is at index 1 in CLASS_ORDER
        recalls = recall_score(y_va, preds, labels=CLASS_ORDER, average=None, zero_division=0)
        fold_draw_recalls.append(recalls[1])

    elapsed = time.time() - t0

    return {
        "mean_cv_accuracy": float(np.mean(fold_accs)),
        "mean_cv_macro_f1": float(np.mean(fold_macro_f1s)),
        "mean_cv_draw_recall": float(np.mean(fold_draw_recalls)),
        "cv_time_seconds": round(elapsed, 2),
    }


def grid_search_chronological(
    model_name: str,
    param_grid: Dict[str, List[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_train_meta: pd.DataFrame,
    n_splits: int = 3,
    primary_metric: str = "mean_cv_macro_f1",
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
    """Exhaustive grid search across chronological folds.

    Returns:
      (best_params, best_scores, all_results_records)
    """
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys])) if keys else [{}]

    all_results = []
    best_params = {}
    best_score = -1.0
    best_metrics = {}

    for combo in combinations:
        params = dict(zip(keys, combo)) if keys else {}
        scores = evaluate_config_on_cv(
            model_name=model_name,
            params=params,
            X_train=X_train,
            y_train=y_train,
            df_train_meta=df_train_meta,
            n_splits=n_splits,
            random_state=random_state,
        )

        record = {**params, **scores}
        all_results.append(record)

        current_score = scores.get(primary_metric, 0.0)
        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_metrics = scores

    return best_params, best_metrics, all_results
