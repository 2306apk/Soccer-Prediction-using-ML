"""
model_training.py  --  Person 2, ML model training & pipeline construction

Provides reusable components for:
- Loading processed EPL data and metadata feature groups
- Creating leak-free chronological splits (Train < Validation < Final Test)
- Creating expanding-window chronological cross-validation folds
- Creating paper-style random 80:20 splits
- Building reproducible scikit-learn pipelines with inside-fold imputation and scaling
- Instantiating Model Families:
    1. Multinomial Logistic Regression
    2. Polynomial SVM (degree 5 baseline and tuned alternatives)
    3. RBF SVM
    4. Gaussian Discriminant Analysis (LDA & QDA)
    5. One-hidden-layer Neural Network (MLPClassifier)
    6. Random Forest
    7. Gradient Boosting
    8. HistGradientBoosting
    9. Soft Voting Ensemble
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_preprocessing import PROCESSED_DIR, read_processed

# Explicit target class ordering required across all evaluations
CLASS_ORDER = ["H", "D", "A"]

# Chronological season sequence (10 seasons: 2009/10 - 2018/19)
SEASONS_ORDER = ["0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819"]

# Chronological partition:
# Training: 7 seasons (0910 to 1516) - 2,657 matches (~70.1%)
# Validation: 1 season (1617) - 379 matches (~10.0%) - strictly for tuning & model selection
# Final Test: 2 seasons (1718 to 1819) - 754 matches (~19.9%) - held out until final evaluation
CHRONO_TRAIN_SEASONS = ["0910", "1011", "1112", "1213", "1314", "1415", "1516"]
CHRONO_VAL_SEASONS = ["1617"]
CHRONO_TEST_SEASONS = ["1718", "1819"]

ID_AND_TARGET_COLS = {
    "paper_row", "match_id", "fixture_id", "season", "date",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "result", "label"
}


def load_dataset_and_feature_groups(
    processed_dir: Path = PROCESSED_DIR,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Load features_engineered.csv and feature_groups.json safely."""
    df = read_processed("features_engineered.csv", processed_dir)
    with open(Path(processed_dir) / "feature_groups.json", "r", encoding="utf-8") as f:
        groups = json.load(f)
    return df, groups


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "result",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract predictor matrix X and target y, validating against target leakage."""
    # Strict validation: never allow ID or target-derived columns in X
    forbidden = set(feature_cols) & ID_AND_TARGET_COLS
    if forbidden:
        raise ValueError(
            f"Target leakage detected! Feature columns contain forbidden columns: {forbidden}"
        )

    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Requested features not present in dataset: {missing_cols}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Validate target values
    unique_targets = set(y.unique())
    if not unique_targets.issubset(set(CLASS_ORDER)):
        raise ValueError(
            f"Target column contains unexpected classes: {unique_targets}. Expected subset of {CLASS_ORDER}"
        )

    return X, y


# --------------------------------------------------------------------------
# Splitting Strategies
# --------------------------------------------------------------------------
def get_chronological_split(
    df: pd.DataFrame,
    train_seasons: Optional[List[str]] = None,
    val_seasons: Optional[List[str]] = None,
    test_seasons: Optional[List[str]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return boolean masks (train_mask, val_mask, test_mask) partition by season.

    Guarantees:
      max(date[train]) < min(date[val]) < min(date[test])
    """
    train_s = train_seasons or CHRONO_TRAIN_SEASONS
    val_s = val_seasons or CHRONO_VAL_SEASONS
    test_s = test_seasons or CHRONO_TEST_SEASONS

    train_mask = df["season"].isin(train_s)
    val_mask = df["season"].isin(val_s)
    test_mask = df["season"].isin(test_s)

    # Sanity checks
    assert (train_mask & val_mask).sum() == 0, "Train and validation overlap!"
    assert (train_mask & test_mask).sum() == 0, "Train and test overlap!"
    assert (val_mask & test_mask).sum() == 0, "Validation and test overlap!"

    train_max_date = df.loc[train_mask, "date"].max()
    val_min_date = df.loc[val_mask, "date"].min()
    val_max_date = df.loc[val_mask, "date"].max()
    test_min_date = df.loc[test_mask, "date"].min()

    if not (train_max_date < val_min_date <= val_max_date < test_min_date):
        raise ValueError(
            f"Chronological boundary violation: "
            f"Train max date ({train_max_date}) must precede Val min date ({val_min_date}), "
            f"which must precede Test min date ({test_min_date})."
        )

    return train_mask, val_mask, test_mask


def get_chronological_cv(
    df_train: pd.DataFrame,
    n_splits: int = 3,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate expanding-window season-based CV splits strictly respecting time order.

    For CHRONO_TRAIN_SEASONS (7 seasons: 0910..1516):
      n_splits = 3:
        Fold 0: Train seasons 0910-1213 (4 seasons), Val season 1314
        Fold 1: Train seasons 0910-1314 (5 seasons), Val season 1415
        Fold 2: Train seasons 0910-1415 (6 seasons), Val season 1516
    """
    available_seasons = [s for s in SEASONS_ORDER if s in df_train["season"].unique()]
    n_total = len(available_seasons)
    if n_total < n_splits + 1:
        raise ValueError(
            f"Not enough seasons ({n_total}) for {n_splits} expanding chronological splits."
        )

    val_seasons_list = available_seasons[-n_splits:]
    df_train_reset = df_train.reset_index(drop=True)

    for i, val_season in enumerate(val_seasons_list):
        train_seasons = available_seasons[: available_seasons.index(val_season)]
        train_idx = df_train_reset.index[df_train_reset["season"].isin(train_seasons)].to_numpy()
        val_idx = df_train_reset.index[df_train_reset["season"] == val_season].to_numpy()
        yield train_idx, val_idx


def get_paper_style_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) using the paper's random 80:20 split convention."""
    indices = np.arange(len(df))
    strat = df["result"].values if stratify else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    return train_idx, test_idx


# --------------------------------------------------------------------------
# Pipeline Construction
# --------------------------------------------------------------------------
def build_model_pipeline(
    model_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a leak-free scikit-learn Pipeline with median imputation and scaling.

    Supported model_name:
      - 'logistic_regression'
      - 'poly_svm'
      - 'rbf_svm'
      - 'linear_svm'
      - 'lda'
      - 'qda'
      - 'mlp'
      - 'random_forest'
      - 'gradient_boosting'
      - 'hist_gradient_boosting'
    """
    params = hyperparameters.copy() if hyperparameters else {}
    needs_scaling = True

    if model_name == "logistic_regression":
        base_clf = LogisticRegression(
            solver=params.pop("solver", "lbfgs"),
            max_iter=params.pop("max_iter", 1000),
            C=params.pop("C", 1.0),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            **params,
        )
    elif model_name == "poly_svm":
        base_clf = SVC(
            kernel="poly",
            degree=params.pop("degree", 5),
            coef0=params.pop("coef0", 1.0),
            C=params.pop("C", 1.0),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            **params,
        )
    elif model_name == "rbf_svm":
        base_clf = SVC(
            kernel="rbf",
            C=params.pop("C", 1.0),
            gamma=params.pop("gamma", "scale"),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            **params,
        )
    elif model_name == "linear_svm":
        base_clf = SVC(
            kernel="linear",
            C=params.pop("C", 1.0),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            **params,
        )
    elif model_name == "lda":
        # Gaussian Discriminant Analysis with shared covariance matrix
        solver = params.pop("solver", "svd")
        shrinkage = params.pop("shrinkage", None)
        if solver == "svd":
            shrinkage = None  # shrinkage only works with lsqr/eigen
        base_clf = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            **params,
        )
    elif model_name == "qda":
        # Gaussian Discriminant Analysis with class-specific covariance matrices
        # reg_param provides shrinkage toward diagonal to prevent singular covariance
        base_clf = QuadraticDiscriminantAnalysis(
            reg_param=params.pop("reg_param", 0.05),
            **params,
        )
    elif model_name == "mlp":
        base_clf = MLPClassifier(
            hidden_layer_sizes=params.pop("hidden_layer_sizes", (64,)),
            activation=params.pop("activation", "relu"),
            alpha=params.pop("alpha", 0.001),
            max_iter=params.pop("max_iter", 400),
            early_stopping=params.pop("early_stopping", True),
            random_state=random_state,
            **params,
        )
    elif model_name == "random_forest":
        needs_scaling = False
        base_clf = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 150),
            max_depth=params.pop("max_depth", 8),
            min_samples_leaf=params.pop("min_samples_leaf", 4),
            max_features=params.pop("max_features", "sqrt"),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
    elif model_name == "gradient_boosting":
        needs_scaling = False
        base_clf = GradientBoostingClassifier(
            n_estimators=params.pop("n_estimators", 100),
            learning_rate=params.pop("learning_rate", 0.05),
            max_depth=params.pop("max_depth", 3),
            min_samples_leaf=params.pop("min_samples_leaf", 4),
            random_state=random_state,
            **params,
        )
    elif model_name == "hist_gradient_boosting":
        needs_scaling = False
        base_clf = HistGradientBoostingClassifier(
            max_iter=params.pop("max_iter", 100),
            learning_rate=params.pop("learning_rate", 0.05),
            max_leaf_nodes=params.pop("max_leaf_nodes", 31),
            min_samples_leaf=params.pop("min_samples_leaf", 20),
            l2_regularization=params.pop("l2_regularization", 0.0),
            class_weight=params.pop("class_weight", None),
            random_state=random_state,
            **params,
        )
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'")

    steps: List[Tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", base_clf))

    return Pipeline(steps)


def build_voting_ensemble(
    estimators: List[Tuple[str, BaseEstimator]],
    voting: str = "soft",
    weights: Optional[List[float]] = None,
) -> VotingClassifier:
    """Construct a voting ensemble from validated candidate pipelines."""
    return VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=weights,
        n_jobs=-1,
    )


def main() -> None:
    print("=" * 70)
    print("SRC.MODEL_TRAINING: DATASET AND CHRONOLOGICAL PIPELINE OVERVIEW")
    print("=" * 70)
    df, groups = load_dataset_and_feature_groups()
    print(f"Total engineered matches: {len(df)}")
    train_mask, val_mask, test_mask = get_chronological_split(df)
    print(f"  Training seasons:   {CHRONO_TRAIN_SEASONS} ({train_mask.sum()} matches)")
    print(f"  Validation seasons: {CHRONO_VAL_SEASONS} ({val_mask.sum()} matches)")
    print(f"  Test seasons:       {CHRONO_TEST_SEASONS} ({test_mask.sum()} matches)")
    print("\nFeature Groups available:")
    for gname in ["paper_F1", "paper_F2", "historical_only", "historical_only_extended", "historical_plus_ratings_sensitivity"]:
        if gname in groups:
            print(f"  - {gname:<35} ({len(groups[gname])} features)")
    print("\nVerifying model pipeline instantiation:")
    for model_name in ["logistic_regression", "poly_svm", "rbf_svm", "lda", "qda", "mlp", "random_forest", "hist_gradient_boosting"]:
        pipe = build_model_pipeline(model_name, random_state=42)
        print(f"  ✓ Pipeline for '{model_name}' built successfully ({len(pipe.steps)} steps)")
    print("=" * 70)


if __name__ == "__main__":
    main()
