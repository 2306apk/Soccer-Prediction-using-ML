"""
test_chronological_split.py  --  Tests for chronological splitting and temporal integrity.
"""
import pandas as pd
import pytest

from src.data_preprocessing import PROCESSED_DIR
from src.model_training import (
    CHRONO_TEST_SEASONS,
    CHRONO_TRAIN_SEASONS,
    CHRONO_VAL_SEASONS,
    get_chronological_cv,
    get_chronological_split,
    load_dataset_and_feature_groups,
)

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "features_engineered.csv").exists(),
    reason="run `bash run_person1.sh` first",
)


def test_chronological_split_partition():
    """Verify that train, validation, and test sets form a strict non-overlapping partition."""
    df, _ = load_dataset_and_feature_groups()
    train_mask, val_mask, test_mask = get_chronological_split(df)

    # Disjointness
    assert (train_mask & val_mask).sum() == 0
    assert (train_mask & test_mask).sum() == 0
    assert (val_mask & test_mask).sum() == 0

    # Exhaustive partition across all seasons
    total_split_rows = train_mask.sum() + val_mask.sum() + test_mask.sum()
    assert total_split_rows == len(df)

    # Correct seasons
    assert set(df.loc[train_mask, "season"].unique()) == set(CHRONO_TRAIN_SEASONS)
    assert set(df.loc[val_mask, "season"].unique()) == set(CHRONO_VAL_SEASONS)
    assert set(df.loc[test_mask, "season"].unique()) == set(CHRONO_TEST_SEASONS)


def test_chronological_ordering_strict_time_arrow():
    """Verify max(date[train]) < min(date[val]) and max(date[val]) < min(date[test])."""
    df, _ = load_dataset_and_feature_groups()
    train_mask, val_mask, test_mask = get_chronological_split(df)

    train_max_date = df.loc[train_mask, "date"].max()
    val_min_date = df.loc[val_mask, "date"].min()
    val_max_date = df.loc[val_mask, "date"].max()
    test_min_date = df.loc[test_mask, "date"].min()

    assert train_max_date < val_min_date, f"Train max ({train_max_date}) >= Val min ({val_min_date})"
    assert val_max_date < test_min_date, f"Val max ({val_max_date}) >= Test min ({test_min_date})"


def test_chronological_cv_folds():
    """Verify expanding-window cross-validation on the training seasons."""
    df, _ = load_dataset_and_feature_groups()
    train_mask, _, _ = get_chronological_split(df)
    df_train = df[train_mask].reset_index(drop=True)

    cv = list(get_chronological_cv(df_train, n_splits=3))
    assert len(cv) == 3

    prev_train_len = 0
    for fold_i, (tr_idx, va_idx) in enumerate(cv):
        assert len(set(tr_idx) & set(va_idx)) == 0, f"Fold {fold_i} has train/val overlap!"
        # Expanding window: training set size must strictly increase
        assert len(tr_idx) > prev_train_len
        prev_train_len = len(tr_idx)

        # Dates in train must strictly precede dates in validation fold
        tr_max = df_train.iloc[tr_idx]["date"].max()
        va_min = df_train.iloc[va_idx]["date"].min()
        assert tr_max < va_min, f"Fold {fold_i}: tr_max ({tr_max}) >= va_min ({va_min})"
