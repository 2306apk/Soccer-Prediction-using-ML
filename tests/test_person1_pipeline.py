"""Automated tests for Person 1's pipeline.  Run:  pytest -q   (after building data/processed)."""
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import PROCESSED_DIR, RAW_DIR, read_processed, verify_alignment_with_paper_vectors
from src.improved_features import build_history_features, feature_groups
from src.leakage_checks import (check_brute_force, check_chronology, check_no_target_columns,
                                check_perturbation, check_prematch_ratings,
                                check_primary_group_separation)
from src.paper_features import verify_paper_features

pytestmark = pytest.mark.skipif(not (PROCESSED_DIR / "features_engineered.csv").exists(),
                                reason="run `bash run_person1.sh` first")
RAW_PAPER_AVAILABLE = (RAW_DIR / "paper_repo" / "match_vectors.csv").exists()
requires_raw_paper = pytest.mark.skipif(
    not RAW_PAPER_AVAILABLE,
    reason="raw paper repository is not downloaded; run `bash run_person1.sh` to fetch it",
)


@pytest.fixture(scope="module")
def fixtures():
    return read_processed("fixtures_full.csv")


@pytest.fixture(scope="module")
def matches():
    return read_processed("matches_clean.csv")


@pytest.fixture(scope="module")
def eng():
    return read_processed("features_engineered.csv")


@pytest.fixture(scope="module")
def hist(fixtures):
    return build_history_features(fixtures)


def test_dataset_shape_and_labels(fixtures, matches):
    assert len(fixtures) == 3800
    assert len(matches) == 3790                                   # 3,791 paper rows minus 1 duplicated fixture
    assert matches["season"].str.len().eq(4).all()                # '0910', not 910
    gd = matches["home_goals"] - matches["away_goals"]
    assert (matches["result"] == np.where(gd > 0, "H", np.where(gd == 0, "D", "A"))).all()
    assert not matches.duplicated(["season", "home_team", "away_team"]).any()


@requires_raw_paper
def test_labels_match_paper_vectors(matches):
    r = verify_alignment_with_paper_vectors(matches)
    assert r["kept_rows_with_label_mismatch"] == 0


@requires_raw_paper
def test_f2_and_f3_reproduced_exactly():
    r = verify_paper_features()
    assert r["F2"]["all_cells_match"] and r["F3"]["all_cells_match"]


def test_chronology(fixtures):
    c = check_chronology(fixtures)
    assert c == {"sorted_by_date": True, "team_twice_same_day": 0, "teams_not_38_games_per_season": 0}


def test_features_match_brute_force(fixtures, hist):
    assert check_brute_force(fixtures, hist, n=150)["mismatches"] == 0


def test_perturbation_no_leakage(fixtures, hist):
    r = check_perturbation(fixtures, hist, n_trials=3)
    assert r["own_features_changed"] == 0
    assert r["earlier_or_same_day_features_changed"] == 0
    assert r["later_matches_of_involved_teams_unchanged"] == 0     # test must not be vacuous


def test_first_matches_have_no_history_and_start_elo(hist, fixtures):
    first_day = fixtures.loc[fixtures["date"] == fixtures["date"].min(), "fixture_id"]
    h = hist.loc[first_day]
    assert (h["n_prev_h"] == 0).all() and (h["n_prev_a"] == 0).all()
    assert h["h_points_last5"].isna().all()
    assert (h["h_elo"] == 1500).all() and (h["a_elo"] == 1500).all()
    assert (h["elo_exp_home"] > 0.5).all()                         # home advantage at equal Elo


def test_elo_is_zero_sum_within_a_season(hist, fixtures):
    """Between regressions the Elo exchange is zero-sum: league mean stays 1500 in 2009/10."""
    s1 = fixtures.loc[fixtures["season"] == "0910"].sort_values(["date", "fixture_id"])
    last_day = s1["date"].max()
    last = hist.loc[s1.loc[s1["date"] == last_day, "fixture_id"]]
    assert 1450 < pd.concat([last["h_elo"], last["a_elo"]]).mean() < 1550


def test_no_target_columns_in_feature_groups(eng):
    g = feature_groups(list(eng.columns))
    assert not check_no_target_columns(g)["groups_with_target_like_columns"]
    assert all(len(v) > 0 for k, v in g.items())


def test_prematch_ratings_use_only_previous_season(eng):
    r = check_prematch_ratings(eng)
    assert r["0910_rows_all_nan"]


@requires_raw_paper
def test_engineered_rows_aligned_with_paper_features(eng, matches):
    """The F2 columns in the engineered table must equal the paper's released vector for the same match."""
    mv = pd.read_csv(RAW_DIR / "paper_repo" / "match_vectors.csv", skipinitialspace=True)
    f2 = mv.drop(columns="label")
    cols = list(f2.columns)
    expect = f2.iloc[eng["paper_row"].to_numpy()][cols].to_numpy()
    assert np.allclose(eng[cols].to_numpy(), expect)
    assert (eng["label"].to_numpy() == mv["label"].iloc[eng["paper_row"].to_numpy()].to_numpy()).all()


def test_historical_only_group_excludes_paper_and_lineup_sensitive_features(eng):
    """The primary group must contain only chronological match-history features."""
    groups = feature_groups(list(eng.columns))
    primary = groups["historical_only"]
    assert primary
    assert not any(c.startswith(("pm_", "diff_roster_rating", "diff_gk_rating",
                                 "diff_def_rating", "diff_mid_rating", "diff_off_rating"))
                   for c in primary)
    assert not any(c in {"h_roster_rating", "a_roster_rating", "h_gk_rating",
                          "h_def_rating", "h_mid_rating", "h_off_rating",
                          "a_gk_rating", "a_def_rating", "a_mid_rating", "a_off_rating"}
                   for c in primary)
    assert set(primary).issubset(set(groups["historical_only_extended"]))


def test_paper_and_sensitivity_groups_are_separate(eng):
    groups = feature_groups(list(eng.columns))
    assert set(groups["paper_F1"]).issubset(groups["paper_F2"])
    assert groups["prematch_ratings_sensitivity"]
    assert set(groups["prematch_ratings_sensitivity"]).isdisjoint(groups["historical_only"])
    assert set(groups["historical_differences"]).isdisjoint(
        {"diff_roster_rating", "diff_gk_rating", "diff_def_rating",
         "diff_mid_rating", "diff_off_rating"}
    )


def test_primary_group_separation_check_passes(eng):
    groups = feature_groups(list(eng.columns))
    report = check_primary_group_separation(groups)
    assert report["passed"]
    assert report["paper_or_lineup_sensitive_features_in_primary"] == []
