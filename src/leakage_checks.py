"""
leakage_checks.py  --  Person 1, Phase 3.5 (prove the features are chronological and leak-free)

Run:  python -m src.leakage_checks          (needs data/processed built first)

Checks
------
1. chronology        fixtures are date-sorted, no team plays twice a day, 38 games / team / season
2. brute force       form (w=5) and venue form (v=10) for a random sample are recomputed with a
                     plain ``date < kickoff`` filter and must equal the pipeline's values
3. perturbation      change one match's score and rebuild everything:
                       - that match's own features must not change,
                       - no feature of any match on or before that date may change,
                       - later matches of the two teams MUST change (so the test is not vacuous)
4. no target columns no feature group may contain goals / result / label / identifiers
4b. group separation the primary historical-only group excludes paper / lineup-sensitive ratings
5. pre-match ratings 2009/10 has no previous season -> all pm_* values must be NaN there
6. correlations      |Spearman| of every feature with the final goal difference, to flag anything
                     suspiciously strong (a leaked feature would stand out)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.data_preprocessing import PROCESSED_DIR, SEASONS, TABLES_DIR, read_processed
from src.improved_features import build_history_features

TARGET_LIKE = {"home_goals", "away_goals", "result", "label", "paper_row", "match_id", "fixture_id",
               "season", "date", "home_team", "away_team"}


# --------------------------------------------------------------------------
def check_chronology(fixtures: pd.DataFrame) -> dict:
    fx = fixtures.sort_values(["date", "fixture_id"])
    long = pd.concat([fx[["date", "home_team"]].rename(columns={"home_team": "team"}),
                      fx[["date", "away_team"]].rename(columns={"away_team": "team"})])
    per_team = pd.concat([fx[["season", "home_team"]].rename(columns={"home_team": "team"}),
                          fx[["season", "away_team"]].rename(columns={"away_team": "team"})]).groupby(["season", "team"]).size()
    return {"sorted_by_date": bool(fx["date"].is_monotonic_increasing),
            "team_twice_same_day": int(long.duplicated(["date", "team"]).sum()),
            "teams_not_38_games_per_season": int((per_team != 38).sum())}


def _stint_start(team_seasons: dict, team: str, season: str) -> str:
    """First season of the team's current uninterrupted run in the league (history resets after a gap)."""
    idx = SEASONS.index(season)
    start = idx
    while start > 0 and SEASONS[start - 1] in team_seasons[team]:
        start -= 1
    return SEASONS[start]


def check_brute_force(fixtures: pd.DataFrame, hist: pd.DataFrame, n: int = 400, seed: int = 0) -> dict:
    """Independent recomputation (plain date filter, no shared state) of w=5 form and v=10 venue form."""
    rng = np.random.default_rng(seed)
    fx = fixtures.set_index("fixture_id")
    team_seasons = {t: set(pd.concat([fixtures.loc[fixtures.home_team == t, "season"],
                                      fixtures.loc[fixtures.away_team == t, "season"]])) for t in
                    set(fixtures.home_team) | set(fixtures.away_team)}

    def games(team, season, date, venue=None):
        s0 = _stint_start(team_seasons, team, season)
        m = fixtures[(fixtures.season >= s0) & (fixtures.date < date) & ((fixtures.home_team == team) | (fixtures.away_team == team))]
        if venue == "home":
            m = m[m.home_team == team]
        elif venue == "away":
            m = m[m.away_team == team]
        m = m.sort_values(["date", "fixture_id"])
        gf = np.where(m.home_team == team, m.home_goals, m.away_goals).astype(float)
        ga = np.where(m.home_team == team, m.away_goals, m.home_goals).astype(float)
        pts = np.where(gf > ga, 3.0, np.where(gf == ga, 1.0, 0.0))
        return pts, gf, ga

    bad, checked = [], 0
    for fid in rng.choice(fixtures["fixture_id"].to_numpy(), size=n, replace=False):
        r, h = fx.loc[fid], hist.loc[fid]
        for side, team, venue, vpre in (("h", r.home_team, "home", "h_home"), ("a", r.away_team, "away", "a_away")):
            pts, gf, ga = games(team, r.season, r.date)
            n_prev = len(pts)
            exp = {f"n_prev_{side}": n_prev}
            if n_prev:
                w = min(5, n_prev)
                exp[f"{side}_points_last5"] = pts[-5:].sum() * 5 / w
                exp[f"{side}_gf_avg_last5"] = gf[-5:].mean()
                exp[f"{side}_ga_avg_last5"] = ga[-5:].mean()
            vp, vgf, vga = games(team, r.season, r.date, venue)
            if len(vp):
                exp[f"{vpre}_ppg_last10"] = vp[-10:].mean()
                exp[f"{vpre}_gf_avg_last10"] = vgf[-10:].mean()
                exp[f"{vpre}_ga_avg_last10"] = vga[-10:].mean()
            for k, v in exp.items():
                checked += 1
                if not np.isclose(h[k], v, equal_nan=True):
                    bad.append((int(fid), k, float(h[k]), float(v)))
            if n_prev == 0:                                    # no history -> NaN, never a fabricated number
                for k in (f"{side}_points_last5", f"{side}_gf_avg_last5"):
                    checked += 1
                    if not np.isnan(h[k]):
                        bad.append((int(fid), k, float(h[k]), "NaN expected"))
    return {"values_checked": checked, "mismatches": len(bad), "examples": bad[:5]}


def check_perturbation(fixtures: pd.DataFrame, base: pd.DataFrame | None = None, n_trials: int = 8, seed: int = 1) -> dict:
    """Change one result at a time and rebuild all features; nothing at or before that match may move."""
    base = build_history_features(fixtures) if base is None else base
    rng = np.random.default_rng(seed)
    cutoff = fixtures["date"].quantile(0.85)                  # leave later matches that can react
    pool = fixtures.loc[fixtures["date"] < cutoff, "fixture_id"].to_numpy()
    out = {"trials": 0, "own_features_changed": 0, "earlier_or_same_day_features_changed": 0,
           "later_matches_of_involved_teams_unchanged": 0}
    for fid in rng.choice(pool, size=n_trials, replace=False):
        fx2 = fixtures.copy()
        i = fx2.index[fx2["fixture_id"] == fid][0]
        fx2.loc[i, "home_goals"] += 3                          # flips the outcome and the margin
        new = build_history_features(fx2)
        d_m, h, a = fx2.loc[i, "date"], fx2.loc[i, "home_team"], fx2.loc[i, "away_team"]
        out["trials"] += 1
        out["own_features_changed"] += int(not base.loc[[fid]].equals(new.loc[[fid]]))
        before = fixtures.loc[fixtures["date"] <= d_m, "fixture_id"]
        out["earlier_or_same_day_features_changed"] += int(not base.loc[before].equals(new.loc[before]))
        later = fixtures.loc[(fixtures["date"] > d_m) & fixtures["home_team"].isin([h, a]) | (fixtures["date"] > d_m) & fixtures["away_team"].isin([h, a]), "fixture_id"]
        out["later_matches_of_involved_teams_unchanged"] += int(base.loc[later].equals(new.loc[later]))   # must be 0
    return out


def check_no_target_columns(groups: dict[str, list[str]]) -> dict:
    offenders = {g: sorted(set(c) & TARGET_LIKE) for g, c in groups.items() if set(c) & TARGET_LIKE}
    return {"groups_checked": len(groups), "groups_with_target_like_columns": offenders}


def check_primary_group_separation(groups: dict[str, list[str]]) -> dict:
    """Ensure the recommended historical-only group excludes paper/lineup-sensitive inputs."""
    primary = set(groups.get("historical_only", []))
    unsafe_prefixes = ("pm_",)
    unsafe_exact = {
        "h_roster_rating", "a_roster_rating",
        "h_gk_rating", "h_def_rating", "h_mid_rating", "h_off_rating",
        "a_gk_rating", "a_def_rating", "a_mid_rating", "a_off_rating",
    }
    unsafe_diff = {
        "diff_roster_rating", "diff_gk_rating", "diff_def_rating",
        "diff_mid_rating", "diff_off_rating",
    }
    offenders = sorted(
        c for c in primary
        if c.startswith(unsafe_prefixes) or c in unsafe_exact or c in unsafe_diff
    )
    return {
        "historical_only_feature_count": len(primary),
        "paper_or_lineup_sensitive_features_in_primary": offenders,
        "historical_only_group_present": bool(primary),
        "passed": bool(primary) and not offenders,
    }


def check_prematch_ratings(eng: pd.DataFrame) -> dict:
    pm = [c for c in eng.columns if c.startswith("pm_") and "coverage" not in c]
    first = eng["season"] == "0910"
    cov = eng.loc[~first, ["pm_h_rating_coverage", "pm_a_rating_coverage"]].to_numpy().mean()
    return {"pm_columns": len(pm), "0910_rows_all_nan": bool(eng.loc[first, pm].isna().all().all()),
            "mean_starter_coverage_1011_onwards": round(float(cov), 3)}


def feature_outcome_correlations(eng: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    gd = (eng["home_goals"] - eng["away_goals"]).astype(float)
    rows = []
    for g, cols in groups.items():
        # Some paper-derived features (especially F3 player-level columns) live in
        # a separate paper feature matrix and are not columns in features_engineered.csv.
        # Skip them here rather than failing the entire leakage-check pipeline.
        for c in cols:
            if c not in eng.columns:
                continue
            x = eng[c]
            ok = x.notna()
            if ok.sum() > 100 and x[ok].nunique() > 1:
                rows.append({"group": g, "feature": c, "spearman_vs_goal_diff": x[ok].corr(gd[ok], method="spearman"),
                             "spearman_vs_home_win": x[ok].corr((eng["result"] == "H").astype(float)[ok], method="spearman")})
    df = pd.DataFrame(rows)
    df["abs_corr"] = df["spearman_vs_goal_diff"].abs()
    return df.sort_values("abs_corr", ascending=False).drop(columns="abs_corr").reset_index(drop=True)


# --------------------------------------------------------------------------
def run_all(n_perturb: int = 8) -> tuple[dict, bool]:
    fixtures = read_processed("fixtures_full.csv")
    eng = read_processed("features_engineered.csv")
    groups = json.loads((PROCESSED_DIR / "feature_groups.json").read_text())
    hist = build_history_features(fixtures)

    rep = {"1_chronology": check_chronology(fixtures),
           "2_brute_force": check_brute_force(fixtures, hist),
           "3_perturbation": check_perturbation(fixtures, hist, n_trials=n_perturb),
           "4_no_target_columns": check_no_target_columns(groups),
           "4b_primary_group_separation": check_primary_group_separation(groups),
           "5_prematch_ratings": check_prematch_ratings(eng)}
    c = rep["1_chronology"]; b = rep["2_brute_force"]; p = rep["3_perturbation"]
    ok = (c["sorted_by_date"] and c["team_twice_same_day"] == 0 and c["teams_not_38_games_per_season"] == 0
          and b["mismatches"] == 0
          and p["own_features_changed"] == 0 and p["earlier_or_same_day_features_changed"] == 0
          and p["later_matches_of_involved_teams_unchanged"] == 0
          and not rep["4_no_target_columns"]["groups_with_target_like_columns"]
          and rep["4b_primary_group_separation"]["passed"]
          and rep["5_prematch_ratings"]["0910_rows_all_nan"])
    corr = feature_outcome_correlations(eng, {k: v for k, v in groups.items() if k not in ("paper_F1",)})
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(TABLES_DIR / "feature_outcome_correlations.csv", index=False)
    rep["6_top_correlations_vs_goal_diff"] = corr.head(8).round(3).to_dict("records")
    rep["ALL_PASSED"] = bool(ok)
    (TABLES_DIR / "leakage_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep, bool(ok)


if __name__ == "__main__":
    report, passed = run_all()
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if passed else 1)
