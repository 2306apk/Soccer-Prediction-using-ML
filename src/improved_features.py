"""
improved_features.py  --  Person 1, Phase 3 (engineered, leakage-free features)

Everything here is computed from the COMPLETE fixture list (``fixtures_full.csv``,
3,800 matches) in strict chronological order:

    for each match day (ascending):
        1. compute features of every fixture on that day from the state BEFORE the day
        2. only then update the state (histories, Elo) with that day's results

so a feature can never see its own match (or any later one).  A team plays at most
once per day, so same-day matches are independent of each other.

Feature groups
--------------
form (windows 3 / 5 / 10, any venue)   h_/a_ points, wins, draws, losses, gf_avg, ga_avg,
                                       gd_avg, ppg, trend
venue form (last 10 at that venue)      h_home_* for the home team's previous HOME games,
                                       a_away_* for the away team's previous AWAY games
elo                                     h_elo, a_elo, elo_diff, elo_exp_home, abs_elo_diff
historical differences                 form / venue differences only in the historical-only groups
paper rating differences                retained in the table for paper reproduction / analysis only
n_prev_h / n_prev_a                     number of earlier matches the team has in the data

Conventions (all documented, none tuned on test data)
-----------------------------------------------------
* A window with fewer than ``w`` earlier matches (start of the data, newly promoted
  team) scales counts/points by w/n so they stay on the "last w matches" scale; with
  0 earlier matches the value is NaN -> impute INSIDE the training fold.
* Elo: K=20, home advantage 65 points, goal-margin multiplier (World-Football-Elo
  style), start 1500.  At each new season ratings regress 1/3 of the way to 1500; clubs
  new to the league (promoted / returning) start at the (regressed) mean Elo of the clubs
  relegated the season before, and their match history is reset.  League membership of a
  season is public before its first match, so this uses no future results.
* 2009/10 is a warm-up for Elo/form (no earlier data) -- keep that in mind when
  choosing evaluation seasons.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import PROCESSED_DIR, RAW_DIR, read_processed
from src.paper_features import F1_COLS, F2_COLS, build_prematch_player_features, load_paper_feature_sets

WINDOWS = (3, 5, 10)
VENUE_WINDOW = 10
FORM_STATS = ("points", "wins", "draws", "losses", "gf_avg", "ga_avg", "gd_avg", "ppg", "trend")
VENUE_STATS = ("ppg", "win_rate", "draw_rate", "gf_avg", "ga_avg")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _form_features(hist: list, prefix: str, w: int) -> dict:
    """Features of one team from its last ``w`` matches; hist items are (points, gf, ga)."""
    keys = [f"{prefix}_{s}_last{w}" for s in FORM_STATS]
    last = hist[-w:]
    n = len(last)
    if n == 0:
        return dict.fromkeys(keys, np.nan)
    pts = np.array([p for p, _, _ in last], float)
    gf = np.array([g for _, g, _ in last], float)
    ga = np.array([g for _, _, g in last], float)
    scale = w / n                                   # keeps partial windows on the "w matches" scale
    half = w // 2
    trend = pts[-half:].mean() - pts[:-half].mean() if (n == w and 0 < half < n) else np.nan
    vals = [pts.sum() * scale, (pts == 3).sum() * scale, (pts == 1).sum() * scale, (pts == 0).sum() * scale,
            gf.mean(), ga.mean(), (gf - ga).mean(), pts.mean(), trend]
    return dict(zip(keys, vals))


def _venue_features(hist: list, prefix: str, v: int) -> dict:
    keys = [f"{prefix}_{s}_last{v}" for s in VENUE_STATS]
    last = hist[-v:]
    if not last:
        return dict.fromkeys(keys, np.nan)
    pts = np.array([p for p, _, _ in last], float)
    gf = np.array([g for _, g, _ in last], float)
    ga = np.array([g for _, _, g in last], float)
    return dict(zip(keys, [pts.mean(), (pts == 3).mean(), (pts == 1).mean(), gf.mean(), ga.mean()]))


def _elo_expected(elo_home: float, elo_away: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(elo_home + home_adv - elo_away) / 400.0))


def _elo_margin_multiplier(gd: int) -> float:
    return 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)


# --------------------------------------------------------------------------
# main chronological pass
# --------------------------------------------------------------------------
def build_history_features(
    fixtures: pd.DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    venue_window: int = VENUE_WINDOW,
    elo_k: float = 20.0,
    elo_home_adv: float = 65.0,
    elo_init: float = 1500.0,
    season_regression: float = 1.0 / 3.0,
) -> pd.DataFrame:
    """Pre-match form / venue-form / Elo features for EVERY fixture, indexed by ``fixture_id``.

    ``fixtures`` needs: fixture_id, season, date, home_team, away_team, home_goals, away_goals.
    """
    fx = fixtures.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    season_teams = {s: set(g["home_team"]) | set(g["away_team"]) for s, g in fx.groupby("season")}
    season_order = list(dict.fromkeys(fx["season"]))            # chronological order of seasons

    hist_all, hist_home, hist_away = defaultdict(list), defaultdict(list), defaultdict(list)
    elo: dict[str, float] = {}
    prev_season = None
    rows = []

    for date, day in fx.groupby("date", sort=True):
        season = day["season"].iloc[0]
        if season != prev_season:                                # ---- new season: league membership is known
            cur = season_teams[season]
            if prev_season is None:
                for t in cur:
                    elo[t] = elo_init
            else:
                prev = season_teams[prev_season]
                relegated = prev - cur
                promo = (1 - season_regression) * np.mean([elo[t] for t in relegated]) + season_regression * elo_init \
                    if relegated else elo_init
                new_elo = {t: (1 - season_regression) * elo[t] + season_regression * elo_init for t in cur & prev}
                for t in cur - prev:                             # promoted / returning clubs: fresh start
                    new_elo[t] = promo
                    hist_all[t].clear(); hist_home[t].clear(); hist_away[t].clear()
                elo.update(new_elo)
            prev_season = season

        updates = []
        for r in day.itertuples(index=False):                    # ---- 1) features from the state BEFORE the day
            h, a = r.home_team, r.away_team
            f = {"fixture_id": r.fixture_id, "n_prev_h": len(hist_all[h]), "n_prev_a": len(hist_all[a])}
            for w in windows:
                f.update(_form_features(hist_all[h], "h", w))
                f.update(_form_features(hist_all[a], "a", w))
            f.update(_venue_features(hist_home[h], "h_home", venue_window))
            f.update(_venue_features(hist_away[a], "a_away", venue_window))
            eh, ea = elo[h], elo[a]
            f.update(h_elo=eh, a_elo=ea, elo_diff=eh - ea,
                     elo_exp_home=_elo_expected(eh, ea, elo_home_adv), abs_elo_diff=abs(eh - ea))
            rows.append(f)
            updates.append(r)
        for r in updates:                                        # ---- 2) only now use the results
            h, a, hg, ag = r.home_team, r.away_team, int(r.home_goals), int(r.away_goals)
            ph, pa = (3, 0) if hg > ag else ((1, 1) if hg == ag else (0, 3))
            hist_all[h].append((ph, hg, ag)); hist_all[a].append((pa, ag, hg))
            hist_home[h].append((ph, hg, ag)); hist_away[a].append((pa, ag, hg))
            exp_h = _elo_expected(elo[h], elo[a], elo_home_adv)
            score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = elo_k * _elo_margin_multiplier(abs(hg - ag)) * (score_h - exp_h)
            elo[h] += delta
            elo[a] -= delta

    return pd.DataFrame(rows).set_index("fixture_id").sort_index()


# --------------------------------------------------------------------------
# relative-strength (difference) features
# --------------------------------------------------------------------------
def add_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add diff_X = h_X - a_X for every pair of matching home/away columns.

    Also: venue pairs (h_home_X - a_away_X) and pre-match rating pairs (pm_h_X - pm_a_X).
    ``elo_diff`` already exists and is not duplicated.
    """
    out, new = df.copy(), {}
    cols = set(df.columns)
    for c in df.columns:
        if c.startswith("h_home_"):
            a = "a_away_" + c[len("h_home_"):]
            if a in cols:
                new["diff_venue_" + c[len("h_home_"):]] = df[c] - df[a]
        elif c.startswith("h_") and c != "h_elo" and ("a_" + c[2:]) in cols:
            new["diff_" + c[2:]] = df[c] - df["a_" + c[2:]]
        elif c.startswith("pm_h_") and "coverage" not in c and ("pm_a_" + c[5:]) in cols:
            new["pm_diff_" + c[5:]] = df[c] - df["pm_a_" + c[5:]]
    return pd.concat([out, pd.DataFrame(new, index=df.index)], axis=1)


def feature_groups(
    columns: list[str],
    paper_f3_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    """Return explicitly separated feature groups for reproduction and valid prediction.

    ``historical_only`` is the recommended primary feature group. It contains only
    pre-match features derived from matches completed before the target fixture.
    Paper F1/F2/F3 are retained solely for reproducing the original paper and are
    not included in any historical-only group.

    ``prematch_ratings_sensitivity`` is intentionally separate: although ratings
    come from the previous season, the source's recorded starting XI may be
    retrospective. Use this group only if lineup availability before kick-off is
    independently verified.
    """
    cols = list(columns)
    available = set(cols)
    groups: dict[str, list[str]] = {
        "paper_F1": [c for c in F1_COLS if c in available],
        "paper_F2": [c for c in F2_COLS if c in available],
    }

    for w in WINDOWS:
        groups[f"recent_form_w{w}"] = [
            c for c in cols
            if re.match(rf"^[ha]_.*_last{w}$", c)
            and "_home_" not in c and "_away_" not in c
        ]
        groups[f"historical_differences_w{w}"] = [
            c for c in cols
            if re.match(rf"^diff_.*_last{w}$", c)
            and not c.startswith("diff_venue_")
        ]

    groups["elo"] = [c for c in ("h_elo", "a_elo", "elo_diff") if c in available]
    groups["elo_extra"] = [c for c in ("elo_exp_home", "abs_elo_diff") if c in available]
    groups["venue_form"] = [c for c in cols if c.startswith(("h_home_", "a_away_"))]
    groups["historical_venue_differences"] = [
        c for c in cols if c.startswith("diff_venue_")
    ]
    groups["history_counts"] = [
        c for c in ("n_prev_h", "n_prev_a") if c in available
    ]

    # Explicitly historical differences only: never include differences computed
    # from the paper's end-of-season F1/F2 values.
    historical_diff_w5 = groups.get("historical_differences_w5", [])
    groups["historical_differences"] = (
        historical_diff_w5 + groups["historical_venue_differences"]
    )

    # Core proposed feature set: all inputs are based on results before the match.
    groups["historical_only"] = list(dict.fromkeys(
        groups.get("recent_form_w5", [])
        + groups["elo"]
        + groups["elo_extra"]
        + groups["venue_form"]
        + groups["historical_differences"]
    ))
    groups["historical_only_extended"] = list(dict.fromkeys(
        groups.get("recent_form_w3", [])
        + groups.get("recent_form_w5", [])
        + groups.get("recent_form_w10", [])
        + groups["elo"]
        + groups["elo_extra"]
        + groups["venue_form"]
        + groups.get("historical_differences_w3", [])
        + groups.get("historical_differences_w5", [])
        + groups.get("historical_differences_w10", [])
        + groups["historical_venue_differences"]
    ))

    # Previous-season player ratings are not automatically safe: the recorded
    # lineup could be retrospective rather than a timestamped pre-match lineup.
    groups["prematch_ratings_sensitivity"] = [
        c for c in cols if c.startswith("pm_")
    ]
    groups["historical_plus_ratings_sensitivity"] = list(dict.fromkeys(
        groups["historical_only"] + groups["prematch_ratings_sensitivity"]
    ))

    # F3 lives in paper_f3.csv, not features_engineered.csv. Pass its columns in
    # from build_engineered_dataset so downstream code knows it is a separate table.
    if paper_f3_columns is not None:
        groups["paper_F3"] = [
            c for c in paper_f3_columns
            if c not in {"paper_row", "fixture_id", "date", "label", "result"}
        ]

    # Do not expose empty optional groups in metadata generated from a small fixture
    # or a synthetic test table. Required groups remain visible when populated.
    return {k: list(dict.fromkeys(c for c in v if c in available or
                                  (k == "paper_F3" and paper_f3_columns is not None)))
            for k, v in groups.items() if v}


# --------------------------------------------------------------------------
# build + save the engineered dataset
# --------------------------------------------------------------------------
ID_COLS = ["paper_row", "match_id", "fixture_id", "season", "date", "home_team", "away_team", "result", "label"]


def build_engineered_dataset(processed_dir: Path = PROCESSED_DIR, raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (engineered features, paper F3, feature_groups) aligned with matches_clean."""
    fixtures = read_processed("fixtures_full.csv", processed_dir)
    matches = read_processed("matches_clean.csv", processed_dir)

    hist = build_history_features(fixtures)
    paper = load_paper_feature_sets(matches, raw_dir)
    pm = build_prematch_player_features(matches, raw_dir)

    base = matches[ID_COLS].copy()
    base["home_goals"] = matches["home_goals"]      # kept ONLY for analysis/targets; never a model input
    base["away_goals"] = matches["away_goals"]
    eng = base.merge(hist.reset_index(), on="fixture_id", how="left", validate="one_to_one")
    eng = pd.concat([eng, paper["F2"], pm], axis=1)
    eng = add_difference_features(eng)
    eng = eng.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    f3 = pd.concat([matches[["paper_row", "fixture_id", "date", "label"]].reset_index(drop=True), paper["F3"]], axis=1)
    f3 = f3.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    return eng, f3, feature_groups(list(eng.columns), list(f3.columns))


def main() -> None:
    eng, f3, groups = build_engineered_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    eng.to_csv(PROCESSED_DIR / "features_engineered.csv", index=False)
    f3.to_csv(PROCESSED_DIR / "paper_f3.csv", index=False)
    (PROCESSED_DIR / "feature_groups.json").write_text(json.dumps(groups, indent=2))
    print("features_engineered:", eng.shape, "| paper_f3:", f3.shape)
    for k, v in groups.items():
        print(f"  {k:<18} {len(v):>3} columns")


if __name__ == "__main__":
    main()
