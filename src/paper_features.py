"""
paper_features.py  --  Person 1, Phase 2.1 (reconstruct the paper's F1 / F2 / F3)

Verified from the original code (``scraping/combine_data.ipynb`` and the F1/F2/F3
notebooks in the paper repository) -- nothing below is guessed:

F1  = columns [0, 5] of ``match_vectors.csv``: h_roster_rating, a_roster_rating.
      roster_rating = plain mean of the player ratings of everyone in the 14-man
      match squad (11 starters + 3 substitutes) whose rating is non-zero.
F2  = columns 0..9 of ``match_vectors.csv`` = F1 + goalkeeper / defender /
      midfielder / attacker ratings for each team.  The positional ratings are
      *time-weighted* means:  sum(rating * w) / sum(w),  w = min(|t|, 1), where
      t is the fraction of THIS match the player played.  Squad members whose
      position is not GK / Defender / Midfielder count as attackers.
F3  = ``match_vectors_extended.csv`` (280 features + result).  Per team 20 slots x 7
      stats [rating, pass_success, passes_per_g, shots_per_g, key_passes_per_g,
      blocks, interceptions], each multiplied by w.  Slot layout:
      0 = GK, 1-6 = defenders, 7-12 = midfielders, 13-16 = forwards, 17-19 =
      substitutes.  Home block = columns 0-139, away block = 140-279.

Two properties of the ORIGINAL pipeline that matter for leakage (documented, and
reproduced here on purpose so the baseline matches the paper):
  (a) the ratings/stats come from the END-OF-SEASON roster file
      (``rosters_with_stats_<season>.csv``), i.e. they already contain the match
      being predicted and every later match of that season;
  (b) w uses the minutes the player actually played in the match (post-match
      information, e.g. it reveals substitutions and red cards).
Both are therefore NOT available before kick-off.  ``build_prematch_player_features``
provides a leakage-free alternative based on the PREVIOUS season's ratings.

Three quirks found in the original F3 code (reproduced by ``rebuild_f3`` so that it
matches the released file bit-for-bit; set ``replicate_quirks=False`` to see the
intended behaviour):
  * away defenders are written one column to the right (+141 instead of +140);
  * the 7 stats of every SUBSTITUTE slot are copied from ``player_index`` left over
    from the previous loop step (the away team's 11th starter), scaled by the
    substitute's own playing time -- they are not the substitute's own stats;
  * a missing substitute zeroes the next substitute slot, which wipes a 5th/6th
    forward that had spilled into slots 17/18 (affects 2 of 3,791 rows).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preprocessing import PAPER_DIR_NAME, RAW_DIR, ROSTER_TAG, SEASONS

F1_COLS = ["h_roster_rating", "a_roster_rating"]
F2_COLS = [
    "h_roster_rating", "h_gk_rating", "h_def_rating", "h_mid_rating", "h_off_rating",
    "a_roster_rating", "a_gk_rating", "a_def_rating", "a_mid_rating", "a_off_rating",
]
ROSTER_STAT_COLS = ["rating", "PS%", "AvgP", "SpG", "KeyP", "Blocks", "Inter"]
POS_KEY = {"Goalkeeper": "gk", "Defender": "def", "Midfielder": "mid"}  # everything else -> attacker


# --------------------------------------------------------------------------
# Released (authoritative) vectors
# --------------------------------------------------------------------------
def load_released_vectors(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The paper's own ``match_vectors.csv`` (F2 + label) and ``match_vectors_extended.csv`` (F3 + result)."""
    repo = Path(raw_dir) / PAPER_DIR_NAME
    mv = pd.read_csv(repo / "match_vectors.csv", skipinitialspace=True)
    ex = pd.read_csv(repo / "match_vectors_extended.csv", skipinitialspace=True)
    return mv, ex


def load_paper_feature_sets(matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Return F1, F2, F3 aligned row-for-row with ``matches`` (matches_clean.csv order).

    Rows are selected through ``matches['paper_row']``, so de-duplicated / re-ordered
    matches still get the correct vector.  Features only -- no labels.
    """
    mv, ex = load_released_vectors(raw_dir)
    rows = matches["paper_row"].to_numpy()
    f2 = mv.drop(columns="label").iloc[rows].reset_index(drop=True)
    f1 = f2[F1_COLS].copy()
    f3 = ex.drop(columns="result").iloc[rows].reset_index(drop=True)
    return {"F1": f1, "F2": f2, "F3": f3}


# --------------------------------------------------------------------------
# Rebuild from raw lineups + rosters (proves we understand the original code)
# --------------------------------------------------------------------------
def _load_season(raw_dir: Path, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo = Path(raw_dir) / PAPER_DIR_NAME
    lineup = pd.read_csv(repo / "lineups" / f"match_lineup_{season}.csv", skipinitialspace=True, encoding_errors="replace")
    roster = pd.read_csv(repo / "rosters" / f"rosters_with_stats_{ROSTER_TAG[season]}.csv", skipinitialspace=True, encoding_errors="replace")
    return lineup, roster


def _stat_lookup(roster: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    """(team-id, player-id) -> the 7 stats, FIRST matching row (as in the original code)."""
    first = roster.drop_duplicates(["team-id", "player-id"], keep="first")
    vals = first[ROSTER_STAT_COLS].to_numpy(dtype=float)
    return {(int(t), int(p)): v for t, p, v in zip(first["team-id"], first["player-id"], vals)}


def rebuild_f2(raw_dir: Path = RAW_DIR) -> np.ndarray:
    """Recompute the 10 F2 columns for all rows (paper_row order) from raw files."""
    out = []
    for season in SEASONS:
        lineup, roster = _load_season(raw_dir, season)
        look = _stat_lookup(roster)
        for row in lineup.to_dict("records"):
            vec = []
            for side in ("h", "a"):
                team_id = int(row[f"{side}_id"])
                on_field, num, den = [], dict.fromkeys(("gk", "def", "mid", "off"), 0.0), dict.fromkeys(("gk", "def", "mid", "off"), 0.0)
                for j in range(1, 15):
                    pid = row[f"{side}_p{j}_id"]
                    if pd.isna(pid) or pid == 0:
                        continue
                    rating = look[(team_id, int(pid))][0]
                    w = min(abs(row[f"{side}_p{j}_t"]), 1.0)
                    key = POS_KEY.get(row[f"{side}_p{j}_pos"], "off")
                    on_field.append(rating)
                    num[key] += rating * w
                    den[key] += w
                nz = [r for r in on_field if r != 0]
                vec += [np.mean(nz)] + [num[k] / den[k] if den[k] else np.nan for k in ("gk", "def", "mid", "off")]
            out.append(vec)
    return np.asarray(out, dtype=float)


def rebuild_f3(raw_dir: Path = RAW_DIR, replicate_quirks: bool = True) -> np.ndarray:
    """Recompute the 280 F3 columns for all rows (paper_row order) from raw files."""
    out = []
    for season in SEASONS:
        lineup, roster = _load_season(raw_dir, season)
        look = _stat_lookup(roster)
        player_stats = None  # mirrors the original's leaked loop variable ``player_index``
        for row in lineup.to_dict("records"):
            vec = np.zeros(280)
            cnt = {"h": dict(d=0, m=0, f=0, s=0), "a": dict(d=0, m=0, f=0, s=0)}
            for j in range(1, 15):
                for side, off in (("h", 0), ("a", 140)):
                    c = cnt[side]
                    pid = row[f"{side}_p{j}_id"]
                    if pd.isna(pid) or pid == 0:           # unregistered player: sub counter moves on
                        if replicate_quirks:
                            # original zeroes the next substitute slot: only its first cell for the
                            # home side, the whole 7-cell slot for the away side.  This wipes a 5th/6th
                            # forward that spilled into slots 17/18 (seen in 2 of 3,791 rows).
                            lo = off + (17 + c["s"]) * 7
                            vec[lo:lo + (1 if side == "h" else 7)] = 0
                        c["s"] += 1
                        continue
                    w = min(abs(row[f"{side}_p{j}_t"]), 1.0)
                    if j <= 11:                              # starter
                        player_stats = look[(int(row[f"{side}_id"]), int(pid))]
                        pos = row[f"{side}_p{j}_pos"]
                        if pos == "Goalkeeper":
                            base = off
                        elif pos == "Defender":
                            base = off + (c["d"] + 1) * 7 + (1 if (side == "a" and replicate_quirks) else 0)
                            c["d"] += 1
                        elif pos == "Midfielder":
                            base = off + (c["m"] + 7) * 7
                            c["m"] += 1
                        elif pos == "Forward":
                            base = off + (c["f"] + 13) * 7
                            c["f"] += 1
                        else:
                            continue
                        vec[base:base + 7] = player_stats * w
                    else:                                    # substitute
                        if not replicate_quirks:
                            player_stats = look[(int(row[f"{side}_id"]), int(pid))]
                        base = off + (c["s"] + 17) * 7
                        vec[base:base + 7] = player_stats * w
                        c["s"] += 1
            out.append(vec)
    return np.asarray(out, dtype=float)


def verify_paper_features(raw_dir: Path = RAW_DIR, rtol: float = 1e-9) -> dict:
    """Compare the rebuilt F2 / F3 with the paper's released vectors."""
    mv, ex = load_released_vectors(raw_dir)
    res = {}
    for name, rebuilt, released in (("F2", rebuild_f2(raw_dir), mv[F2_COLS].to_numpy(float)),
                                    ("F3", rebuild_f3(raw_dir), ex.drop(columns="result").to_numpy(float))):
        same = np.isclose(rebuilt, released, rtol=rtol, atol=1e-9, equal_nan=True)
        res[name] = {"shape": rebuilt.shape, "all_cells_match": bool(same.all()),
                     "cells_mismatching": int((~same).sum()),
                     "rows_with_any_mismatch": int((~same).any(axis=1).sum()),
                     "max_abs_diff": float(np.nanmax(np.abs(rebuilt - released)))}
    return res


# --------------------------------------------------------------------------
# Leakage-free player-rating alternative (uses PREVIOUS season ratings only)
# --------------------------------------------------------------------------
def build_prematch_player_features(matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Pre-match team ratings from the *previous* season's player ratings.

    Available before kick-off: the announced starting XI (positions included) and
    last season's season-long player ratings.  Not used: this season's ratings and
    minutes played.  Players with no rating last season (new signings, promoted-club
    players, 2009/10 when there is no previous season) are skipped; the share of the
    XI that has a rating is stored in ``*_coverage`` so models/analysis can see it.
    Team values are NaN when no starter has a rating -- impute INSIDE the training fold.
    """
    cols = []
    rows = []
    for k, season in enumerate(SEASONS):
        lineup, _ = _load_season(RAW_DIR if raw_dir is None else raw_dir, season)
        prior = None
        if k > 0:
            _, prior_roster = _load_season(raw_dir, SEASONS[k - 1])
            prior = prior_roster.groupby("player-id")["rating"].mean()   # a player on 2 clubs -> mean of rows
        for pr, row in enumerate(lineup.to_dict("records")):
            rec = {"season": season, "_i": pr}
            for side in ("h", "a"):
                by_pos = {"gk": [], "def": [], "mid": [], "off": []}
                n_start, n_rated = 0, 0
                for j in range(1, 12):                                   # starting XI only
                    pid = row[f"{side}_p{j}_id"]
                    if pd.isna(pid) or pid == 0:
                        continue
                    n_start += 1
                    r = prior.get(int(pid), np.nan) if prior is not None else np.nan
                    if np.isnan(r):
                        continue
                    n_rated += 1
                    by_pos[POS_KEY.get(row[f"{side}_p{j}_pos"], "off")].append(r)
                allr = [x for v in by_pos.values() for x in v]
                rec[f"pm_{side}_roster_rating"] = np.mean(allr) if allr else np.nan
                for key in ("gk", "def", "mid", "off"):
                    rec[f"pm_{side}_{key}_rating"] = np.mean(by_pos[key]) if by_pos[key] else np.nan
                rec[f"pm_{side}_rating_coverage"] = n_rated / n_start if n_start else np.nan
            rows.append(rec)
    pm = pd.DataFrame(rows)
    # attach by paper_row (rows were produced in paper order)
    pm["paper_row"] = np.arange(len(pm))
    pm = pm.drop(columns=["season", "_i"])
    keep = [c for c in pm.columns if c != "paper_row"]
    return matches[["paper_row"]].merge(pm, on="paper_row", how="left")[keep].reset_index(drop=True)
