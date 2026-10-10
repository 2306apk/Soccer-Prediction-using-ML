"""
data_preprocessing.py  --  Person 1, Phase 1 (dataset acquisition + cleaning)

What this module does
---------------------
1. Downloads the two raw sources (see ``fetch_raw_data``):
     * the paper's repository (player ratings, line-ups, final scores), and
     * the openfootball/england fixture lists, which supply the MATCH DATES.
   The paper's data has NO dates, and the row order of its files is not
   chronological after 2013/14, so dates are required for any leakage-free
   rolling feature (form, Elo).
2. Cleans and merges them into two tables:
     * ``fixtures_full.csv``  - all 3,800 EPL fixtures 2009/10-2018/19 with
       date, teams, goals and result.  Used ONLY to compute historical form /
       Elo (a team's earlier results are public pre-match information).
     * ``matches_clean.csv``  - the 3,791 matches that exist in the paper's
       dataset, with ``paper_row`` pointing back to the row of the paper's
       ``match_vectors.csv`` / ``match_vectors_extended.csv``.
3. Builds the H / D / A target from the final score.
4. Produces a data-quality report (missing values, duplicates, team-name
   consistency, invalid dates, score agreement between the two sources).

Run from the repository root:
    python -m src.data_preprocessing --fetch --build
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "results" / "tables"

PAPER_REPO_URL = "https://github.com/varunharbola/EPL_match_prediction"
PAPER_DIR_NAME = "paper_repo"
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/england/master/{season}/1-premierleague.txt"
)

# paper's short season code -> openfootball folder -> paper's roster file tag
SEASONS = ["0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819"]
SEASON_LONG = {s: f"20{s[:2]}-{s[2:]}" for s in SEASONS}                # '0910' -> '2009-10'
ROSTER_TAG = {s: f"20{s[:2]}-20{s[2:]}" for s in SEASONS}               # '0910' -> '2009-2010'
SEASON_START_YEAR = {s: 2000 + int(s[:2]) for s in SEASONS}

# openfootball name -> name used in the paper's (WhoScored) files.
# Both sources contain exactly the same 36 clubs, so this is a 1:1 map.
TEAM_MAP = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa": "Aston Villa",
    "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn", "Blackpool FC": "Blackpool",
    "Bolton Wanderers": "Bolton", "Brighton & Hove Albion": "Brighton", "Burnley FC": "Burnley",
    "Cardiff City": "Cardiff", "Chelsea FC": "Chelsea", "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull", "Leicester City": "Leicester", "Liverpool FC": "Liverpool",
    "Manchester City": "Man City", "Manchester United": "Man Utd", "Middlesbrough FC": "Middlesbrough",
    "Newcastle United": "Newcastle", "Norwich City": "Norwich", "Portsmouth FC": "Portsmouth",
    "Queens Park Rangers": "QPR", "Reading FC": "Reading", "Southampton FC": "Southampton",
    "Stoke City": "Stoke", "Sunderland AFC": "Sunderland", "Swansea City": "Swansea",
    "Tottenham Hotspur": "Tottenham", "Watford FC": "Watford", "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham", "Wigan Athletic": "Wigan", "Wolverhampton Wanderers": "Wolves",
}

# paper label convention is -1 / 0 / 1 (loss / draw / win for the home side)
LABEL_TO_RESULT = {-1: "A", 0: "D", 1: "H"}
RESULT_TO_INT = {"A": -1, "D": 0, "H": 1}

_DATE_RE = re.compile(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})\b")
_MATCH_RE = re.compile(r"^\s*(?:\d{1,2}[:.]\d{2}\s+)?(.+?)\s+(\d+)-(\d+)(?:\s+\([^)]*\))?\s+(.+?)\s*$")


# --------------------------------------------------------------------------
# 1. Acquisition
# --------------------------------------------------------------------------
def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` (urllib first, curl as a fallback)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:
        subprocess.run(["curl", "-fsSL", "-m", "60", "-o", str(dest), url], check=True)


def fetch_raw_data(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    """Clone the paper repository and download the openfootball fixture lists."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paper = raw_dir / PAPER_DIR_NAME
    if force and paper.exists():
        shutil.rmtree(paper)
    if not paper.exists():
        print(f"Cloning {PAPER_REPO_URL} ...")
        subprocess.run(["git", "clone", "--depth", "1", PAPER_REPO_URL, str(paper)], check=True)
    for s in SEASONS:
        dest = raw_dir / "openfootball" / f"pl_{SEASON_LONG[s]}.txt"
        if force or not dest.exists():
            print(f"Downloading fixtures {SEASON_LONG[s]} ...")
            _download(OPENFOOTBALL_URL.format(season=SEASON_LONG[s]), dest)


# --------------------------------------------------------------------------
# 2. Loading
# --------------------------------------------------------------------------
def parse_openfootball(path: Path, season: str) -> pd.DataFrame:
    """Parse one openfootball season file into a tidy DataFrame.

    The file lists matches under date headers such as ``Sat Aug 16`` (no
    year), so the year is inferred from the season: Aug-Dec belong to the
    start year, Jan-Jul to the following year.
    """
    start = SEASON_START_YEAR[season]
    rows, current_date = [], None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("=", "#", "▪")):
            continue
        m = _DATE_RE.match(s)
        if m:
            mon, day = m.group(1), int(m.group(2))
            month_num = datetime.strptime(mon, "%b").month
            year = start if month_num >= 8 else start + 1
            current_date = pd.Timestamp(year=year, month=month_num, day=day)
            continue
        m = _MATCH_RE.match(line.rstrip("\n"))
        if m and current_date is not None:
            rows.append(
                dict(
                    season=season,
                    date=current_date,
                    home_team_of=m.group(1).strip(),
                    away_team_of=m.group(4).strip(),
                    home_goals=int(m.group(2)),
                    away_goals=int(m.group(3)),
                )
            )
    df = pd.DataFrame(rows)
    df["home_team"] = df["home_team_of"].map(TEAM_MAP)
    df["away_team"] = df["away_team_of"].map(TEAM_MAP)
    return df


def load_fixtures_full(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """All fixtures from openfootball (3,800 expected), with dates."""
    frames = [parse_openfootball(raw_dir / "openfootball" / f"pl_{SEASON_LONG[s]}.txt", s) for s in SEASONS]
    return pd.concat(frames, ignore_index=True)


def load_paper_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Paper results + line-up team names, in the paper's own row order.

    ``paper_row`` is the 0-based row index into ``match_vectors.csv`` and
    ``match_vectors_extended.csv`` (the paper built those files by looping over
    the seasons and, inside each season, over the rows of the results file).
    """
    repo = raw_dir / PAPER_DIR_NAME
    frames = []
    for s in SEASONS:
        res = pd.read_csv(repo / "match result" / f"match_results-{s}.csv", skipinitialspace=True)
        lu = pd.read_csv(
            repo / "lineups" / f"match_lineup_{s}.csv",
            skipinitialspace=True,
            usecols=["match_id", "h_id", "h_name", "a_id", "a_name"],
            encoding_errors="replace",
        )
        if len(res) != len(lu) or not (res["match_id"].values == lu["match_id"].values).all():
            raise ValueError(f"results / lineups are not row-aligned for season {s}")
        # rows are aligned (checked above); concat by position because a match_id
        # can repeat inside a season file, which would break a key-based merge
        df = pd.concat([res.reset_index(drop=True), lu.drop(columns="match_id").reset_index(drop=True)], axis=1)
        df["season"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "paper_row", np.arange(len(out)))
    return out.rename(columns={"h_name": "home_team", "a_name": "away_team", "h_id": "home_team_id", "a_id": "away_team_id"})


# --------------------------------------------------------------------------
# 3. Target + merge
# --------------------------------------------------------------------------
def make_result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """H / D / A from the FINAL score. (Goals are never used as model inputs.)"""
    return pd.Series(
        np.where(home_goals > away_goals, "H", np.where(home_goals == away_goals, "D", "A")),
        index=home_goals.index,
    )


def build_clean_datasets(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (fixtures_full, matches_clean, quality_report_dict)."""
    fixtures = load_fixtures_full(raw_dir)
    paper = load_paper_matches(raw_dir)
    report: dict = {}

    # ---- fixtures_full -----------------------------------------------------
    fixtures["result"] = make_result(fixtures["home_goals"], fixtures["away_goals"])
    fixtures = fixtures.sort_values(["date", "home_team"]).reset_index(drop=True)
    fixtures["fixture_id"] = np.arange(len(fixtures))
    report["fixtures_total"] = len(fixtures)
    report["fixtures_unmapped_team_names"] = int(fixtures[["home_team", "away_team"]].isna().sum().sum())
    report["fixtures_duplicate_pairs"] = int(fixtures.duplicated(["season", "home_team", "away_team"]).sum())
    report["fixtures_missing_values"] = int(fixtures[["date", "home_goals", "away_goals"]].isna().sum().sum())
    # a team can play at most once per calendar day
    long = pd.concat([fixtures[["date", "home_team"]].rename(columns={"home_team": "team"}),
                      fixtures[["date", "away_team"]].rename(columns={"away_team": "team"})])
    report["fixtures_team_twice_same_day"] = int(long.duplicated(["date", "team"]).sum())
    # each club plays 38 games per season
    per_team = pd.concat([fixtures[["season", "home_team"]].rename(columns={"home_team": "team"}),
                          fixtures[["season", "away_team"]].rename(columns={"away_team": "team"})]).groupby(["season", "team"]).size()
    report["fixtures_teams_not_38_games"] = int((per_team != 38).sum())
    # dates inside a plausible season window (Aug of start year .. Jun of next)
    lo = fixtures["season"].map(lambda s: pd.Timestamp(SEASON_START_YEAR[s], 7, 25))
    hi = fixtures["season"].map(lambda s: pd.Timestamp(SEASON_START_YEAR[s] + 1, 6, 1))
    report["fixtures_dates_outside_season_window"] = int(((fixtures["date"] < lo) | (fixtures["date"] > hi)).sum())

    # ---- paper side --------------------------------------------------------
    report["paper_rows"] = len(paper)
    report["paper_missing_values"] = int(paper[["home_goals", "away_goals", "home_team", "away_team"]].isna().sum().sum())
    report["paper_duplicate_match_id_rows"] = int(paper["match_id"].duplicated().sum())
    report["paper_distinct_team_names"] = int(pd.unique(paper[["home_team", "away_team"]].values.ravel()).size)
    report["paper_team_names_not_in_map"] = sorted(set(pd.unique(paper[["home_team", "away_team"]].values.ravel())) - set(TEAM_MAP.values()))
    paper["result_from_paper_score"] = make_result(paper["home_goals"], paper["away_goals"])

    # ---- merge on (season, home, away): unique per season ------------------
    merged = paper.merge(
        fixtures[["fixture_id", "season", "home_team", "away_team", "date", "home_goals", "away_goals"]],
        on=["season", "home_team", "away_team"], how="left", suffixes=("", "_fixture"), validate="many_to_one",
    )
    report["paper_rows_without_date"] = int(merged["date"].isna().sum())
    report["paper_duplicate_fixture_rows"] = int(merged.duplicated(["season", "home_team", "away_team"]).sum())
    both = merged.dropna(subset=["date"])
    mism = (both["home_goals"] != both["home_goals_fixture"]) | (both["away_goals"] != both["away_goals_fixture"])
    report["score_mismatches_between_sources"] = int(mism.sum())
    report["score_mismatch_rows"] = both.loc[mism, ["paper_row", "season", "home_team", "away_team",
                                                    "home_goals", "away_goals", "home_goals_fixture",
                                                    "away_goals_fixture"]].to_dict("records")

    matches = merged.dropna(subset=["date"]).copy()
    # drop a paper row whose scoreline disagrees with the fixture list (kept in the report)
    matches = matches.loc[~mism.reindex(matches.index, fill_value=False)].copy()
    # The paper's data contains one fixture twice (Man City v Chelsea, 2010/11, identical
    # feature vectors).  In a random 80/20 split the copies can land on both sides, so the
    # duplicate is removed here; its paper_row is recorded so the paper-faithful baseline
    # (which uses the raw vector files) can still be reproduced exactly.
    dup = matches.duplicated(["season", "home_team", "away_team"], keep="first")
    report["duplicate_fixture_paper_rows_dropped"] = matches.loc[dup, "paper_row"].tolist()
    matches = matches.loc[~dup].copy()
    matches["result"] = matches["result_from_paper_score"]
    matches["label"] = matches["result"].map(RESULT_TO_INT)
    matches = matches.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    matches = matches[["paper_row", "match_id", "fixture_id", "season", "date", "home_team", "away_team",
                       "home_team_id", "away_team_id", "home_goals", "away_goals", "result", "label"]]
    report["matches_clean"] = len(matches)
    report["fixtures_not_in_paper_dataset"] = int(len(fixtures) - matches["fixture_id"].nunique())
    report["class_counts"] = matches["result"].value_counts().to_dict()
    report["class_share"] = (matches["result"].value_counts(normalize=True).round(4)).to_dict()
    return fixtures, matches, report


def verify_alignment_with_paper_vectors(matches: pd.DataFrame, raw_dir: Path = RAW_DIR) -> dict:
    """Sanity check: label of each row of the paper's ``match_vectors.csv``
    (-1/0/1) must equal the label we derive for the same ``paper_row``."""
    mv = pd.read_csv(raw_dir / PAPER_DIR_NAME / "match_vectors.csv", skipinitialspace=True)
    ours = matches.set_index("paper_row")["label"].reindex(range(len(mv)))
    present = ours.notna()
    ok = present & (ours.values == mv["label"].values)
    return {"vector_rows": len(mv), "rows_kept_in_matches_clean": int(present.sum()),
            "kept_rows_with_matching_label": int(ok.sum()),
            "kept_rows_with_label_mismatch": int((present & ~ok).sum()),
            "rows_removed_as_duplicates": int((~present).sum())}


def read_processed(name: str, processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read a processed CSV with the right dtypes.

    ``season`` MUST be read as text: pandas would turn '0910' into the integer 910,
    silently losing the leading zero and breaking season comparisons.
    """
    return pd.read_csv(Path(processed_dir) / name, dtype={"season": str}, parse_dates=["date"])


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data-quality report", ""]
    for k, v in report.items():
        lines.append(f"- **{k}**: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download raw data first")
    ap.add_argument("--build", action="store_true", help="build processed tables")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    a = ap.parse_args()
    if a.fetch:
        fetch_raw_data(a.raw_dir)
    if a.build:
        fixtures, matches, report = build_clean_datasets(a.raw_dir)
        report["alignment_with_match_vectors"] = verify_alignment_with_paper_vectors(matches, a.raw_dir)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        fixtures.drop(columns=["home_team_of", "away_team_of"]).to_csv(PROCESSED_DIR / "fixtures_full.csv", index=False)
        matches.to_csv(PROCESSED_DIR / "matches_clean.csv", index=False)
        save_report(report, TABLES_DIR / "data_quality_report.md")
        for k, v in report.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
