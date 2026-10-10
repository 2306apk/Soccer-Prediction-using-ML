"""
feature_analysis.py  --  Person 1, task 6 (feature analysis + visualisations)

Run:  python -m src.feature_analysis

Writes tables to results/tables and figures to results/figures.  Colours are the
validated categorical slots 1-4 (blue, orange, aqua, yellow); the aqua/yellow slots are
below 3:1 on the light surface, so every chart carries direct labels and every figure has
a CSV "table view" next to it in results/tables.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_preprocessing import PROCESSED_DIR, TABLES_DIR, read_processed
from src.improved_features import build_history_features

FIG_DIR = TABLES_DIR.parent / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLASS_COLOR = {"H": BLUE, "D": ORANGE, "A": AQUA}
CLASS_NAME = {"H": "Home win", "D": "Draw", "A": "Away win"}


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def dataset_summary(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in matches.groupby("season"):
        c = g["result"].value_counts()
        rows.append({"season": s, "fixtures_total": int((fixtures["season"] == s).sum()), "matches_in_paper_data": len(g),
                     "H": int(c.get("H", 0)), "D": int(c.get("D", 0)), "A": int(c.get("A", 0)),
                     "home_win_pct": round(100 * c.get("H", 0) / len(g), 1), "draw_pct": round(100 * c.get("D", 0) / len(g), 1),
                     "away_win_pct": round(100 * c.get("A", 0) / len(g), 1),
                     "avg_goals_per_match": round((g["home_goals"] + g["away_goals"]).mean(), 2)})
    return pd.DataFrame(rows)


def feature_catalog(groups: dict[str, list[str]]) -> pd.DataFrame:
    desc = {
        "paper_F1": "Paper F1: mean season-end rating of the match squad (home, away)",
        "paper_F2": "Paper F2: F1 + GK/DEF/MID/ATT time-weighted ratings",
        "recent_form": "Points/W/D/L, goals for/against/diff averages, PPG and trend over the last w matches (any venue)",
        "differences": "Home-minus-away differences of the corresponding features",
        "elo": "Pre-match Elo of home and away team and their difference",
        "elo_extra": "Elo-based home win expectancy and absolute Elo gap (closeness -> draws)",
        "venue_form": "Home team's last 10 HOME games / away team's last 10 AWAY games: PPG, win & draw rate, goals for/against",
        "prematch_ratings": "Starting-XI ratings from the PREVIOUS season (pre-match-safe alternative to F2)",
        "history_counts": "Number of earlier matches a team has in the data (history length)",
    }
    leak = {"paper_F1": "NOT pre-match (season-end rating includes this match)", "paper_F2": "NOT pre-match (season-end rating + minutes played in this match)"}
    rows = []
    def description_key(group: str) -> str:
        # Feature-group metadata contains windowed and composite groups such as
        # historical_differences_w3, historical_only and paper_F3. Map them
        # deliberately so new groups do not crash catalog generation.
        if group == "paper_F1":
            return "paper_F1"
        if group == "paper_F2":
            return "paper_F2"
        if group == "paper_F3":
            return "paper_F3"
        if group.startswith("recent_form"):
            return "recent_form"
        if "differences" in group:
            return "differences"
        if group.startswith("historical_only"):
            return "historical_only"
        if group.startswith("historical_plus_ratings"):
            return "historical_plus_ratings"
        if group.startswith("prematch_ratings"):
            return "prematch_ratings"
        if group.startswith("venue_form"):
            return "venue_form"
        if group.startswith("historical_venue"):
            return "historical_venue_differences"
        if group.startswith("elo_extra"):
            return "elo_extra"
        if group.startswith("elo"):
            return "elo"
        if group.startswith("history_counts"):
            return "history_counts"
        return "other"

    desc.update({
        "paper_F3": "Paper F3: detailed player-level statistics from the original implementation",
        "historical_only": "Leakage-conscious pre-match historical features: form, Elo, venue form and historical differences",
        "historical_plus_ratings": "Historical features plus previous-season starting-XI ratings; sensitivity experiment",
        "historical_venue_differences": "Home-minus-away differences in venue-specific historical form",
        "other": "Feature group generated by the preprocessing pipeline; inspect the feature name and source definition",
    })
    leak.update({
        "paper_F3": "NOT pre-match unless original feature availability is independently verified",
        "historical_plus_ratings_sensitivity": "sensitivity experiment; lineup availability requires verification",
        "prematch_ratings_sensitivity": "sensitivity experiment; lineup availability requires verification",
    })

    for g, cols in groups.items():
        key = description_key(g)
        description = desc.get(key, desc["other"])
        availability = leak.get(g, "yes")
        for c in cols:
            rows.append({"feature": c, "group": g, "description": description,
                         "available_before_kickoff": availability})
    return pd.DataFrame(rows).drop_duplicates("feature")



def missingness(eng: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    # Some feature groups, especially paper_F3, are stored separately
    # and are not columns of features_engineered.csv.
    requested_cols = {
        col
        for group_cols in groups.values()
        for col in group_cols
    }

    # Only use columns that actually exist in this dataframe.
    cols = sorted(requested_cols.intersection(eng.columns))

    if not cols:
        return pd.DataFrame(
            columns=["feature", "missing", "missing_pct"]
        )

    missing_counts = eng[cols].isna().sum()
    missing_counts = missing_counts.rename("missing").to_frame()
    missing_counts["missing_pct"] = (
        100 * missing_counts["missing"] / len(eng)
    ).round(2)

    return (
        missing_counts[missing_counts["missing"] > 0]
        .sort_values("missing", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )


# ---------------------------------------------------------------- figures
def fig_outcomes_by_season(matches: pd.DataFrame) -> pd.DataFrame:
    t = matches.groupby(["season", "result"]).size().unstack(fill_value=0)[["H", "D", "A"]]
    share = 100 * t.div(t.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(len(share))
    x = np.arange(len(share))
    for k in ("H", "D", "A"):
        ax.bar(x, share[k], bottom=bottom, color=CLASS_COLOR[k], width=0.78, edgecolor=SURFACE, linewidth=2, label=CLASS_NAME[k])
        for xi, (b, v) in enumerate(zip(bottom, share[k])):
            ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if k == "H" else INK)
        bottom += share[k].to_numpy()
    ax.set_xticks(x, [f"20{s[:2]}/{s[2:]}" for s in share.index])
    ax.set_ylim(0, 100)
    _style(ax, "Draws are about a quarter of matches; home wins about 46%", ylabel="% of matches")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), labelcolor=INK2)
    _save(fig, "01_outcome_share_by_season.png")
    return share.round(1).reset_index()


def fig_draw_rate_vs_elo_gap(eng: pd.DataFrame) -> pd.DataFrame:
    d = eng[eng["season"] != "0910"].copy()                       # skip the Elo warm-up season
    d["gap_bin"] = pd.qcut(d["abs_elo_diff"], 8)
    t = d.groupby("gap_bin", observed=True).agg(draw_rate=("result", lambda r: 100 * (r == "D").mean()),
                                                 matches=("result", "size"), mean_gap=("abs_elo_diff", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t["mean_gap"], t["draw_rate"], color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for _, r in t.iterrows():
        ax.annotate(f"{r['draw_rate']:.0f}%", (r["mean_gap"], r["draw_rate"]), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    ax.set_ylim(0, max(40, t["draw_rate"].max() + 8))
    _style(ax, "Evenly matched teams draw more often", "|Elo difference| before the match (octile means)", "draw rate (%)")
    _save(fig, "02_draw_rate_vs_elo_gap.png")
    return t.drop(columns="gap_bin")


def fig_elo_by_outcome(eng: pd.DataFrame) -> pd.DataFrame:
    d = eng[eng["season"] != "0910"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    data = [d.loc[d["result"] == k, "elo_diff"] for k in ("H", "D", "A")]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, linewidth=1.6))
    for patch, k in zip(bp["boxes"], ("H", "D", "A")):
        patch.set(facecolor=CLASS_COLOR[k], edgecolor=SURFACE, linewidth=2, alpha=0.9)
    ax.set_xticks([1, 2, 3], [CLASS_NAME[k] for k in ("H", "D", "A")])
    ax.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
    _style(ax, "Elo advantage separates home from away wins; draws sit in between", "", "home Elo - away Elo (pre-match)")
    _save(fig, "03_elo_diff_by_outcome.png")
    return d.groupby("result")["elo_diff"].describe().round(1).reset_index()


def fig_correlations(corr: pd.DataFrame) -> pd.DataFrame:
    leaky = {"paper_F2"}
    pre_ratings = corr["group"].isin(["prematch_ratings"])
    c = corr.copy()
    c["abs"] = c["spearman_vs_goal_diff"].abs()
    c["origin"] = np.where(c["group"].isin(leaky) | c["feature"].isin(["diff_roster_rating", "diff_gk_rating", "diff_def_rating", "diff_mid_rating", "diff_off_rating"]), "paper ratings (season-end, not pre-match)", "pre-match")
    top = c.sort_values("abs", ascending=False).drop_duplicates("feature").head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    cols = [ORANGE if o.startswith("paper") else BLUE for o in top["origin"]]
    mag = top["abs"].to_numpy()
    ax.barh(top["feature"], mag, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, (m, v) in enumerate(zip(mag, top["spearman_vs_goal_diff"])):
        ax.text(m + 0.008, y, f"{v:+.2f}", va="center", fontsize=8, color=INK2)   # sign shown in the label
    ax.set_xlim(0, mag.max() + 0.08)
    _style(ax, "Strength of association with final goal difference (|Spearman|; sign in label)", "|correlation|", "")
    ax.grid(axis="x", color=GRID, linewidth=0.7); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="pre-match feature"), Patch(color=ORANGE, label="paper rating (season-end, not pre-match)")],
              frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.12), labelcolor=INK2)
    _save(fig, "04_top_feature_correlations.png")
    return top.iloc[::-1][["group", "feature", "origin", "spearman_vs_goal_diff", "spearman_vs_home_win"]].round(3).reset_index(drop=True)


def fig_elo_trajectories(fixtures: pd.DataFrame, hist: pd.DataFrame, teams=("Man City", "Man Utd", "Chelsea", "Leicester")) -> pd.DataFrame:
    fx = fixtures.set_index("fixture_id").join(hist[["h_elo", "a_elo"]])
    rows = []
    for t in teams:
        h = fx[fx["home_team"] == t][["date", "h_elo"]].rename(columns={"h_elo": "elo"})
        a = fx[fx["away_team"] == t][["date", "a_elo"]].rename(columns={"a_elo": "elo"})
        s = pd.concat([h, a]).sort_values("date"); s["team"] = t
        rows.append(s)
    long = pd.concat(rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ends = {t: long.loc[long["team"] == t, "elo"].iloc[-1] for t in teams}
    label_y, prev = {}, None
    for t in sorted(teams, key=lambda k: ends[k]):               # nudge end labels apart (min 22 Elo)
        y = ends[t] if prev is None else max(ends[t], prev + 22)
        label_y[t], prev = y, y
    for t, col in zip(teams, (BLUE, ORANGE, AQUA, YELLOW)):
        s = long[long["team"] == t]
        ax.plot(s["date"], s["elo"], color=col, linewidth=2, label=t)
        ax.text(s["date"].iloc[-1] + pd.Timedelta(days=25), label_y[t], t, color=INK, fontsize=8, va="center")
    ax.set_xlim(long["date"].min(), long["date"].max() + pd.Timedelta(days=170))
    _style(ax, "Pre-match Elo over time (2009/10 is the warm-up season)", "", "Elo")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), labelcolor=INK2)
    _save(fig, "05_elo_trajectories.png")
    return long.groupby("team")["elo"].agg(["min", "max", "last"]).round(0).reset_index()


def fig_coverage(eng: pd.DataFrame) -> pd.DataFrame:
    t = eng.groupby("season").agg(pm_rating_coverage=("pm_h_rating_coverage", "mean")).reset_index()
    t["pm_rating_coverage"] = (100 * t["pm_rating_coverage"]).round(1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(np.arange(len(t)), t["pm_rating_coverage"].fillna(0), color=BLUE, width=0.7, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(t["pm_rating_coverage"]):
        ax.text(i, (0 if np.isnan(v) else v) + 2, "n/a" if np.isnan(v) else f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(t)), [f"20{s[:2]}/{s[2:]}" for s in t["season"]])
    ax.set_ylim(0, 100)
    _style(ax, "Share of the starting XI with a previous-season rating", "", "% of starters")
    _save(fig, "06_prematch_rating_coverage.png")
    return t


def main() -> None:
    fixtures = read_processed("fixtures_full.csv")
    matches = read_processed("matches_clean.csv")
    eng = read_processed("features_engineered.csv")
    groups = json.loads((PROCESSED_DIR / "feature_groups.json").read_text())
    corr = pd.read_csv(TABLES_DIR / "feature_outcome_correlations.csv")
    hist = build_history_features(fixtures)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summary(matches, fixtures).to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
    feature_catalog(groups).to_csv(TABLES_DIR / "feature_catalog.csv", index=False)
    missingness(eng, groups).to_csv(TABLES_DIR / "feature_missingness.csv", index=False)
    fig_outcomes_by_season(matches).to_csv(TABLES_DIR / "fig01_outcome_share_by_season.csv", index=False)
    fig_draw_rate_vs_elo_gap(eng).to_csv(TABLES_DIR / "fig02_draw_rate_vs_elo_gap.csv", index=False)
    fig_elo_by_outcome(eng).to_csv(TABLES_DIR / "fig03_elo_diff_by_outcome.csv", index=False)
    fig_correlations(corr).to_csv(TABLES_DIR / "fig04_top_feature_correlations.csv", index=False)
    fig_elo_trajectories(fixtures, hist).to_csv(TABLES_DIR / "fig05_elo_trajectories.csv", index=False)
    fig_coverage(eng).to_csv(TABLES_DIR / "fig06_prematch_rating_coverage.csv", index=False)
    print("tables:", sorted(p.name for p in TABLES_DIR.glob("*")))
    print("figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))


if __name__ == "__main__":
    main()
