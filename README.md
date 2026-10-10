# EPL Match-Outcome Prediction — Person 1: Data & Feature Engineering

This repository prepares the dataset and feature tables for a two-person project that compares the paper's original EPL match-prediction approach with a separate, historical-only prediction approach.

**Important distinction:** the original paper features are retained for reproduction. They are not treated as valid pre-match predictors because the released ratings/statistics use end-of-season roster information and actual minutes played. The recommended primary model uses only information derived from matches completed before the fixture.

## Quick start

```bash
pip install -r requirements.txt
bash run_person1.sh
```

The script fetches the raw source data, rebuilds processed data/features, runs leakage checks and feature analysis, then runs the tests.

Raw downloads are not committed (`data/raw/` is git-ignored). Run the script from the repository root.

## Outputs

| Command | Main outputs |
|---|---|
| `python -m src.data_preprocessing --fetch --build` | `matches_clean.csv`, `fixtures_full.csv`, data-quality report |
| `python -m src.improved_features` | `features_engineered.csv`, `paper_f3.csv`, `feature_groups.json` |
| `python -m src.leakage_checks` | `leakage_report.json`, feature/outcome correlation table |
| `python -m src.feature_analysis` | figures and supporting CSV tables |
| `python -m pytest -q` | automated tests |

## Dataset notes

- The paper describes 3,800 EPL matches across 2009/10–2018/19.
- Its released vectors contain 3,791 rows, including one duplicated fixture. The clean match table retains 3,790 unique matches.
- The fixture table contains all 3,800 fixtures, including matches absent from the paper's vector files. These fixtures are retained to make chronological team histories more complete.
- Dates are joined from the openfootball fixture lists. Scores and labels are checked against the paper's vectors.
- Labels are `H` (home win), `D` (draw), and `A` (away win). `label` is an encoded target column, not a model feature.

## Feature tracks

### Track A — faithful paper reproduction

- `paper_F1`: overall home/away roster ratings.
- `paper_F2`: overall and position-wise ratings.
- `paper_F3`: detailed player statistics in the separate `paper_f3.csv` table.
- These groups exist to reproduce and compare with the original paper. They are **not leakage-free pre-match features**: the original roster statistics are end-of-season values and positional weights use minutes played in the target match.

Do not combine these groups into the primary historical-only model and then describe that model as leakage-free.

### Track B — recommended historical-only prediction features

All historical features are calculated chronologically. For each date, the pipeline calculates features for all fixtures before applying any results from that date. No feature uses the target fixture's result or a later result.

| Feature group | Contents |
|---|---|
| `recent_form_w3`, `recent_form_w5`, `recent_form_w10` | Previous-match points, W/D/L counts, goals for/against, goal difference, points per match and form trend |
| `elo`, `elo_extra` | Pre-match home/away Elo, Elo difference, expected home result and absolute Elo gap |
| `venue_form` | Home team's previous home form and away team's previous away form |
| `historical_differences_w3/w5/w10` | Differences between the two teams' corresponding historical-form features |
| `historical_venue_differences` | Differences between home-specific and away-specific historical venue statistics |
| `historical_differences` | Historical form and venue differences only; excludes rating-derived differences |
| `historical_only` | Recommended starting feature set: last-5 form + Elo + venue form + historical differences |
| `historical_only_extended` | Expanded historical-only feature set using 3/5/10-match windows and the same Elo/venue information |
| `history_counts` | Number of prior observed matches for each team |

The `historical_only` and `historical_only_extended` groups do not include `paper_F1`, `paper_F2`, `paper_F3`, or player-rating features.

### Separate sensitivity track — previous-season player ratings

- `prematch_ratings_sensitivity` contains `pm_*` player-rating features derived from the previous season's ratings.
- These features use the target match's recorded starting XI. Unless the team sheet is verified as available before kickoff, this lineup may be retrospective.
- Consequently, `historical_plus_ratings_sensitivity` is a **sensitivity experiment**, not the strict leakage-free primary model.
- 2009/10 ratings are NaN because there is no previous season. Missing values must be imputed inside each training fold, never before splitting.

The generic `diff_*` columns derived from the paper's roster/position ratings are retained for analysis, but are deliberately excluded from `historical_differences` and the historical-only groups.

## Experiment plan for Person 2

1. **E0 — Paper reproduction:** compare the original F1/F2/F3 features with the paper's model families, where the source files allow it. The paper reports 58.89% test accuracy for F1 + degree-5 polynomial SVM.
2. **E1 — Historical baseline:** `historical_only` with a tuned polynomial SVM.
3. **E2 — Expanded history:** compare `historical_only_extended` with E1.
4. **E3 — New models:** evaluate Random Forest, Gradient Boosting and HistGradientBoosting on the same training/validation protocol.
5. **E4 — Sensitivity analysis:** optionally test `historical_plus_ratings_sensitivity`, clearly labelled as lineup-dependent.
6. **E5 — Ensemble:** try soft voting only if validation indicates complementary model performance.

Change one feature group at a time where possible. Use cross-validation only on training data for tuning and selection. Report both a paper-style random-split comparison and a temporal evaluation on later seasons. Keep the final test seasons untouched until the final evaluation.

Recommended metrics: accuracy, macro F1, per-class precision/recall/F1, draw precision/recall, confusion matrix, and train/validation gap.

## How Person 2 should load feature groups

```python
import json
import pandas as pd
from src.data_preprocessing import read_processed

df = read_processed("features_engineered.csv")
with open("data/processed/feature_groups.json", encoding="utf-8") as f:
    groups = json.load(f)

# Recommended primary feature set
X = df[groups["historical_only"]]
y = df["label"]

# Expanded historical-only experiment
X_extended = df[groups["historical_only_extended"]]

# Paper F3 is stored separately; align by paper_row/fixture_id before using it
# for the paper-reproduction experiment only.
paper_f3 = pd.read_csv("data/processed/paper_f3.csv")
```

Use a scikit-learn `Pipeline` so imputation, scaling, feature selection and the classifier are fitted inside each training fold. Do not include `home_goals`, `away_goals`, `result`, `label`, IDs, dates or team names in `X`.

## Historical-feature implementation details

- Rolling windows use matches strictly earlier than the target date.
- Partial windows are scaled to the configured window for point/count features; no-history features are NaN and should be imputed within training folds.
- Elo defaults: initial rating 1500, K=20, home advantage 65, goal-margin multiplier and 1/3 seasonal regression toward 1500. These are initial settings, not tuned parameters.
- Promoted/returning clubs receive a regressed starting Elo and their within-league histories are reset for the season.
- These settings should be documented and, if time permits, varied through training-only validation.

## Leakage checks and limitations

`src/leakage_checks.py` checks chronological ordering, independently recomputes selected history values, perturbs match results to verify that the match's own and earlier features remain unchanged, checks target-like columns in feature groups, and checks previous-season rating availability.

Run the complete script and test suite from a clean environment before submission. The tests require raw data to be fetched first.

The paper's exact reported split may not be fully reproducible because the release's shuffled vectors do not document the shuffle seed for every feature set. Record differences honestly instead of forcing our reproduction to match 58.89%.

## Suggested contribution split

**Person 1:** dataset acquisition, cleaning, target labels, paper feature reconstruction, chronological historical features, leakage checks and feature analysis.

**Person 2:** model pipelines, hyperparameter tuning, cross-validation, temporal evaluation, model comparisons, ensemble testing and final model serialization.

**Both:** integration, demo, README, report, presentation, repository maintenance and viva preparation.
