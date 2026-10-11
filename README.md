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

## Person 2 — Model Training, Evaluation & Inference Implementation

Person 2 builds on Person 1's clean, leakage-free feature tables to implement the complete modeling, temporal evaluation, hyperparameter tuning, model comparison, ensembling, and serialization infrastructure.

### Quick start (Person 2)

```bash
# Run the complete Person 2 pipeline and test suite
bash run_person2.sh
```

Or execute individual modular commands:

```bash
# 1. Inspect dataset splits and verify model pipeline creation
python -m src.model_training

# 2. Run Protocol A (Experiment E0: Paper-style reproduction)
python scripts/run_paper_reproduction.py

# 3. Run Protocol B (Experiments E1-E5: Chronological evaluation, tuning, ensemble, serialization)
python scripts/run_model_experiments.py

# 4. Run production inference demonstration on held-out test fixtures
python scripts/demo_inference.py --fixture_index 0

# 5. Run full automated test suite (25 tests)
python -m pytest -v
```

---

## Evaluation Protocols

### Protocol A: Original Paper Reproduction (Random 80:20 Split)
- Reproduces the original paper's random 80:20 train/test split.
- Evaluates feature sets `paper_F1` (2 squad ratings), `paper_F2` (10 position ratings), and `paper_F3` (280 detailed player match statistics).
- Evaluates original model families: Polynomial SVM (degree 5 & 3), RBF SVM, Multinomial Logistic Regression, Gaussian Discriminant Analysis (LDA & QDA), and One-hidden-layer Neural Network (MLP).
- **Paper Reference Comparison:**
  - Original paper reports **58.89%** test accuracy for F1 with a degree-5 polynomial SVM.
  - Our reproduction achieves **57.12%** test accuracy (a difference of only -1.77 percentage points, attributable to the paper's undocumented shuffle seed and deduplication of fixture row 434).

### Protocol B: Chronological Future-Season Evaluation (Primary Protocol)
In real-world soccer forecasting, matches occur chronologically: a model must be trained on past seasons and predict unseen future fixtures without lookahead leakage.
- **Strict Chronological Ordering:**
  - **Training Period:** 2009/10–2015/16 (7 seasons, 2,657 matches, ~70.1%)
  - **Validation Period:** 2016/17 (1 season, 379 matches, ~10.0%) — used strictly for model selection, thresholding, and ensemble tuning.
  - **Final Test Period:** 2017/18–2018/19 (2 seasons, 754 matches, ~19.9%) — held out untouched until final model evaluation.
  - Verification: `max(Date_Train) < min(Date_Val) <= max(Date_Val) < min(Date_Test)`. Zero cross-boundary leakage.
- **Cross-Validation within Training:**
  - Expanding-window 3-fold chronological cross-validation:
    - Fold 0: Train 2009/10–2012/13, Validate 2013/14
    - Fold 1: Train 2009/10–2013/14, Validate 2014/15
    - Fold 2: Train 2009/10–2014/15, Validate 2015/16

---

## Experimental Results Summary

Master results are saved to `results/tables/model_comparison.csv`.

| Exp | Protocol | Feature Group | Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Draw Recall | Key Insight |
|---|---|---|---|---|---|---|---|---|---|
| **E0** | Protocol A (Random 80:20) | `paper_F1` | Poly-5 SVM | — | — | **57.12%** | 0.4262 | 0.5% | Reproduces paper reference (58.89%) within 1.77% |
| **E0** | Protocol A (Random 80:20) | `paper_F3` | Logistic Reg | — | — | **54.75%** | 0.5031 | 27.7% | 280 player stats improve draw recall under random split |
| **E1** | Protocol B (Chronological) | `historical_only` | Tuned Poly-2 SVM | 51.19% | 0.4885 | 51.19% | 0.4953 | 41.8% | Balanced weighting boosts draw recall from 0% to 41.8% |
| **E2** | Protocol B (Chronological) | `historical_only_extended` | Tuned Poly-2 SVM | 48.28% | 0.4628 | 49.20% | 0.4754 | 38.8% | Adding 3/10-match windows diluted performance |
| **E3** | Protocol B (Chronological) | `historical_only` | Multinomial Logistic Reg | 52.24% | 0.4951 | **53.45%** | **0.5162** | 42.4% | Highest individual test accuracy & Macro-F1 |
| **E3** | Protocol B (Chronological) | `historical_only` | RBF SVM | 50.66% | 0.4915 | 51.19% | 0.4974 | 42.9% | Strong balanced recall across H, D, and A |
| **E3** | Protocol B (Chronological) | `historical_only` | GDA (LDA) | 57.52% | 0.4736 | **55.84%** | 0.4641 | 10.6% | High accuracy by predicting mostly Home wins; misses draws |
| **E3** | Protocol B (Chronological) | `historical_only` | GDA (QDA) | 52.77% | 0.4893 | 52.65% | 0.4856 | 27.6% | Class-specific covariances improve draw capture |
| **E3** | Protocol B (Chronological) | `historical_only` | Neural Net (MLP) | 55.41% | 0.4675 | 54.77% | 0.4718 | 15.9% | Moderate generalization, tends toward majority class |
| **E3** | Protocol B (Chronological) | `historical_only` | Random Forest | 50.66% | 0.4901 | 50.40% | 0.4944 | **46.5%** | Highest draw recall among all single models |
| **E3** | Protocol B (Chronological) | `historical_only` | HistGradientBoosting | 53.03% | 0.4922 | 51.72% | 0.4766 | 24.7% | Fast tree-based boosting with binned features |
| **E4** | Protocol B (Chronological) | `historical_plus_ratings_sensitivity` | Logistic Reg (Sensitivity) | 53.30% | **0.5118** | 51.59% | 0.4928 | 34.7% | Previous-season ratings do not beat pure match history |
| **E5** | Protocol B (Chronological) | `historical_only` | **Soft Voting Ensemble (HistGB+LR+RF)** | **54.62%** | **0.5074** | **51.86%** | **0.4826** | 28.2% | **Champion Architecture:** Highest validation score, smooth predicted probabilities via soft voting |
| **Refit** | Protocol B (Train + Val Refit) | `historical_only` | **Selected Champion (Refitted on 0910–1617)** | — | — | **51.99%** | **0.4877** | 30.0% | **Production Model:** Refitted on all pre-test data (3,036 matches) for final serialization |

> **Note on Evaluation Protocol & Discrepancies:**
> 1. In `results/tables/model_comparison.csv`, all Protocol B models (including E5) are fitted strictly on `df_train` (seasons 2009/10–2015/16, 2,657 matches) for unbiased, fair benchmark comparison. Under this partition, E5 achieves 51.86% test accuracy.
> 2. Model selection was performed **strictly using validation macro-F1** on season 2016/17 (Ensemble 0.5074 vs Logistic Regression 0.4951). Test labels were completely withheld during selection.
> 3. Standard ML deployment practice refits the selected architecture on `df_train + df_val` (seasons 2009/10–2016/17, 3,036 matches) before saving to disk. On the held-out test set (seasons 2017/18–2018/19, 754 matches), this serialized model achieves **51.99% test accuracy**, **0.4877 test macro-F1**, and **30.0% draw recall**.
> 4. Probabilities are generated via soft-voting class aggregation across base estimators rather than formal post-hoc calibration curves.

---

## Key Viva & Mini-Project Findings

1. **Reproducibility of the Original Paper:**
   - The paper's 58.89% result on F1 with degree-5 SVM is closely reproducible (we observed 57.12%).
   - However, the paper's F1/F2/F3 features rely on end-of-season roster ratings and match minutes played (post-match information), making them unsuitable for live pre-match forecasting.
2. **Historical Match Features Generalize Well:**
   - On true unseen future seasons (2017/18 and 2018/19, 754 matches), historical-only features (last-5 form, Elo ratings, home/away venue form, and differential metrics) achieve **53.45% accuracy** and **0.5162 Macro-F1** with Logistic Regression, and **51.99% accuracy** with the refitted ensemble.
   - The majority class baseline on the final test set (always predicting Home Win) is **46.29%** (349 / 754 matches; 46.17% across the entire 10-season dataset), meaning the pre-match historical features provide a solid statistical edge of **+7.16 percentage points** for Logistic Regression and **+5.70 percentage points** for the ensemble.
3. **Compact vs Extended History:**
   - `historical_only` (47 features) outperformed `historical_only_extended` (101 features, 49.20% accuracy). Expanding rolling windows to 3 and 10 games introduced feature collinearity without adding predictive signal.
4. **The Draw Prediction Challenge:**
   - Draws represent ~24.8% of matches. Unweighted models frequently achieve 0% to 10% draw recall because predicting Home/Away wins maximizes raw accuracy.
   - Incorporating class weighting and soft ensembling boosted draw recall to **28%–46%**, dramatically improving Macro-F1 and tactical utility.
5. **Model Selection & Champion Ensemble:**
   - Selecting strictly by validation Macro-F1 identified the **Soft Voting Ensemble (HistGradientBoosting + Multinomial Logistic Regression + Random Forest)**.
   - Refit on combined 2009/10–2016/17 data and evaluated once on the untouched 2017/18–2018/19 test period, it delivers balanced performance across all three match outcomes (Home Recall: 59.0%, Draw Recall: 30.0%, Away Recall: 57.5%).

---

## Production Model Serialization & Inference

The final trained champion pipeline is serialized and ready for deployment:
- **Model weights:** `results/models/epl_model.joblib`
- **Metadata & schema:** `results/models/model_metadata.json`

### Python Inference API

```python
import pandas as pd
from src.model_inference import EPLMatchPredictor

# 1. Load the serialized pipeline and schema
predictor = EPLMatchPredictor.load(
    model_path="results/models/epl_model.joblib",
    metadata_path="results/models/model_metadata.json",
)

# 2. Predict on fixture features (validates schema & auto-aligns column order)
sample_fixture = pd.read_csv("data/processed/features_engineered.csv").iloc[[0]]
prediction = predictor.predict(sample_fixture)

print(prediction)
# Output columns:
# ['predicted_outcome', 'predicted_label_full', 'prob_H', 'prob_D', 'prob_A']
```

---

## Generated Artifacts

### Figures (`results/figures/`)
- `01_outcome_share_by_season.png` — Outcome distribution by season
- `02_draw_rate_vs_elo_gap.png` — Draw frequency vs Elo difference
- `03_elo_diff_by_outcome.png` — Elo differences by result
- `04_top_feature_correlations.png` — Spearman correlations with goal difference
- `05_elo_trajectories.png` — Team Elo trajectory over time
- `06_prematch_rating_coverage.png` — Pre-match player rating coverage
- `07_model_accuracy_comparison.png` — Chronological test accuracy across model families
- `08_model_macro_f1_comparison.png` — Chronological test macro-F1 comparison
- `09_per_class_recall_comparison.png` — Grouped recall by class (H / D / A)
- `10_selected_model_confusion_matrix.png` — Champion model confusion matrix heatmap
- `11_feature_group_comparison.png` — Historical vs Extended vs Sensitivity feature performance
- `12_draw_prediction_performance.png` — Draw recall comparison
- `e0_paper_reproduction_accuracy.png` — Protocol A reproduction vs paper reference

### Tables (`results/tables/`)
- `model_comparison.csv` — Master evaluation table with all required columns
- `paper_reproduction_results.csv` — Protocol A results table
- `chronological_confusion_matrices.csv` — Confusion matrices for all chronological models
- `e0_confusion_matrices.csv` — Confusion matrices for paper reproduction models
- `dataset_summary.csv`, `feature_catalog.csv`, `leakage_report.json` — Preprocessing & quality reports

---

## Automated Test Suite

All 25 unit and integration tests run in under 8 seconds:

```bash
python -m pytest -v
```

1. `tests/test_person1_pipeline.py` (14 tests): Dataset shape, duplicate elimination, paper vector alignment, chronology, perturbation leakage checks, zero-sum Elo, feature separation.
2. `tests/test_model_pipeline.py` (5 tests): Dataset loading, target leakage prevention, valid target classes, missing value imputation inside pipeline, metric calculation and label order `["H", "D", "A"]`.
3. `tests/test_chronological_split.py` (3 tests): Strict non-overlapping partition, time arrow verification (`Train < Val < Test`), expanding-window cross-validation.
4. `tests/test_model_inference.py` (3 tests): Model serialization round-trip, feature reordering, missing feature rejection, output prediction schema.

---

## Authors & Responsibilities

- **Person 1:** Dataset acquisition, cleaning, target labels, paper feature reconstruction, chronological historical feature engineering, leakage verification, exploratory feature analysis.
- **Person 2:** Model pipeline architecture, chronological evaluation design, hyperparameter tuning, paper-style reproduction, model comparison (8 model families), ensemble design, model serialization, inference interface, automated test suite, documentation.
- **Both:** End-to-end integration, demo preparation, research report & viva presentation.
