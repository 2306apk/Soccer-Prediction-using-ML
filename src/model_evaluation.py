"""
model_evaluation.py  --  Person 2, Model evaluation, metrics, and visualization

Computes comprehensive classification metrics with explicit label order:
  ["H", "D", "A"]
Confusion matrix rows = actual, columns = predicted.
Generates evaluation tables, confusion matrices, and publication-ready figures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Explicit target class ordering
CLASS_ORDER = ["H", "D", "A"]
CLASS_LABELS_FULL = {"H": "Home win (H)", "D": "Draw (D)", "A": "Away win (A)"}

# Styling constants matching Person 1 aesthetic
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLASS_COLORS = [BLUE, ORANGE, AQUA]


def compute_metrics(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    prefix: str = "",
) -> Dict[str, float]:
    """Compute overall accuracy, macro-F1, weighted-F1, and per-class precision/recall/F1."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    pfx = f"{prefix}_" if prefix and not prefix.endswith("_") else prefix

    acc = float(accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=CLASS_ORDER, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=CLASS_ORDER, average="weighted", zero_division=0))

    # Per-class metrics in order H, D, A
    prec = precision_score(y_t, y_p, labels=CLASS_ORDER, average=None, zero_division=0)
    rec = recall_score(y_t, y_p, labels=CLASS_ORDER, average=None, zero_division=0)
    f1 = f1_score(y_t, y_p, labels=CLASS_ORDER, average=None, zero_division=0)

    metrics = {
        f"{pfx}accuracy": round(acc, 4),
        f"{pfx}macro_f1": round(macro_f1, 4),
        f"{pfx}weighted_f1": round(weighted_f1, 4),
        f"{pfx}home_precision": round(float(prec[0]), 4),
        f"{pfx}home_recall": round(float(rec[0]), 4),
        f"{pfx}home_f1": round(float(f1[0]), 4),
        f"{pfx}draw_precision": round(float(prec[1]), 4),
        f"{pfx}draw_recall": round(float(rec[1]), 4),
        f"{pfx}draw_f1": round(float(f1[1]), 4),
        f"{pfx}away_precision": round(float(prec[2]), 4),
        f"{pfx}away_recall": round(float(rec[2]), 4),
        f"{pfx}away_f1": round(float(f1[2]), 4),
    }
    return metrics


def compute_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    labels: List[str] = CLASS_ORDER,
) -> np.ndarray:
    """Compute confusion matrix with rows = actual, columns = predicted."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def confusion_matrix_to_dataframe(
    cm: np.ndarray,
    labels: List[str] = CLASS_ORDER,
) -> pd.DataFrame:
    """Convert confusion matrix to formatted DataFrame."""
    index = [f"Actual_{lbl}" for lbl in labels]
    columns = [f"Pred_{lbl}" for lbl in labels]
    return pd.DataFrame(cm, index=index, columns=columns)


def get_classification_report(
    y_true: Union[pd.Series, np.ndarray, List[str]],
    y_pred: Union[pd.Series, np.ndarray, List[str]],
    labels: List[str] = CLASS_ORDER,
) -> str:
    """Return text classification report formatted with explicit class order."""
    target_names = [f"{lbl} ({CLASS_LABELS_FULL[lbl].split()[0]})" for lbl in labels]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )


# --------------------------------------------------------------------------
# Plotting Utilities
# --------------------------------------------------------------------------
def _setup_figure(title: str, xlabel: str = "", ylabel: str = "") -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10, labelpad=8)
    ax.tick_params(colors=INK2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    return fig, ax


def plot_metric_comparison(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    output_path: Path,
    ylabel: str = "Score",
    group_col: str = "model_name",
    baseline_val: Optional[float] = None,
) -> None:
    """Plot a horizontal bar comparison chart with direct value labels."""
    fig, ax = _setup_figure(title, xlabel=ylabel, ylabel="")
    ax.grid(axis="x", color=GRID, linewidth=0.7, linestyle="--", alpha=0.7)
    ax.grid(axis="y", visible=False)

    df_valid = df.dropna(subset=[metric_col]).copy()
    if df_valid.empty:
        plt.close(fig)
        return
    df_sorted = df_valid.sort_values(by=metric_col, ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_sorted))
    values = df_sorted[metric_col].values
    labels = df_sorted[group_col].values

    bars = ax.barh(y_pos, values, color=BLUE, height=0.6, edgecolor="none", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=INK)

    # Highlight maximum bar
    max_idx = int(np.nanargmax(values))
    bars[max_idx].set_color(ORANGE)

    # Direct value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="bold" if bar == bars[max_idx] else "normal",
        )

    if baseline_val is not None:
        ax.axvline(baseline_val, color=INK2, linestyle=":", linewidth=1.2, label=f"Baseline ({baseline_val:.3f})")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    max_val = float(np.nanmax(values))
    ax.set_xlim(0, max(max_val * 1.15, 0.6))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_per_class_recall(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Per-Class Recall Comparison (H / D / A)",
) -> None:
    """Grouped bar chart showing recall on H, D, and A for each evaluated model."""
    fig, ax = _setup_figure(title, xlabel="Model", ylabel="Recall")
    models = df["model_name"].values
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.24

    rec_h = df["home_recall"].values
    rec_d = df["draw_recall"].values
    rec_a = df["away_recall"].values

    rects1 = ax.bar(x - width, rec_h, width, label="Home Win (H)", color=BLUE, alpha=0.9)
    rects2 = ax.bar(x, rec_d, width, label="Draw (D)", color=ORANGE, alpha=0.9)
    rects3 = ax.bar(x + width, rec_a, width, label="Away Win (A)", color=AQUA, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    labels: List[str] = CLASS_ORDER,
) -> None:
    """Confusion matrix heatmap with actual vs predicted counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=12)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([f"Pred {lbl}" for lbl in labels], fontsize=9, color=INK)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([f"Actual {lbl}" for lbl in labels], fontsize=9, color=INK)

    ax.set_xlabel("Predicted Match Outcome", fontsize=9.5, color=INK2, labelpad=8)
    ax.set_ylabel("Actual Match Outcome", fontsize=9.5, color=INK2, labelpad=8)

    # Annotate counts and row-wise percentages
    row_sums = cm.sum(axis=1)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sums[i] * 100) if row_sums[i] > 0 else 0
            text_color = "white" if val > thresh else INK
            ax.text(
                j, i,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontsize=9, fontweight="bold",
            )

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_color(GRID)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
