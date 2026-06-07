"""
evaluate.py
-----------
Metrics, calibration plots, ROC curves, feature importance charts,
and the final model comparison table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "logistic":      "#4C8EDA",
    "random_forest": "#E07B4F",
    "xgboost":       "#3BAA6E",
    "neural_net":    "#9B5ED5",
}


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a dict of all evaluation metrics."""
    return {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1":               f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":          roc_auc_score(y_true, y_prob),
        "avg_precision":    average_precision_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
def print_report(model_name: str, y_true, y_pred, y_prob):
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\n{'═'*50}")
    print(f"  {model_name.upper()}")
    print(f"{'═'*50}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (binary):     {metrics['f1']:.4f}")
    print(f"  F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:   {metrics['avg_precision']:.4f}")
    print(classification_report(y_true, y_pred, target_names=["no swing", "swinging ★"]))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, save: bool = True):
    """
    Overlay ROC curves for all models.
    results = {model_name: {"y_true": ..., "y_prob": ...}}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        auc = roc_auc_score(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(name, "grey"),
                label=f"{name}  (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Swing-and-Miss Prediction", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
        print(f"Saved → {RESULTS_DIR / 'roc_curves.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_pr_curves(results: dict, save: bool = True):
    """Precision-Recall curves (more informative under class imbalance)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ap = average_precision_score(res["y_true"], res["y_prob"])
        ax.plot(rec, prec, lw=2, color=PALETTE.get(name, "grey"),
                label=f"{name}  (AP={ap:.3f})")

    positive_rate = np.mean(list(results.values())[0]["y_true"])
    ax.axhline(positive_rate, color="k", ls="--", lw=1, alpha=0.5,
               label=f"Baseline ({positive_rate:.2%})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "pr_curves.png", dpi=150)
        print(f"Saved → {RESULTS_DIR / 'pr_curves.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_calibration(results: dict, save: bool = True):
    """Calibration curve: does predicted probability match empirical frequency?"""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")

    for name, res in results.items():
        frac_pos, mean_pred = calibration_curve(
            res["y_true"], res["y_prob"], n_bins=10, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "o-", lw=2, ms=5,
                color=PALETTE.get(name, "grey"), label=name)

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives",      fontsize=12)
    ax.set_title("Calibration Curves",          fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / "calibration.png", dpi=150)
        print(f"Saved → {RESULTS_DIR / 'calibration.png'}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, top_n: int = 20,
                            model_name: str = "", save: bool = True):
    try:
        fi = model.feature_importance(feature_names).head(top_n)
    except (AttributeError, Exception):
        print(f"  (feature importance not available for {model_name})")
        return

    fig, ax = plt.subplots(figsize=(8, top_n * 0.35 + 1))
    colors = [PALETTE.get(model_name, "#4C8EDA")] * len(fi)
    fi[::-1].plot.barh(ax=ax, color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=13, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    if save:
        fname = RESULTS_DIR / f"feature_importance_{model_name}.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison_table(all_metrics: dict, save: bool = True):
    """
    Side-by-side bar chart of key metrics across all models.
    all_metrics = {model_name: {metric: value, ...}}
    """
    df = pd.DataFrame(all_metrics).T[["accuracy", "f1", "roc_auc", "avg_precision"]]
    df.columns = ["Accuracy", "F1", "ROC-AUC", "Avg Precision"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(df.columns))
    width  = 0.18
    models = list(df.index)

    for i, (name, row) in enumerate(df.iterrows()):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width,
                      label=name, color=PALETTE.get(name, "grey"),
                      edgecolor="white", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Swing-and-Miss Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    plt.tight_layout()

    if save:
        fig.savefig(RESULTS_DIR / "comparison.png", dpi=150)
        print(f"Saved → {RESULTS_DIR / 'comparison.png'}")

    # Also save CSV
    df.to_csv(RESULTS_DIR / "metrics.csv")
    print(f"Saved → {RESULTS_DIR / 'metrics.csv'}")
    print("\n", df.to_string())
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion(y_true, y_pred, model_name: str = "", save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No swing", "Swinging ★"],
                yticklabels=["No swing", "Swinging ★"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save:
        fig.savefig(RESULTS_DIR / f"confusion_{model_name}.png", dpi=150)
    return fig
