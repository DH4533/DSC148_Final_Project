"""
train.py
--------
Main entry point: loads data, engineers features, trains models, evaluates.

Usage:
    python train.py                   # train all models
    python train.py --model xgboost   # train one model
    python train.py --model all --test_year 2023
"""

import argparse
import sys
from pathlib import Path

# Allow `python train.py` from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_processed
from src.feature_engineering import build_features
from src.models import get_model, MODEL_REGISTRY
from src.evaluate import (
    compute_metrics, print_report,
    plot_roc_curves, plot_pr_curves, plot_calibration,
    plot_comparison_table, plot_feature_importance, plot_confusion,
)
from src.utils import set_seed, temporal_split, stratified_split, model_save_path


def train_and_evaluate(model_name: str, X_train, y_train, X_val, y_val, X_test, y_test,
                       feature_names: list, save_model: bool = True):

    model = get_model(model_name)

    # XGBoost can use a validation set for early stopping
    if model_name == "xgboost":
        model.fit(X_train.values, y_train.values,
                  X_val=X_val.values, y_val=y_val.values)
    elif model_name == "neural_net":
        model.fit(X_train.values, y_train.values,
                  X_val=X_val.values, y_val=y_val.values)
    else:
        model.fit(X_train.values, y_train.values)

    # Predict on test set
    y_pred = model.predict(X_test.values)
    y_prob = model.predict_proba(X_test.values)

    metrics = print_report(model_name, y_test.values, y_pred, y_prob)

    plot_confusion(y_test.values, y_pred, model_name=model_name)
    plot_feature_importance(model, feature_names, model_name=model_name)

    if save_model:
        model.save(model_save_path(model_name))

    return model, metrics, {"y_true": y_test.values, "y_prob": y_prob}


def main():
    parser = argparse.ArgumentParser(description="Train swing-and-miss prediction models")
    parser.add_argument("--model",     default="all",
                        choices=list(MODEL_REGISTRY) + ["all"],
                        help="Which model to train")
    parser.add_argument("--test_year", type=int, default=None,
                        help="If set, use temporal split on this year as test set")
    parser.add_argument("--no_save",   action="store_true",
                        help="Don't save model weights")
    args = parser.parse_args()

    set_seed(42)

    # ── Load & feature-engineer ───────────────────────────────────────────────
    print("\n=== Loading data ===")
    df = load_processed()
    print(f"Loaded {len(df):,} pitches")

    X, y, feature_names = build_features(df)

    # ── Split ─────────────────────────────────────────────────────────────────
    print("\n=== Splitting data ===")
    if args.test_year and "game_date" in df.columns:
        import pandas as pd
        df["game_date"] = pd.to_datetime(df["game_date"])
        test_mask  = df["game_date"].dt.year >= args.test_year
        X_test_df  = X[test_mask];  y_test_s  = y[test_mask]
        X_train_full = X[~test_mask]; y_train_full = y[~test_mask]
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1,
            stratify=y_train_full, random_state=42
        )
        X_test, y_test = X_test_df, y_test_s
        print(f"Temporal split: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # ── Train ─────────────────────────────────────────────────────────────────
    model_names = list(MODEL_REGISTRY) if args.model == "all" else [args.model]

    all_results = {}
    all_metrics = {}

    print(f"\n=== Training {len(model_names)} model(s) ===")
    for name in model_names:
        _, metrics, res = train_and_evaluate(
            name, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names, save_model=not args.no_save,
        )
        all_metrics[name] = metrics
        all_results[name] = res

    # ── Aggregate plots ───────────────────────────────────────────────────────
    if len(model_names) > 1:
        print("\n=== Generating comparison plots ===")
        plot_roc_curves(all_results)
        plot_pr_curves(all_results)
        plot_calibration(all_results)
        plot_comparison_table(all_metrics)

    print("\n✅  Done! Results saved to results/")


if __name__ == "__main__":
    main()
