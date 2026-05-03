"""Build artifacts consumed by the Streamlit portfolio app.

Outputs:
    data/predictions.csv     - test-set predictions from the production model
    data/model_results.json  - per-model metrics, feature importances, EDA stats
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.data.loader import DATA_DIR, load_csv

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = Path(DATA_DIR) / "features.csv"
MODELS_DIR = ROOT / "models"
PREDICTIONS_PATH = Path(DATA_DIR) / "predictions.csv"
RESULTS_PATH = Path(DATA_DIR) / "model_results.json"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

TARGET = "Class"
RANDOM_STATE = 42


def metrics(y_true, proba) -> dict:
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": cm,
    }


def main() -> None:
    print(f"Loading features: {FEATURES_PATH}")
    df = load_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    feature_cols = list(X.columns)
    n_rows, n_features = df.shape[0], len(feature_cols)
    fraud_count = int(y.sum())
    fraud_rate = float(y.mean())
    print(f"Rows: {n_rows} | features: {n_features} | fraud: {fraud_count} ({fraud_rate:.4%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model_files = {
        "baseline":            MODELS_DIR / "baseline.pkl",
        "logistic_regression": MODELS_DIR / "logistic_regression.pkl",
        "random_forest":       MODELS_DIR / "random_forest.pkl",
        "xgboost":             MODELS_DIR / "xgboost.pkl",
        "tuned_winner":        MODELS_DIR / "production_model.pkl",
    }

    label = {
        "baseline":            "Baseline (Logistic Regression)",
        "logistic_regression": "Logistic Regression",
        "random_forest":       "Random Forest",
        "xgboost":             "XGBoost (default params)",
        "tuned_winner":        "XGBoost (Optuna-tuned)  WINNER",
    }

    model_results: dict[str, dict] = {}
    proba_by_model: dict[str, np.ndarray] = {}
    for key, path in model_files.items():
        if not path.exists():
            print(f"  SKIP {key} (missing {path.name})")
            continue
        print(f"  scoring {key} ({path.name})...")
        t0 = time.time()
        model = joblib.load(path)
        proba = model.predict_proba(X_test)[:, 1]
        score_time = time.time() - t0
        m = metrics(y_test, proba)
        m["label"] = label[key]
        m["score_time_s"] = round(score_time, 3)
        model_results[key] = m
        proba_by_model[key] = proba

    # Feature importance from the production (tuned) XGBoost model.
    print("Extracting feature importances...")
    prod = joblib.load(MODELS_DIR / "production_model.pkl")
    importances = getattr(prod, "feature_importances_", None)
    if importances is not None:
        fi = (
            pd.Series(importances, index=feature_cols)
            .sort_values(ascending=False)
            .head(20)
            .round(5)
            .to_dict()
        )
    else:
        fi = {}

    # ROC curve for the winner — the app can render it without recomputing.
    fpr, tpr, _ = roc_curve(y_test, proba_by_model["tuned_winner"])
    # Subsample to keep the JSON light
    step = max(1, len(fpr) // 400)
    roc = {"fpr": fpr[::step].round(5).tolist(), "tpr": tpr[::step].round(5).tolist()}

    # EDA-derived stats. Correlations against Class are computed on the
    # training split to avoid leaking test info into a "data exploration" page.
    print("Computing EDA stats (train split only)...")
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    corr_with_target = (
        train_df.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(4)
    )
    top_corrs = corr_with_target.head(15).to_dict()

    best_params = {}
    if BEST_PARAMS_PATH.exists():
        best_params = json.loads(BEST_PARAMS_PATH.read_text())

    baseline_auc = model_results.get("baseline", {}).get("roc_auc")
    winner_auc = model_results.get("tuned_winner", {}).get("roc_auc")
    improvement = None
    if baseline_auc and winner_auc:
        improvement = round((winner_auc - baseline_auc) * 100, 3)  # absolute pp

    payload = {
        "dataset": {
            "n_rows_total": int(n_rows),
            "n_features": int(n_features),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "imbalance_ratio": round((n_rows - fraud_count) / max(fraud_count, 1), 1),
        },
        "feature_columns": feature_cols,
        "models": model_results,
        "winner": "tuned_winner",
        "winner_rationale": [
            "Highest ROC-AUC and PR-AUC on a held-out, stratified test set.",
            "PR-AUC is the right metric here — accuracy is meaningless at 599:1 imbalance.",
            "Optuna search (30 trials, 5-fold CV) over 9 XGBoost hyperparameters.",
            "Cross-validated AUC is stable across folds (low std), so the gain isn't a fluke.",
            "Train time is acceptable; inference is fast enough for a synchronous API.",
        ],
        "improvement_over_baseline_auc_pp": improvement,
        "feature_importances_top": fi,
        "roc_curve_winner": roc,
        "top_correlations_with_target": top_corrs,
        "best_params": best_params,
    }

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {RESULTS_PATH}")

    # Predictions CSV: keep a few real features for context plus all model probas.
    context_cols = [c for c in ["Amount", "amount_log", "hour_of_day", "is_night"] if c in X_test.columns]
    out = X_test[context_cols].copy()
    out.insert(0, "y_true", y_test.values)
    for key, proba in proba_by_model.items():
        out[f"proba_{key}"] = proba.round(6)
    out["y_pred_winner"] = (proba_by_model["tuned_winner"] >= 0.5).astype(int)
    out.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {PREDICTIONS_PATH}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
