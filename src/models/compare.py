from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.data.loader import DATA_DIR, load_csv

FEATURES_PATH = Path(DATA_DIR) / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TARGET = "Class"
RANDOM_STATE = 42


def build_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def main() -> None:
    print(f"Loading: {FEATURES_PATH}")
    df = load_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns | positives: {int(y.sum())} ({y.mean():.4%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | scale_pos_weight: {spw:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model in build_models(spw).items():
        print(f"\n--- {name} ---")
        print("5-fold CV (roc_auc)...")
        t0 = time.time()
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        cv_time = time.time() - t0
        print(f"CV folds: {cv_scores.round(4).tolist()}  ({cv_time:.1f}s)")

        print("Fitting on full train set...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        proba = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)

        out_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, out_path)
        print(f"Saved: {out_path}")

        rows.append({
            "model": name,
            "cv_mean_auc": round(float(cv_scores.mean()), 4),
            "cv_std_auc": round(float(cv_scores.std()), 4),
            "test_auc": round(float(test_auc), 4),
            "train_time_s": round(float(train_time), 2),
        })

    table = pd.DataFrame(rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison (ROC-AUC) ===")
    print(table.to_string(index=False))

    best = table.iloc[0]
    print(f"\nBest by test ROC-AUC: {best['model']} ({best['test_auc']:.4f})")


if __name__ == "__main__":
    main()
