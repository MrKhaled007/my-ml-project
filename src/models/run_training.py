from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.loader import DATA_DIR, load_csv

FEATURES_PATH = Path(DATA_DIR) / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
PRODUCTION_PATH = MODELS_DIR / "production_model.pkl"

TARGET = "Class"
RANDOM_STATE = 42
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "fraud_detection"


def compute_metrics(model, X, y, prefix: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y, preds)),
        f"{prefix}_precision": float(precision_score(y, preds, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y, preds, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y, preds, zero_division=0)),
        f"{prefix}_roc_auc": float(roc_auc_score(y, proba)),
        f"{prefix}_pr_auc": float(average_precision_score(y, proba)),
    }


def build_baseline(scale_pos_weight: float):
    params = {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    return LogisticRegression(**params), params


def build_tuned(scale_pos_weight: float):
    tuned = json.loads(BEST_PARAMS_PATH.read_text())
    fixed = {
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    model = XGBClassifier(**tuned, **fixed)
    return model, {**tuned, **fixed}


def train_and_log(name, model, hyperparams, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_name", name)
        mlflow.log_param("model_class", model.__class__.__name__)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        train_metrics = compute_metrics(model, X_train, y_train, "train")
        test_metrics = compute_metrics(model, X_test, y_test, "test")
        mlflow.log_metric("train_time_s", train_time)
        mlflow.log_metrics({**train_metrics, **test_metrics})

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / f"{name}.pkl"
            joblib.dump(model, artifact_path)
            mlflow.log_artifact(str(artifact_path), artifact_path="model")

        print(
            f"[{name}] train_time={train_time:.1f}s "
            f"test_roc_auc={test_metrics['test_roc_auc']:.4f} "
            f"test_pr_auc={test_metrics['test_pr_auc']:.4f} "
            f"test_f1={test_metrics['test_f1']:.4f}"
        )
        return model, test_metrics


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking: {MLFLOW_TRACKING_URI} | experiment: {EXPERIMENT_NAME}")

    print(f"Loading: {FEATURES_PATH}")
    df = load_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | scale_pos_weight: {spw:.2f}")

    baseline_model, baseline_params = build_baseline(spw)
    tuned_model, tuned_params = build_tuned(spw)

    configs = [
        ("baseline", baseline_model, baseline_params),
        ("tuned_best", tuned_model, tuned_params),
    ]

    final_model = None
    for name, model, hyperparams in configs:
        trained, _ = train_and_log(name, model, hyperparams, X_train, y_train, X_test, y_test)
        if name == "tuned_best":
            final_model = trained

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, PRODUCTION_PATH)
    print(f"\nProduction model saved: {PRODUCTION_PATH}")
    print(f"View runs at: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
