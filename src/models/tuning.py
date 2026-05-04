from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import optuna
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.data.loader import DATA_DIR, load_csv

FEATURES_PATH = Path(DATA_DIR) / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PARAMS_PATH = MODELS_DIR / "best_params.json"
MODEL_PATH = MODELS_DIR / "tuned_model.pkl"
TARGET = "Class"
RANDOM_STATE = 42
N_TRIALS = 30
N_SPLITS = 5

log = logging.getLogger("tuning")


def make_objective(X_train, y_train, scale_pos_weight: float):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        t0 = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        elapsed = time.time() - t0

        mean = float(scores.mean())
        std = float(scores.std())
        trial.set_user_attr("cv_std", std)
        trial.set_user_attr("cv_time_s", elapsed)

        log.info(
            f"Trial {trial.number:>2}: cv_auc={mean:.4f} (+/-{std:.4f}) "
            f"time={elapsed:.1f}s | {params}"
        )
        return mean

    return objective


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Loading: {FEATURES_PATH}")
    df = load_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns | positives: {int(y.sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | scale_pos_weight: {spw:.2f}")

    print(f"\nStarting Optuna study: {N_TRIALS} trials, {N_SPLITS}-fold CV (roc_auc)\n")
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_fraud_tuning")
    study.optimize(make_objective(X_train, y_train, spw), n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  CV roc_auc mean: {best.value:.4f}")
    print(f"  CV roc_auc std:  {best.user_attrs.get('cv_std', float('nan')):.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best.params, indent=2))
    print(f"\nSaved best params: {PARAMS_PATH}")

    print("\nTraining final model on full training set with best params...")
    final = XGBClassifier(
        **best.params,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    final.fit(X_train, y_train)
    print(f"Fit time: {time.time() - t0:.1f}s")

    metrics = evaluate(final, X_test, y_test)
    print("\nTest set metrics (threshold=0.5):")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    joblib.dump(final, MODEL_PATH)
    print(f"\nSaved tuned model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
