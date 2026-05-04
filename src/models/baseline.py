from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.loader import DATA_DIR, load_csv

FEATURES_PATH = Path(DATA_DIR) / "features.csv"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "baseline.pkl"
TARGET = "Class"


def is_classification(y: pd.Series) -> bool:
    if y.dtype == object or pd.api.types.is_bool_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return True
    return False


def evaluate_classification(model, X_test, y_test) -> None:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    average = "binary" if y_test.nunique() == 2 else "weighted"

    if y_test.nunique() == 2:
        scores = proba[:, 1]
        auc = roc_auc_score(y_test, scores)
    else:
        auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, average=average, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, preds, average=average, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_test, preds, average=average, zero_division=0):.4f}")
    print(f"AUC-ROC:   {auc:.4f}")


def evaluate_regression(model, X_test, y_test) -> None:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")


def main() -> None:
    print(f"Loading: {FEATURES_PATH}")
    df = load_csv(FEATURES_PATH)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    classification = is_classification(y)
    task = "classification" if classification else "regression"
    print(f"Task: {task} (target='{TARGET}')")

    stratify = y if classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    if classification:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = LinearRegression()

    print(f"\nTraining {model.__class__.__name__}...")
    model.fit(X_train, y_train)

    print("\nTest set metrics:")
    if classification:
        evaluate_classification(model, X_test, y_test)
    else:
        evaluate_regression(model, X_test, y_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
