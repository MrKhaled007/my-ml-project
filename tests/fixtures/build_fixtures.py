"""Regenerate small test fixtures from data/cleaned.csv.

Run after the cleaned-data schema or feature-engineering output changes:
    python tests/fixtures/build_fixtures.py

Outputs (committed to the repo so CI can run pytest without raw data):
    cleaned_sample.csv   stratified subset of cleaned data
    features_sample.csv  result of create_features on the subset
    model_sample.pkl     LogisticRegression trained on those features
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_csv
from src.features.engineering import create_features

ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = ROOT / "data" / "cleaned.csv"
FIXTURES_DIR = ROOT / "tests" / "fixtures"

N_ROWS = 1000
RANDOM_STATE = 42
TARGET = "Class"


def main() -> None:
    df = load_csv(CLEANED_PATH)
    sample, _ = train_test_split(
        df,
        train_size=N_ROWS,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    sample = sample.reset_index(drop=True)

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(FIXTURES_DIR / "cleaned_sample.csv", index=False)
    print(f"  cleaned_sample.csv: {sample.shape}")

    enriched = create_features(sample)
    enriched.to_csv(FIXTURES_DIR / "features_sample.csv", index=False)
    print(f"  features_sample.csv: {enriched.shape}")

    X = enriched.drop(columns=[TARGET])
    y = enriched[TARGET]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    model.fit(X, y)
    joblib.dump(model, FIXTURES_DIR / "model_sample.pkl")
    print(f"  model_sample.pkl: {type(model).__name__} on {X.shape[1]} features")


if __name__ == "__main__":
    main()
