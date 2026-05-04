from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_csv

FEATURES_PATH = Path(__file__).resolve().parent / "fixtures" / "features_sample.csv"
TARGET = "Class"


# Train fresh per session instead of loading a pickled model. A pickled model
# from one sklearn version often can't be loaded by another (cross-version
# attribute drift), and CI's pinned-by-pip sklearn won't match the dev env.
@pytest.fixture(scope="module")
def model_and_X():
    df = load_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe, X.head(20)


def test_model_loads_and_predicts(model_and_X):
    model, sample_X = model_and_X
    proba = model.predict_proba(sample_X)
    assert proba.shape == (len(sample_X), 2)
    assert proba.sum(axis=1) == pytest.approx(1.0, abs=1e-6)


def test_predictions_in_expected_range(model_and_X):
    model, sample_X = model_and_X
    proba = model.predict_proba(sample_X)[:, 1]
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()
