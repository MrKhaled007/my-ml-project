from pathlib import Path

import joblib
import pytest

from src.data.loader import load_csv

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODEL_PATH = FIXTURES / "model_sample.pkl"
FEATURES_PATH = FIXTURES / "features_sample.csv"
TARGET = "Class"


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_X():
    return load_csv(FEATURES_PATH).drop(columns=[TARGET]).head(20)


def test_model_loads_and_predicts(model, sample_X):
    proba = model.predict_proba(sample_X)
    assert proba.shape == (len(sample_X), 2)
    assert proba.sum(axis=1) == pytest.approx(1.0, abs=1e-6)


def test_predictions_in_expected_range(model, sample_X):
    proba = model.predict_proba(sample_X)[:, 1]
    assert (proba >= 0.0).all()
    assert (proba <= 1.0).all()
