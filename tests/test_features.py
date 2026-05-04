from pathlib import Path

import pytest

from src.data.loader import load_csv
from src.features.engineering import create_features

CLEANED_PATH = Path(__file__).resolve().parent / "fixtures" / "cleaned_sample.csv"

EXPECTED_NEW_COLS = [
    "amount_log",
    "is_zero_amount",
    "hour_of_day",
    "is_night",
    "amount_zscore",
    "amount_rolling_mean",
    "amount_rolling_std",
    "amount_dev_from_rolling",
    "v_extreme_count",
    "top_signal_magnitude",
    "top_signal_min",
    "v17_x_v14",
    "v14_x_v12",
    "amount_per_v17_magnitude",
    "night_high_amount",
]


@pytest.fixture(scope="module")
def enriched():
    df = load_csv(CLEANED_PATH)
    return create_features(df)


def test_create_features_adds_expected_columns(enriched):
    missing = [c for c in EXPECTED_NEW_COLS if c not in enriched.columns]
    assert not missing, f"missing engineered columns: {missing}"
    assert enriched.shape[1] == 31 + len(EXPECTED_NEW_COLS), (
        f"unexpected column count: {enriched.shape[1]}"
    )


def test_no_nan_in_output(enriched):
    null_counts = enriched.isna().sum()
    offenders = null_counts[null_counts > 0].to_dict()
    assert not offenders, f"NaN found in columns: {offenders}"


def test_feature_ranges(enriched):
    assert enriched["is_zero_amount"].isin([0, 1]).all()
    assert enriched["is_night"].isin([0, 1]).all()
    assert enriched["hour_of_day"].between(0, 23).all()
    assert (enriched["amount_log"] >= 0).all()
    assert (enriched["amount_rolling_std"] >= 0).all()
    assert (enriched["v_extreme_count"] >= 0).all()
    assert (enriched["top_signal_magnitude"] >= 0).all()
