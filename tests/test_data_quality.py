from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_csv
from src.data.quality import check_data_quality

CLEANED_PATH = Path(__file__).resolve().parent / "fixtures" / "cleaned_sample.csv"


@pytest.fixture(scope="module")
def cleaned_df():
    return load_csv(CLEANED_PATH)


def test_quality_gate_passes_on_cleaned_data(cleaned_df):
    result = check_data_quality(cleaned_df)
    assert result["success"], f"unexpected failures: {result['failures']}"


def test_quality_gate_catches_broken_dataframe():
    bad = pd.DataFrame({
        "Time": [1, 2, 3, 4, 5],
        "Amount": [10.0, np.inf, -5.0, 20.0, 30.0],
        "count_items": [1, -1, 2, -2, 3],
    })
    result = check_data_quality(bad)

    assert not result["success"]
    assert len(result["failures"]) > 0
