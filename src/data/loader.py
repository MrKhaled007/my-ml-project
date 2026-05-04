from pathlib import Path
import sys

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def find_csv(data_dir: Path = DATA_DIR) -> Path:
    csvs = sorted(p for p in data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return csvs[0]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def describe(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumns and dtypes:")
    print(df.dtypes.to_string())

    numeric = df.select_dtypes(include="number")
    print("\nSummary statistics (numeric columns):")
    if numeric.empty:
        print("  (no numeric columns)")
    else:
        print(numeric.agg(["mean", "std", "min", "max"]).to_string())

    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2) if len(df) else missing * 0
    print("\nMissing values:")
    print(pd.DataFrame({"count": missing, "percent": pct}).to_string())


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    print(f"Loading: {path}")
    df = load_csv(path)
    describe(df)


if __name__ == "__main__":
    main()
