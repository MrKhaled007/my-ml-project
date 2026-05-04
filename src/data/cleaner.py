from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loader import DATA_DIR, find_csv, load_csv
from src.data.quality import check_data_quality

DEFAULT_OUTPUT = DATA_DIR / "cleaned.csv"


def clean_data(
    df: pd.DataFrame,
    target: str | None = None,
    time_series: bool = False,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    null_rate = df.isna().mean()
    high_null_cols = null_rate[null_rate > 0.50].index.tolist()
    df = df.drop(columns=high_null_cols)

    if target and target in df.columns:
        df = df[df[target].notna()]

    if time_series:
        df = df.ffill()
    else:
        df = df.dropna()

    df = df.drop_duplicates(keep="first")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().all():
            df[col] = coerced
        else:
            df[col] = df[col].astype("string")

    df = df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    quality = check_data_quality(df, target=target)
    return df, quality


def main() -> None:
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    print(f"Loading: {path}")

    raw = load_csv(path)
    print(f"Before cleaning: {raw.shape[0]} rows, {raw.shape[1]} columns")

    cleaned, quality = clean_data(raw)
    print(f"After cleaning:  {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(f"Saved to: {DEFAULT_OUTPUT}\n")

    print(f"Quality success: {quality['success']}")
    print(f"\nFailures ({len(quality['failures'])}):")
    for f in quality["failures"]:
        print(f"  - {f}")
    print(f"\nWarnings ({len(quality['warnings'])}):")
    for w in quality["warnings"]:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
