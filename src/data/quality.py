from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.loader import find_csv, load_csv


def check_data_quality(
    df: pd.DataFrame,
    required_columns: Sequence[str] | None = None,
    required_dtypes: Mapping[str, str] | None = None,
    target: str | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> dict:
    failures: list[str] = []
    warnings: list[str] = []
    stats: dict = {
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "total_nulls_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    _check_schema(df, required_columns, required_dtypes, failures)
    _check_row_count(df, failures, warnings)
    _check_null_rates(df, failures, warnings, stats)
    _check_value_ranges(df, bounds or {}, failures, warnings)
    _check_target_distribution(df, target, failures, warnings, stats)

    return {
        "success": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "statistics": stats,
    }


def _check_schema(df, required_columns, required_dtypes, failures):
    for col in required_columns or []:
        if col not in df.columns:
            failures.append(f"Schema: missing required column '{col}'")

    for col, expected in (required_dtypes or {}).items():
        if col not in df.columns:
            failures.append(f"Schema: missing column '{col}' (expected dtype {expected})")
            continue
        actual = str(df[col].dtype)
        if not _dtype_matches(actual, expected):
            failures.append(f"Schema: column '{col}' has dtype {actual}, expected {expected}")


def _dtype_matches(actual: str, expected: str) -> bool:
    expected = expected.lower()
    actual = actual.lower()
    if expected in {"numeric", "number"}:
        return any(k in actual for k in ("int", "float"))
    if expected in {"int", "integer"}:
        return "int" in actual
    if expected == "float":
        return "float" in actual
    if expected in {"str", "string", "object"}:
        return actual in {"object", "string"}
    if expected in {"bool", "boolean"}:
        return "bool" in actual
    return actual == expected


def _check_row_count(df, failures, warnings):
    n = len(df)
    if n < 100:
        failures.append(f"Row count: only {n} rows (need >= 100)")
    elif n < 1000:
        warnings.append(f"Row count: {n} rows is below 1000 (small sample)")


def _check_null_rates(df, failures, warnings, stats):
    n = max(len(df), 1)
    rates = (df.isna().sum() / n).to_dict()
    stats["null_rates_by_column"] = {c: float(round(r, 4)) for c, r in rates.items()}
    for col, rate in rates.items():
        if rate > 0.50:
            failures.append(f"Nulls: column '{col}' is {rate:.1%} null (> 50%)")
        elif rate > 0.20:
            warnings.append(f"Nulls: column '{col}' is {rate:.1%} null (> 20%)")


def _check_value_ranges(df, bounds, failures, warnings):
    numeric = df.select_dtypes(include="number")

    for col in numeric.columns:
        s = numeric[col]

        n_inf = int(np.isinf(s).sum())
        if n_inf:
            failures.append(f"Range: column '{col}' has {n_inf} infinite values")

        lo, hi = bounds.get(col, (None, None))
        if lo is not None and (s < lo).any():
            failures.append(f"Range: column '{col}' has values below {lo}")
        if hi is not None and (s > hi).any():
            failures.append(f"Range: column '{col}' has values above {hi}")

        name = col.lower()
        if any(tok in name for tok in ("count", "qty", "quantity", "n_", "num_")):
            if (s < 0).any():
                failures.append(f"Range: count-like column '{col}' has negative values")

        if any(tok in name for tok in ("pct", "percent", "percentage")):
            if (s < 0).any() or (s > 100).any():
                warnings.append(f"Range: percent-like column '{col}' outside [0, 100]")

        if any(tok in name for tok in ("rate", "ratio", "prob", "probability")):
            if (s < 0).any() or (s > 1).any():
                warnings.append(f"Range: rate-like column '{col}' outside [0, 1]")


def _check_target_distribution(df, target, failures, warnings, stats):
    if target is None:
        return
    if target not in df.columns:
        failures.append(f"Target: column '{target}' not found")
        return

    counts = df[target].value_counts(dropna=False)
    n = max(len(df), 1)
    proportions = (counts / n).to_dict()
    stats["target_distribution"] = {str(k): float(round(v, 4)) for k, v in proportions.items()}

    if len(counts) < 2:
        failures.append(f"Target: '{target}' has {len(counts)} class(es), need >= 2")
        return

    for cls, p in proportions.items():
        if p < 0.05:
            failures.append(f"Target: class '{cls}' is {p:.1%} of data (< 5%)")

    p_max, p_min = max(proportions.values()), min(proportions.values())
    if p_min > 0 and p_max / p_min > 3:
        warnings.append(
            f"Target: imbalanced ({p_max:.1%} majority vs {p_min:.1%} minority)"
        )


def _print_report(result: dict) -> None:
    print(f"Success: {result['success']}")

    print(f"\nFailures ({len(result['failures'])}):")
    for f in result["failures"]:
        print(f"  - {f}")

    print(f"\nWarnings ({len(result['warnings'])}):")
    for w in result["warnings"]:
        print(f"  - {w}")

    print("\nStatistics:")
    for k, v in result["statistics"].items():
        print(f"  {k}: {v}")


def main() -> None:
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    print(f"Loading: {path}\n")
    df = load_csv(path)
    result = check_data_quality(df)
    _print_report(result)


if __name__ == "__main__":
    main()
