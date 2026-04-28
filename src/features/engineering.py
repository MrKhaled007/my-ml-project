from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import DATA_DIR, load_csv

logger = logging.getLogger(__name__)

CLEANED_PATH = Path(DATA_DIR) / "cleaned.csv"

# Drop one of any feature pair whose absolute Pearson correlation exceeds
# this threshold — they carry essentially the same information and inflate
# multicollinearity for linear models without helping tree models either.
CORR_THRESHOLD = 0.95

# Variance floor as a fraction of the mean per-feature variance. Anything
# below this is near-constant and contributes no discriminative power.
VARIANCE_THRESHOLD_RATIO = 0.01

# Columns the selector must never drop — the target plus any business keys.
PROTECTED_COLS = ("Class",)

# Top fraud-correlated PCA components from EDA (|corr| with Class >= 0.19).
TOP_SIGNAL_COLS = ["V17", "V14", "V12", "V10", "V16"]

# Rolling window over time-sorted transactions. ~500 rows ≈ a short
# behavioral window in this dataset's transaction cadence — large enough
# to be stable, small enough to react to bursts of card-testing activity.
ROLLING_WINDOW = 500


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # === Domain-specific features ===

    # Amount is skew ≈ 17 (median $22, max $25,691). Linear/distance models
    # see the long tail as noise; log1p compresses it while preserving order
    # and keeping zero-amount transactions at zero.
    out["amount_log"] = np.log1p(out["Amount"])

    # $0 (or near-zero) authorizations are a classic card-testing pattern:
    # fraudsters validate stolen cards with tiny pre-auths before a real charge.
    out["is_zero_amount"] = (out["Amount"] < 1.0).astype(int)

    # Time is seconds since the first transaction in the dataset. Modulo
    # 86,400 recovers hour-of-day, which exposes circadian fraud patterns
    # that a raw monotonic Time column hides from tree splits.
    out["hour_of_day"] = ((out["Time"] % 86_400) // 3_600).astype(int)

    # Card-not-present fraud disproportionately fires overnight when the
    # cardholder is asleep and unlikely to notice an SMS alert.
    out["is_night"] = ((out["hour_of_day"] >= 0) & (out["hour_of_day"] < 6)).astype(int)

    # Global standardization of Amount — turns "is this charge unusually
    # large?" into a single comparable number across the whole population.
    amt_mean, amt_std = out["Amount"].mean(), out["Amount"].std()
    out["amount_zscore"] = (out["Amount"] - amt_mean) / (amt_std + 1e-9)

    # === Statistical / rolling features ===

    # Sort by Time so the rolling window respects causal order, then restore
    # the original row order at the end so downstream joins still align.
    order = out["Time"].argsort(kind="stable")
    amt_sorted = out["Amount"].iloc[order]

    # Local average spend in the recent transaction window. Bursts of
    # similar-sized small charges (testing) or a sudden jump above the
    # local norm (cash-out) both deviate from this baseline.
    roll_mean = amt_sorted.rolling(ROLLING_WINDOW, min_periods=1).mean()

    # Local volatility. Calm windows (low std) followed by a large charge
    # are more suspicious than the same charge inside an already-volatile
    # window — std contextualizes the size of the deviation below.
    roll_std = amt_sorted.rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0.0)

    out["amount_rolling_mean"] = roll_mean.reindex(out.index)
    out["amount_rolling_std"] = roll_std.reindex(out.index)

    # Signed deviation from the local norm. Positive spikes flag charges
    # that are anomalously large relative to recent activity, regardless
    # of where the dataset sits on its global Amount scale.
    out["amount_dev_from_rolling"] = out["Amount"] - out["amount_rolling_mean"]

    # PCA components are standardized to roughly unit variance; |z|>3 on
    # any single V* is a tail event. Counting how many V* are simultaneously
    # extreme is a model-agnostic anomaly score that complements tree splits.
    v_cols = [c for c in out.columns if c.startswith("V") and c[1:].isdigit()]
    v_block = out[v_cols]
    out["v_extreme_count"] = (v_block.abs() > 3).sum(axis=1)

    # Aggregate magnitude of the top fraud-correlated PCA components.
    # Single-feature thresholds miss fraud that spreads its signal across
    # several correlated components; the L2 norm captures that combined push.
    top_block = out[TOP_SIGNAL_COLS]
    out["top_signal_magnitude"] = np.sqrt((top_block ** 2).sum(axis=1))

    # Fraud in this dataset shows strongly negative tails on V17/V14/V12,
    # so the minimum across the top signals is a sharper fraud indicator
    # than the magnitude (which is symmetric and ignores direction).
    out["top_signal_min"] = top_block.min(axis=1)

    # === Interaction features ===

    # V17 and V14 are the two strongest individual fraud signals. Their
    # product blows up only when BOTH are extreme in the same direction —
    # a co-occurrence pattern a single-feature split cannot represent.
    out["v17_x_v14"] = out["V17"] * out["V14"]

    # Secondary interaction: V14 × V12. Captures the "second tier" of the
    # fraud signature for cases where V17 alone is unremarkable but the
    # supporting components are firing together.
    out["v14_x_v12"] = out["V14"] * out["V12"]

    # Large dollar amount paired with an extreme V17 signal is the classic
    # "successful cash-out after testing" pattern — neither feature alone
    # is rare, but the combination is. Divide by magnitude (not raw V17)
    # so direction is absorbed and the ratio stays well-defined.
    out["amount_per_v17_magnitude"] = out["amount_log"] / (out["V17"].abs() + 1e-3)

    # Night-time + high amount: cardholder is asleep AND the charge is
    # unusually large. Either alone is common; together they're rare and
    # disproportionately fraudulent in card-not-present datasets.
    out["night_high_amount"] = out["is_night"] * out["amount_log"]

    return out


def select_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    # Operate only on numeric candidate features. Protected columns (target,
    # ids) are passed through unchanged regardless of their stats.
    protected = [c for c in PROTECTED_COLS if c in df.columns]
    candidates = [
        c for c in df.select_dtypes(include="number").columns if c not in protected
    ]

    # --- Variance filter ---
    # Min-max scale each candidate to [0, 1] before measuring variance.
    # Raw variance is scale-dependent — Time (seconds, ~5e7) would
    # otherwise drown out unit-variance PCA features and the threshold
    # would prune everything. Scaling makes "near-constant" mean the
    # same thing across dollar amounts, seconds, and PCA components.
    raw = df[candidates]
    span = raw.max() - raw.min()
    scaled = (raw - raw.min()) / span.replace(0, np.nan)
    variances = scaled.var().fillna(0.0)
    overall_variance = variances.mean()
    variance_floor = VARIANCE_THRESHOLD_RATIO * overall_variance

    low_var = variances[variances < variance_floor]
    for col, var in low_var.items():
        logger.info(
            "drop %s: scaled variance=%.6g < floor=%.6g (%.1f%% of mean %.6g)",
            col, var, variance_floor, VARIANCE_THRESHOLD_RATIO * 100, overall_variance,
        )

    surviving = [c for c in candidates if c not in low_var.index]

    # --- Correlation filter ---
    # Walk the upper triangle of |corr|; for each pair above threshold,
    # drop the column appearing later in `surviving` and keep the first
    # one. This gives stable, order-deterministic selection.
    corr = df[surviving].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    to_drop: list[str] = []
    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper.index[upper[col] > CORR_THRESHOLD].tolist()
        for partner in partners:
            if partner in to_drop or partner == col:
                continue
            logger.info(
                "drop %s: |corr| with %s = %.4f > %.2f (kept %s, seen first)",
                col, partner, upper.loc[partner, col], CORR_THRESHOLD, partner,
            )
            to_drop.append(col)
            break

    selected = [c for c in surviving if c not in to_drop]

    logger.info(
        "selection summary: %d candidates -> %d kept (%d low-variance, %d redundant)",
        len(candidates), len(selected), len(low_var), len(to_drop),
    )

    reduced = df[selected + protected].copy()
    return selected, reduced


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Loading cleaned data: {CLEANED_PATH}")
    df = load_csv(CLEANED_PATH)
    print(f"Input:  {df.shape[0]} rows, {df.shape[1]} columns")

    enriched = create_features(df)
    new_cols = [c for c in enriched.columns if c not in df.columns]
    print(f"Output: {enriched.shape[0]} rows, {enriched.shape[1]} columns")
    print(f"Engineered {len(new_cols)} new features:")
    for c in new_cols:
        print(f"  - {c}")

    print("\nRunning feature selection...")
    selected, reduced = select_features(enriched)
    print(f"Selected {len(selected)} features, reduced shape: {reduced.shape}")


if __name__ == "__main__":
    main()
