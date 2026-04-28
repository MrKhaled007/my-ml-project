from __future__ import annotations

import logging
import time
from pathlib import Path

from src.data.loader import DATA_DIR, load_csv
from src.features.engineering import create_features, select_features

CLEANED_PATH = Path(DATA_DIR) / "cleaned.csv"
FEATURES_PATH = Path(DATA_DIR) / "features.csv"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()

    print(f"Loading: {CLEANED_PATH}")
    df = load_csv(CLEANED_PATH)
    print(f"Before: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nEngineering features...")
    enriched = create_features(df)
    print(f"After create_features: {enriched.shape[0]} rows, {enriched.shape[1]} columns")

    print("\nSelecting features...")
    selected, reduced = select_features(enriched)
    print(f"After select_features: {reduced.shape[0]} rows, {reduced.shape[1]} columns")

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved to: {FEATURES_PATH}")

    print(f"\nKept {len(selected)} features:")
    for c in selected:
        print(f"  - {c}")

    print(f"\nElapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
