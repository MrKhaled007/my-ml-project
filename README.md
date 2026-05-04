# my-ml-project

[![CI](https://github.com/MrKhaled007/my-ml-project/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MrKhaled007/my-ml-project/actions/workflows/ci.yml)

Credit card fraud detection on anonymized transaction data.

## Exploratory Data Analysis

See [`notebooks/eda.ipynb`](notebooks/eda.ipynb) for the full analysis.

**Dataset.** 283,726 rows × 31 columns. 30 numeric features (`float64`) plus an integer `Class` target (0 = legitimate, 1 = fraud). Features are `Time`, `Amount`, and 28 PCA-derived components `V1`–`V28`. No missing values.

**Key findings.**
- **Severe class imbalance** — 283,253 legitimate vs. 473 fraud (~599:1). Accuracy is meaningless here; evaluation must use precision/recall, ROC-AUC, and PR-AUC.
- **`Amount` is heavily right-skewed** (skew ≈ 17.0; median $22, max $25,691). Linear models will need a `log1p` transform before scaling.
- **Top fraud signals** are `V17`, `V14`, `V12`, `V10`, `V16` (|corr| with `Class` from 0.31 down to 0.19) — the rest of the `V*` features are near-zero correlated.
- **`V*` features are mutually near-orthogonal** (PCA components by construction); the strongest inter-feature correlation is `V2`–`Amount` at 0.53.

**Modeling implications.**
- Use stratified splits and resampling (SMOTE / class weights) to handle the 599:1 imbalance.
- Apply `log1p` to `Amount` and standard-scale before any distance- or gradient-based model.
