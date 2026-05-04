# Credit Card Fraud Detection

[![CI](https://github.com/MrKhaled007/my-ml-project/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MrKhaled007/my-ml-project/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end binary classifier that flags fraudulent card transactions on a 283K-row dataset with a 599:1 class imbalance, served behind a Streamlit UI and packaged for one-command Docker deployment.

**🚀 Live demo:** _coming soon — placeholder URL_ → `https://my-ml-project.streamlit.app/`

---

## 1. Project Overview

**Problem.** Card-not-present fraud is rare (≈0.17% of transactions in this dataset) but expensive — every false negative is a chargeback, and every false positive is a legitimate customer locked out at checkout. A naïve "predict not-fraud" classifier scores 99.83% accuracy and catches zero fraud, so the entire modeling problem is about getting useful signal from the long tail.

**End user.** A risk analyst or fraud-ops team that wants a confidence score per transaction, plus an interpretable view of which features pushed each decision. The Streamlit app is the demo surface; the underlying pickled model is the artifact a production API would load.

**Data.** [Kaggle's Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 283,726 rows × 31 columns. Features are `Time` (seconds since first txn), `Amount` (USD), 28 PCA-anonymized numeric components `V1`–`V28`, and the binary `Class` target. No missing values. Severe imbalance: 283,253 legitimate vs. 473 fraud.

**Model output.** Per-transaction fraud probability ∈ [0, 1] from a tuned XGBoost classifier. The Streamlit app exposes a default 0.5 threshold but the rationale page argues for a tunable operating point chosen from the precision-recall curve based on the cost of a missed fraud vs. a false alarm.

**Key design decision.** We optimized for **PR-AUC, not ROC-AUC**. At 599:1 imbalance, ROC-AUC is dominated by trivially-classified negatives and saturates near 1.0 for any half-decent model. PR-AUC is the signal that actually distinguishes a useful fraud detector from a useless one — see [Key Decisions](#9-key-decisions--lessons) for the receipts.

---

## 2. Architecture

```
                       ┌───────────────────┐
                       │  creditcard.csv   │   raw Kaggle dataset
                       │  (283K × 31)      │   (gitignored, ~150 MB)
                       └─────────┬─────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │   src/data/cleaner.py       │   drop dupes, dropna,
                  │   src/data/quality.py       │   coerce dtypes,
                  │   (Great Expectations-      │   schema + null + range
                  │    style quality gate)      │   + target distribution
                  └──────────────┬──────────────┘
                                 │
                          cleaned.csv  ◄── 283,726 rows × 31 cols
                                 │
                  ┌──────────────▼──────────────┐
                  │   src/features/             │   15 engineered features
                  │   engineering.py            │   (log/zscore/rolling/
                  │   + select_features         │    interactions/anomaly)
                  └──────────────┬──────────────┘
                                 │
                          features.csv  ◄── 283,726 rows × 38 cols
                                 │
                  ┌──────────────▼──────────────┐
                  │   src/models/baseline.py    │   stratified split
                  │   src/models/compare.py     │   train 4 candidates,
                  │   src/models/tuning.py      │   Optuna search winner,
                  │   (MLflow tracked)          │   persist *.pkl
                  └──────────────┬──────────────┘
                                 │
                       ┌─────────┴─────────┐
                       │                   │
                  models/*.pkl       data/predictions.csv
                       │             data/model_results.json
                       │                   │
                       └─────────┬─────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │   app/streamlit_app.py      │   4-page demo:
                  │   (containerized via        │   overview, EDA,
                  │    Dockerfile + compose)    │   models, predictions
                  └─────────────────────────────┘
```

---

## 3. Results

All metrics on a held-out 20% stratified test split (56,746 rows, 95 fraud cases). Inference time is wall-clock for the full test set on a single CPU core.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | **PR-AUC** | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** (Logistic Regression) | 0.9681 | 0.0450 | **0.8947** | 0.0857 | 0.9754 | 0.6080 | 0.05 |
| Logistic Regression (re-trained) | 0.9681 | 0.0450 | 0.8947 | 0.0857 | 0.9754 | 0.6080 | 0.01 |
| Random Forest (n=100, default) | 0.9995 | **0.9571** | 0.7053 | 0.8121 | 0.9291 | 0.8072 | 0.18 |
| XGBoost (default params) | 0.9995 | 0.9359 | 0.7684 | 0.8439 | 0.9695 | 0.7941 | 0.19 |
| **XGBoost (Optuna-tuned)** 🏆 | 0.9993 | 0.7849 | 0.7684 | 0.7766 | **0.9760** | **0.7793** | **0.05** |

**Baseline → Winner improvement.**

| Metric | Baseline LR | Tuned XGBoost | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.9754 | 0.9760 | **+0.06 pp** |
| PR-AUC | 0.6080 | 0.7793 | **+17.13 pp** |
| F1 | 0.0857 | 0.7766 | **+69.09 pp** |
| Precision | 0.0450 | 0.7849 | **+73.99 pp** |
| False positives (test) | 1,803 | 20 | **−98.9%** |

Read the result table: ROC-AUC moved by a hair, but precision went from ~5% to ~78% — that's the difference between paging a human on every 22nd transaction and paging them on actual fraud. PR-AUC made the trade-off visible; ROC-AUC hid it.

---

## 4. Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.9+ | Runtime; CI tests on 3.9, dev on 3.12 |
| **pandas** | latest | Tabular data manipulation, CSV I/O |
| **NumPy** | 2.x | Numeric arrays, vectorized features |
| **scikit-learn** | 1.6+ | Pipelines, preprocessing, baseline LR/RF, metrics, train/test split |
| **XGBoost** | 2.x | Gradient-boosted trees; the production model |
| **LightGBM** | 4.x | Comparison candidate during model selection |
| **Optuna** | 4.x | Bayesian hyperparameter search (30 trials × 5-fold CV) |
| **MLflow** | 3.x | Experiment tracking + model registry; runs land in `mlruns/` |
| **Great Expectations** | 1.x | (Pattern, not direct dep) inspired the homegrown `data/quality.py` gate |
| **Streamlit** | 1.50 | Interactive 4-page portfolio app |
| **Plotly** | 6.x | Charts inside Streamlit (ROC, PR curves, feature importance) |
| **joblib** | (sklearn dep) | Model serialization |
| **pytest** | 8.x / 9.x | Unit + integration tests; fixtures train models in-process |
| **ruff** | latest | Linting + import sorting (config in `ruff.toml`) |
| **Docker + Compose** | — | One-command containerized deploy of the Streamlit app |
| **GitHub Actions** | — | CI matrix: lint job + test job, runs on every push/PR |

---

## 5. Setup & Installation

**Prerequisites.** Python 3.9 or higher, ~500 MB free disk for dependencies, and the raw dataset (gitignored — see below).

```bash
# 1. Clone
git clone https://github.com/MrKhaled007/my-ml-project.git
cd my-ml-project

# 2. Create a venv (Windows PowerShell shown; bash equivalent works)
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate      # macOS/Linux

# 3. Install dependencies + register the src package
pip install -r requirements.txt          # runtime deps (app + tests)
pip install -r requirements-train.txt    # add training deps (mlflow, optuna, lightgbm, great-expectations)
pip install -e .

# 4. Download the raw dataset
#    Save Kaggle's creditcard.csv into ./data/creditcard.csv
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

The repo gitignores `data/` and `models/` because the raw CSV is ~150 MB and the trained pickles are bulky. Tests use small committed fixtures in `tests/fixtures/` so CI works without either.

---

## 6. How to Run

### 6.1 Train the full pipeline

Each step persists its output so you can re-run individual stages without rebuilding from scratch.

```bash
python -m src.data.cleaner               # creditcard.csv → cleaned.csv
python -m src.features.run_features      # cleaned.csv  → features.csv
python -m src.models.baseline            # baseline LR → models/baseline.pkl
python -m src.models.compare             # LR / RF / XGB → models/*.pkl
python -m src.models.tuning              # Optuna → models/production_model.pkl
python -m src.models.build_app_artifacts # → data/predictions.csv + model_results.json
```

MLflow runs land in `mlruns/`. Open the UI with `mlflow ui` and point it at <http://localhost:5000>.

### 6.2 Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens at <http://localhost:8501>. Four pages: **Overview**, **EDA**, **Models**, **Predictions**. The app runs in *demo mode* with synthetic data if `data/predictions.csv` or `models/production_model.pkl` are missing — useful for the live deploy.

### 6.3 Docker

```bash
docker compose up --build
```

Builds a Python 3.9-slim image, installs deps, copies `src/`, `app/`, `data/`, `models/`, and exposes the app on **8501**. `docker-compose.yml` bind-mounts `./data` and `./models` so artifact regeneration shows up live without a rebuild.

### 6.4 Tests

```bash
pytest tests/ -v
```

7 tests across 3 files (data quality gate, feature engineering shape/range/NaN, model load + predict). Total runtime ~1.5 s locally. CI runs the same suite on Python 3.9 plus a `ruff check src/ app/` lint job.

---

## 7. Feature Engineering

Raw inputs: `Time`, `Amount`, 28 PCA components `V1`–`V28`. The PCA components are powerful but anonymous — there's no way to say "fraud spikes on overseas transactions" because that signal is buried inside `V14` somewhere. The engineered features expose the patterns the PCA hides.

### Top 10 features by XGBoost gain

| # | Feature | Importance | Why it works |
|---|---|---:|---|
| 1 | `top_signal_min` | 0.482 | Min across the five top fraud-correlated PCA components (`V17, V14, V12, V10, V16`). Fraud has strongly negative tails on these — taking the min picks up the deepest excursion in the right direction. |
| 2 | `top_signal_magnitude` | 0.218 | L2 norm of the same five components. Catches fraud cases where the signal is spread across components rather than concentrated in one. |
| 3 | `v17_x_v14` | 0.046 | Multiplicative interaction of the two strongest individual signals. Blows up only when **both** are extreme together — a co-occurrence pattern a single split can't represent. |
| 4 | `V4` | 0.016 | Raw PCA component, useful as a residual signal after the engineered aggregates. |
| 5 | `V14` | 0.016 | Strong individual signal even after it's been folded into `top_signal_min` and `v17_x_v14`. |
| 6 | `V8` | 0.013 | Raw PCA. |
| 7 | `amount_log` | 0.012 | `log1p(Amount)` compresses the long right tail (skew ≈ 17, max $25,691). Zero-amount auths stay at zero. |
| 8 | `night_high_amount` | 0.012 | `is_night × amount_log`. Cardholder is asleep AND the charge is unusually large — either alone is common, the combination is not. |
| 9 | `V12` | 0.012 | Raw PCA. |
| 10 | `V19` | 0.010 | Raw PCA. |

### Other engineered features (lower importance but useful for robustness)

- **`is_zero_amount`** — flags $0 pre-authorizations, a classic card-testing pattern (fraudsters validate stolen cards with tiny pre-auths before a real charge).
- **`hour_of_day`** — derived from `Time % 86400 // 3600`. The raw monotonic `Time` column hides circadian fraud patterns from tree splits.
- **`is_night`** — `hour_of_day ∈ [0, 6)`. Card-not-present fraud disproportionately fires overnight.
- **`amount_zscore`** — global z-score of `Amount`, lets a single split answer "is this charge unusually large?".
- **`amount_rolling_mean` / `amount_rolling_std`** — local baseline over a 500-row sorted-by-Time window. Gives the model "what does normal look like *right now*".
- **`amount_dev_from_rolling`** — signed deviation from the local norm. Spikes flag charges anomalous against recent activity, regardless of global scale.
- **`v_extreme_count`** — count of `V*` components with `|z| > 3`. A model-agnostic anomaly score that complements tree splits.
- **`amount_per_v17_magnitude`** — large dollar amount paired with extreme V17. The classic "successful cash-out after testing" pattern.

After `select_features` runs a variance + correlation filter (drop `var < 1% of mean var`, drop one of any pair with `|corr| > 0.95`), the model trains on **37 final features**.

---

## 8. Key Decisions & Lessons

1. **PR-AUC is the right metric, not ROC-AUC.** Optuna tuning only moved ROC-AUC by **+0.06 pp** (0.9754 → 0.9760). On that number alone, the 30-trial search would look like wasted compute. But PR-AUC went from **0.608 → 0.779 (+17.1 pp)** and false positives dropped from 1,803 to 20 on the test set. ROC-AUC saturates at high imbalance because true-negatives dominate the denominator; PR-AUC stays sensitive because precision is a hard test. *Lesson: pick your metric from the business cost function before you tune anything.*

2. **The strongest features are engineered, not raw.** The top two features by gain (`top_signal_min`, `top_signal_magnitude`) together account for **~70% of the model's importance** and are both simple aggregations over five PCA columns. *Lesson: the model isn't doing the feature engineering for you; even with gradient-boosted trees, a 10-line aggregation can outweigh hundreds of trees.*

3. **Failure: the gitignore silently deleted source code.** The `.gitignore` had unanchored `data/` and `models/` patterns intended to ignore the raw data and trained pickles at the repo root. They also matched `src/data/` and `src/models/` — meaning `src/data/loader.py`, `quality.py`, and most of `src/models/` were never committed. Local tests passed because the files existed on disk; CI failed with `ModuleNotFoundError: No module named 'src.data'` and the cause was invisible until I ran `git ls-files src/` and saw the gaps. Fixed by anchoring the patterns to `/data/` and `/models/`. *Lesson: `git ls-files` is the source of truth for what your collaborators (and CI) actually have. "It works on my machine" includes "the files exist on my disk."*

4. **Don't ship pickled models across Python/sklearn versions.** The first version of the test fixture committed a `model_sample.pkl` trained locally on sklearn 1.8.0. CI installed sklearn 1.6.1 and threw `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` because the attribute had been renamed between versions. Replaced with a session-scoped pytest fixture that retrains the model in-memory from the committed feature CSV. *Lesson: pickles are a private contract between two specific library versions; anything that crosses a process boundary (CI, prod, a colleague's laptop) should re-train or use a portable format like ONNX / `Booster.save_model`.*

5. **The Streamlit app needs to render with no artifacts.** When `data/predictions.csv` or `models/*.pkl` are missing (e.g. on a fresh Streamlit Cloud deploy), the app falls back to plausible synthetic numbers and shows a banner explaining the demo mode. *Lesson: portfolio apps fail open in front of recruiters — never blank, never tracebacks.*

---

## 9. File Structure

```
my-ml-project/
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions: lint + test on push/PR
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # 4-page Streamlit UI with demo-mode fallback
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py              # find_csv, load_csv, describe
│   │   ├── cleaner.py             # dropna, dedupe, dtype coercion → cleaned.csv
│   │   └── quality.py             # schema/null/range/target-dist quality gate
│   ├── features/
│   │   ├── engineering.py         # create_features (15 new) + select_features
│   │   └── run_features.py        # CLI wrapper: cleaned.csv → features.csv
│   └── models/
│       ├── baseline.py            # Logistic Regression baseline
│       ├── compare.py             # LR / RF / XGB / LightGBM bake-off
│       ├── tuning.py              # Optuna 30-trial Bayesian search
│       ├── run_training.py        # CLI driver for the full training run
│       └── build_app_artifacts.py # → predictions.csv + model_results.json
├── tests/
│   ├── fixtures/
│   │   ├── build_fixtures.py      # regenerate sample CSVs from cleaned.csv
│   │   ├── cleaned_sample.csv     # 1000-row stratified sample (committed)
│   │   └── features_sample.csv    # create_features output (committed)
│   ├── test_data_quality.py       # quality gate passes/catches broken data
│   ├── test_features.py           # column count, no-NaN, range invariants
│   └── test_model.py              # in-process train + predict_proba checks
├── notebooks/
│   └── eda.ipynb                  # exploratory analysis (read this first)
├── data/                          # gitignored — raw + intermediate artifacts
├── models/                        # gitignored — trained .pkl files
├── conftest.py                    # adds repo root to sys.path for tests
├── Dockerfile                     # Python 3.9-slim → Streamlit on :8501
├── docker-compose.yml             # bind-mounts data/ and models/
├── requirements.txt               # pinned-by-pip top-level deps
├── ruff.toml                      # E + F + I, ignores E701/E501
├── setup.py                       # exposes src/ as an installable package
└── README.md                      # this file
```

---

## License

MIT — see [LICENSE](LICENSE) (add one if you fork this).

## Acknowledgements

Dataset by the [Machine Learning Group at ULB](https://mlg.ulb.ac.be/wordpress/) via Kaggle. The 28 PCA-anonymized features in this dataset are the result of their preprocessing — not mine.
