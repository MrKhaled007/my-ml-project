"""Portfolio Streamlit app for the credit-card fraud detection project.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"
RESULTS_PATH = DATA_DIR / "model_results.json"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.pkl"
FEATURES_PATH = DATA_DIR / "features.csv"

GITHUB_URL = "https://github.com/MrKhaled007/my-ml-project"

PRIMARY = "#4F8BF9"
ACCENT = "#22C55E"
DANGER = "#EF4444"
MUTED = "#6B7280"
BG_CARD = "#0F172A"

st.set_page_config(
    page_title="Credit Card Fraud Detection — Portfolio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = f"""
<style>
:root {{
    --primary: {PRIMARY};
    --accent: {ACCENT};
    --danger: {DANGER};
    --muted: {MUTED};
}}

/* Header band */
.app-header {{
    background: linear-gradient(120deg, #1E3A8A 0%, #4F8BF9 100%);
    padding: 22px 28px;
    border-radius: 14px;
    color: white;
    margin-bottom: 20px;
    box-shadow: 0 6px 24px rgba(30, 58, 138, 0.25);
}}
.app-header h1 {{
    margin: 0;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.2px;
}}
.app-header p {{
    margin: 4px 0 0;
    font-size: 14px;
    opacity: 0.9;
}}

/* Hero section on Overview page */
.hero {{
    background: radial-gradient(circle at top right, #4F8BF9 0%, #1E3A8A 60%, #0F172A 100%);
    padding: 48px 36px;
    border-radius: 16px;
    color: white;
    margin: 8px 0 24px;
}}
.hero h2 {{
    font-size: 38px;
    font-weight: 800;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}}
.hero p {{
    font-size: 17px;
    margin: 0;
    opacity: 0.92;
    max-width: 760px;
}}

/* Tech badges */
.badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }}
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    background: rgba(79, 139, 249, 0.12);
    color: #4F8BF9;
    border: 1px solid rgba(79, 139, 249, 0.35);
}}
.badge.green  {{ background: rgba(34, 197, 94, 0.12);  color: #22C55E; border-color: rgba(34,197,94,.35); }}
.badge.purple {{ background: rgba(168, 85, 247, 0.12); color: #A855F7; border-color: rgba(168,85,247,.35); }}
.badge.amber  {{ background: rgba(245, 158, 11, 0.12); color: #F59E0B; border-color: rgba(245,158,11,.35); }}
.badge.pink   {{ background: rgba(236, 72, 153, 0.12); color: #EC4899; border-color: rgba(236,72,153,.35); }}

/* Callout boxes */
.callout {{
    padding: 14px 16px;
    border-radius: 10px;
    border-left: 4px solid var(--primary);
    background: rgba(79, 139, 249, 0.08);
    margin: 8px 0;
}}
.callout.warn   {{ border-color: #F59E0B; background: rgba(245, 158, 11, 0.08); }}
.callout.danger {{ border-color: var(--danger); background: rgba(239, 68, 68, 0.08); }}
.callout.ok     {{ border-color: var(--accent); background: rgba(34, 197, 94, 0.08); }}
.callout strong {{ display: block; margin-bottom: 4px; font-size: 14px; }}
.callout span   {{ font-size: 14px; opacity: 0.85; }}

/* Footer */
.footer {{
    margin-top: 48px;
    padding: 18px 0 8px;
    border-top: 1px solid rgba(120, 120, 120, 0.2);
    font-size: 13px;
    color: var(--muted);
    text-align: center;
}}
.footer a {{ color: var(--primary); text-decoration: none; }}

/* Tighten metric cards */
[data-testid="stMetric"] {{
    background: rgba(79, 139, 249, 0.05);
    padding: 14px 16px;
    border-radius: 12px;
    border: 1px solid rgba(120, 120, 120, 0.15);
}}
[data-testid="stMetricLabel"] {{ font-weight: 600; opacity: 0.8; }}

/* Hide Streamlit chrome we don't need */
#MainMenu, footer {{visibility: hidden;}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

# Module-level flags: which artifacts are missing? Each is a single stat call,
# so it's fine to evaluate at import time. Demo fallbacks keep the app runnable
# on a fresh clone (or on Streamlit Cloud before the model files are uploaded).
DEMO_RESULTS = not RESULTS_PATH.exists()
DEMO_PREDICTIONS = not PREDICTIONS_PATH.exists()
DEMO_FEATURES = not FEATURES_PATH.exists()
DEMO_MODEL = not PRODUCTION_MODEL_PATH.exists()
ANY_DEMO = DEMO_RESULTS or DEMO_PREDICTIONS or DEMO_FEATURES or DEMO_MODEL


@st.cache_data(show_spinner=False)
def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return _synth_results()


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return _synth_predictions()


@st.cache_data(show_spinner="Loading data…")
def load_features_sample(n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Read the engineered-features CSV (sampled), or synthesize if missing."""
    if FEATURES_PATH.exists():
        df = pd.read_csv(FEATURES_PATH)
        if len(df) > n:
            # Stratify by Class so the (rare) fraud rows survive sampling.
            pos = df[df["Class"] == 1]
            neg = df[df["Class"] == 0].sample(n=n - len(pos), random_state=seed)
            df = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
        return df
    return _synth_features(n, seed)


@st.cache_resource(show_spinner=False)
def load_production_model():
    if PRODUCTION_MODEL_PATH.exists():
        return joblib.load(PRODUCTION_MODEL_PATH)
    return _DemoModel()


# ---------------------------------------------------------------------------
# Demo data synthesis (used only when real artifacts are missing)
# ---------------------------------------------------------------------------

_DEMO_FEATURE_COLS = [
    "Time", "V1", "V2", "V3", "V4", "V6", "V8", "V9", "V10", "V11",
    "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V21", "V22",
    "V24", "V25", "V26", "Amount", "amount_log", "is_zero_amount",
    "hour_of_day", "is_night", "amount_rolling_mean", "amount_rolling_std",
    "v_extreme_count", "top_signal_magnitude", "top_signal_min",
    "v17_x_v14", "v14_x_v12", "amount_per_v17_magnitude", "night_high_amount",
]


def _cm(tn: int, fp: int, fn: int, tp: int) -> list[list[int]]:
    return [[tn, fp], [fn, tp]]


def _synth_results() -> dict:
    """Plausible-but-illustrative metrics so every page renders without real artifacts."""
    n_rows, n_test = 280_000, 56_000
    n_fraud = 470

    models = {
        "baseline": {
            "accuracy": 0.9680, "precision": 0.045, "recall": 0.895,
            "f1": 0.0857, "roc_auc": 0.9720, "pr_auc": 0.610,
            "confusion_matrix": _cm(54_100, 1_805, 10, 85),
            "label": "Baseline (Logistic Regression)", "score_time_s": 0.05,
        },
        "logistic_regression": {
            "accuracy": 0.9682, "precision": 0.046, "recall": 0.895,
            "f1": 0.0860, "roc_auc": 0.9725, "pr_auc": 0.612,
            "confusion_matrix": _cm(54_115, 1_790, 10, 85),
            "label": "Logistic Regression", "score_time_s": 0.05,
        },
        "random_forest": {
            "accuracy": 0.9994, "precision": 0.957, "recall": 0.705,
            "f1": 0.812, "roc_auc": 0.9290, "pr_auc": 0.807,
            "confusion_matrix": _cm(55_902, 3, 28, 67),
            "label": "Random Forest", "score_time_s": 0.40,
        },
        "xgboost": {
            "accuracy": 0.9996, "precision": 0.936, "recall": 0.768,
            "f1": 0.844, "roc_auc": 0.9695, "pr_auc": 0.794,
            "confusion_matrix": _cm(55_900, 5, 22, 73),
            "label": "XGBoost (default params)", "score_time_s": 0.12,
        },
        "tuned_winner": {
            "accuracy": 0.9994, "precision": 0.785, "recall": 0.768,
            "f1": 0.776, "roc_auc": 0.9760, "pr_auc": 0.780,
            "confusion_matrix": _cm(55_885, 20, 22, 73),
            "label": "XGBoost (Optuna-tuned)  WINNER", "score_time_s": 0.13,
        },
    }

    # Synthetic ROC curve — fast climb then plateau, integrating to ~0.976.
    fpr = np.linspace(0.0, 1.0, 200)
    tpr = np.clip(1 - (1 - fpr) ** 0.045, 0.0, 1.0)
    roc = {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}

    return {
        "_demo": True,
        "dataset": {
            "n_rows_total": n_rows,
            "n_features": len(_DEMO_FEATURE_COLS),
            "n_train": n_rows - n_test,
            "n_test": n_test,
            "fraud_count": n_fraud,
            "fraud_rate": n_fraud / n_rows,
            "imbalance_ratio": round((n_rows - n_fraud) / n_fraud, 1),
        },
        "feature_columns": list(_DEMO_FEATURE_COLS),
        "models": models,
        "winner": "tuned_winner",
        "winner_rationale": [
            "Highest ROC-AUC and PR-AUC on a held-out, stratified test set.",
            "PR-AUC is the right metric here — accuracy is meaningless at 600:1 imbalance.",
            "Optuna search (30 trials, 5-fold CV) over 9 XGBoost hyperparameters.",
            "Cross-validated AUC is stable across folds (low std), so the gain isn't a fluke.",
            "Train time is acceptable; inference is fast enough for a synchronous API.",
        ],
        "improvement_over_baseline_auc_pp": round((0.9760 - 0.9720) * 100, 3),
        "feature_importances_top": {
            "top_signal_min":            0.482,
            "top_signal_magnitude":      0.218,
            "v17_x_v14":                 0.046,
            "V4":                        0.016,
            "V14":                       0.016,
            "v14_x_v12":                 0.014,
            "V17":                       0.013,
            "amount_per_v17_magnitude":  0.012,
            "v_extreme_count":           0.011,
            "V10":                       0.010,
            "V12":                       0.009,
            "Amount":                    0.008,
            "is_night":                  0.007,
            "amount_log":                0.006,
            "hour_of_day":               0.005,
        },
        "roc_curve_winner": roc,
        "top_correlations_with_target": {
            "v14_x_v12":                 0.585,
            "v17_x_v14":                 0.543,
            "top_signal_magnitude":      0.441,
            "V17":                      -0.326,
            "V14":                      -0.302,
            "V12":                      -0.262,
            "V10":                      -0.218,
            "V16":                      -0.198,
            "top_signal_min":           -0.612,
            "v_extreme_count":           0.298,
            "V11":                       0.155,
            "V4":                        0.133,
            "amount_per_v17_magnitude":  0.087,
            "is_night":                  0.045,
            "Amount":                    0.005,
        },
        "best_params": {
            "n_estimators":     598,
            "max_depth":        8,
            "learning_rate":    0.0233,
            "subsample":        0.985,
            "colsample_bytree": 0.824,
            "min_child_weight": 7,
            "gamma":            4.84,
            "reg_alpha":        0.053,
            "reg_lambda":       0.368,
        },
    }


def _synth_predictions(n: int = 5_000, seed: int = 42) -> pd.DataFrame:
    """Bimodal probabilities — most rows near 0, a long tail near 1 for the synthetic frauds."""
    rng = np.random.default_rng(seed)
    n_fraud = max(8, int(round(n * 0.0017)))
    n_legit = n - n_fraud
    y_true = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    # Legit: probas hugging zero with a small false-alarm tail.
    legit_proba = np.clip(rng.beta(0.4, 35.0, size=n_legit), 0, 1)
    # Fraud: most probas high, a few hard cases low.
    fraud_proba = np.clip(rng.beta(8.0, 1.5, size=n_fraud), 0, 1)
    proba_winner = np.concatenate([legit_proba, fraud_proba])

    # Other models track the winner with added noise; baseline is the noisiest.
    def jitter(base, scale):
        return np.clip(base + rng.normal(0, scale, size=base.shape), 0, 1)

    df = pd.DataFrame({
        "y_true":             y_true,
        "Amount":             np.exp(rng.normal(3.0, 1.2, size=n)).round(2),
        "amount_log":         np.log1p(np.exp(rng.normal(3.0, 1.2, size=n))).round(4),
        "hour_of_day":        rng.integers(0, 24, size=n),
        "is_night":           rng.integers(0, 2, size=n),
        "proba_baseline":            jitter(proba_winner, 0.18),
        "proba_logistic_regression": jitter(proba_winner, 0.18),
        "proba_random_forest":       jitter(proba_winner, 0.10),
        "proba_xgboost":             jitter(proba_winner, 0.07),
        "proba_tuned_winner":        proba_winner,
    })
    df["y_pred_winner"] = (df["proba_tuned_winner"] >= 0.5).astype(int)
    # Shuffle so legit/fraud aren't blocked at the bottom.
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _synth_features(n: int = 25_000, seed: int = 42) -> pd.DataFrame:
    """Synthesize an engineered-features dataset with the same schema and class skew."""
    rng = np.random.default_rng(seed)
    n_fraud = max(40, int(round(n * 0.0017)))
    n_legit = n - n_fraud

    def normal(loc, scale, size):
        return rng.normal(loc, scale, size=size)

    # PCA-style standard normal for V columns; fraud rows get heavy negative tails on top signals.
    v_cols = [c for c in _DEMO_FEATURE_COLS if c.startswith("V") and c[1:].isdigit()]
    legit_v = {c: normal(0, 1, n_legit) for c in v_cols}
    fraud_v = {c: normal(0, 1, n_fraud) for c in v_cols}
    for c, mu in [("V17", -6), ("V14", -5.5), ("V12", -5), ("V10", -4.5), ("V16", -4)]:
        if c in fraud_v:
            fraud_v[c] = normal(mu, 1.5, n_fraud)
    if "V4" in fraud_v:
        fraud_v["V4"] = normal(3.5, 1.5, n_fraud)

    def assemble(side_v: dict, n_side: int, is_fraud: int) -> pd.DataFrame:
        amount = np.exp(normal(3.0 + 0.6 * is_fraud, 1.1, n_side)).clip(0, 25_000)
        hour = rng.integers(0, 24, size=n_side)
        is_night = ((hour < 6) | (hour >= 22)).astype(int)
        # Skew night more for fraud
        if is_fraud:
            night_flip = rng.random(n_side) < 0.4
            is_night[night_flip] = 1
        amount_log = np.log1p(amount)
        is_zero_amount = (amount < 1).astype(int)
        amount_rolling_mean = np.full(n_side, 88.0) + normal(0, 12, n_side)
        amount_rolling_std = np.full(n_side, 250.0) + normal(0, 30, n_side)
        top_block = np.column_stack([side_v[c] for c in ("V17", "V14", "V12", "V10", "V16")])
        top_signal_magnitude = np.sqrt((top_block ** 2).sum(axis=1))
        top_signal_min = top_block.min(axis=1)
        v_extreme_count = (np.abs(top_block) > 3).sum(axis=1)
        v17_x_v14 = side_v["V17"] * side_v["V14"]
        v14_x_v12 = side_v["V14"] * side_v["V12"]
        amount_per_v17_magnitude = amount_log / (np.abs(side_v["V17"]) + 1e-3)
        night_high_amount = is_night * amount_log

        cols = {
            "Time": rng.integers(0, 172_800, size=n_side),
            **side_v,
            "Amount": amount,
            "amount_log": amount_log,
            "is_zero_amount": is_zero_amount,
            "hour_of_day": hour,
            "is_night": is_night,
            "amount_rolling_mean": amount_rolling_mean,
            "amount_rolling_std": amount_rolling_std,
            "v_extreme_count": v_extreme_count,
            "top_signal_magnitude": top_signal_magnitude,
            "top_signal_min": top_signal_min,
            "v17_x_v14": v17_x_v14,
            "v14_x_v12": v14_x_v12,
            "amount_per_v17_magnitude": amount_per_v17_magnitude,
            "night_high_amount": night_high_amount,
            "Class": np.full(n_side, is_fraud, dtype=int),
        }
        # Order columns as defined and ensure the Class is last.
        ordered = [c for c in _DEMO_FEATURE_COLS if c in cols] + ["Class"]
        return pd.DataFrame({c: cols[c] for c in ordered})

    legit_df = assemble(legit_v, n_legit, is_fraud=0)
    fraud_df = assemble(fraud_v, n_fraud, is_fraud=1)
    return pd.concat([legit_df, fraud_df]).sample(frac=1, random_state=seed).reset_index(drop=True)


class _DemoModel:
    """Heuristic stand-in for the production XGBoost model.

    Used only when production_model.pkl is missing. Direction of response
    matches the real model (top-signal extremes + night + zero-amount push
    fraud probability up); magnitude is calibrated to feel responsive.
    """

    feature_importances_ = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        signal     = -X["top_signal_min"].clip(-15, 5).to_numpy() * 0.40
        magnitude  =  X["top_signal_magnitude"].clip(0, 30).to_numpy() * 0.08
        night      =  X["is_night"].to_numpy() * 0.6
        zero_amt   =  X["is_zero_amount"].to_numpy() * 0.5
        extreme    =  X["v_extreme_count"].to_numpy() * 0.3
        amt_anomly =  X["amount_log"].clip(0, 10).to_numpy() * 0.05

        logit = -2.0 + signal + magnitude + night + zero_amt + extreme + amt_anomly
        proba = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1 - proba, proba])


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <h1>🛡️ Credit Card Fraud Detection</h1>
            <p>End-to-end ML portfolio project — EDA → feature engineering → modeling → tuning → MLflow tracking</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
            Built by <strong>Khaled Alam</strong> ·
            <a href="{GITHUB_URL}" target="_blank">View source on GitHub</a> ·
            Powered by Streamlit, scikit-learn, XGBoost &amp; Optuna
        </div>
        """,
        unsafe_allow_html=True,
    )


def demo_banner() -> None:
    """Show a non-intrusive notice if any artifact is being faked."""
    if not ANY_DEMO:
        return
    missing = []
    if DEMO_RESULTS:     missing.append("`data/model_results.json`")
    if DEMO_PREDICTIONS: missing.append("`data/predictions.csv`")
    if DEMO_FEATURES:    missing.append("`data/features.csv`")
    if DEMO_MODEL:       missing.append("`models/production_model.pkl`")
    st.info(
        "**Demo mode** — illustrative synthetic data is being shown because "
        f"the following artifact{'s are' if len(missing) > 1 else ' is'} missing: "
        + ", ".join(missing)
        + ".  \nTo see real model outputs, run `python -m src.models.build_app_artifacts` "
        "after training the models.",
        icon="🧪",
    )


def callout(kind: str, title: str, body: str) -> None:
    cls = {"info": "", "warn": "warn", "ok": "ok", "danger": "danger"}.get(kind, "")
    st.markdown(
        f'<div class="callout {cls}"><strong>{title}</strong><span>{body}</span></div>',
        unsafe_allow_html=True,
    )


def badges(items: list[tuple[str, str]]) -> None:
    chips = " ".join(f'<span class="badge {color}">{name}</span>' for name, color in items)
    st.markdown(f'<div class="badges">{chips}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 1 — Project Overview
# ---------------------------------------------------------------------------

def page_overview(results: dict) -> None:
    ds = results["dataset"]
    winner = results["models"][results["winner"]]
    baseline = results["models"]["baseline"]

    st.markdown(
        """
        <div class="hero">
            <h2>Catching the 0.17%</h2>
            <p>A production-grade machine-learning pipeline that flags fraudulent credit-card transactions
            in a dataset where only 1 in every 600 transactions is actually fraud — a setting where naive
            accuracy is meaningless and recall is everything.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What this project does")
    st.write(
        "This project builds a complete fraud-detection pipeline from raw anonymized "
        "transaction data: cleaning, engineering 14 domain-aware features, comparing four "
        "model families on stratified cross-validation, and tuning the winner with Optuna. "
        "Every run is tracked in MLflow, and the final model is served through this app for "
        "live experimentation."
    )

    st.markdown("### Key metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", f"{ds['n_rows_total']:,}")
    c2.metric("Engineered features", f"{ds['n_features']}", delta="from 31 raw columns")
    c3.metric(
        "Production model ROC-AUC",
        f"{winner['roc_auc']:.4f}",
        delta=f"{(winner['roc_auc'] - baseline['roc_auc']) * 100:+.2f} pp vs baseline",
    )
    c4.metric(
        "Production model PR-AUC",
        f"{winner['pr_auc']:.4f}",
        delta=f"{(winner['pr_auc'] - baseline['pr_auc']) * 100:+.2f} pp vs baseline",
        help="PR-AUC matters more than ROC-AUC at 599:1 imbalance.",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Class imbalance", f"{ds['imbalance_ratio']:,.0f} : 1", help="legitimate : fraud")
    c6.metric("Fraud cases caught", f"{int(winner['recall'] * winner['confusion_matrix'][1][0] + winner['confusion_matrix'][1][1])}",
              delta=f"recall = {winner['recall']:.1%}")
    c7.metric("Precision @ 0.5", f"{winner['precision']:.1%}",
              help="Of transactions we flag as fraud, what % actually are.")
    c8.metric("F1 score", f"{winner['f1']:.4f}",
              delta=f"{(winner['f1'] - baseline['f1']):+.4f} vs baseline")

    st.markdown("### Tech stack")
    badges([
        ("Python 3.11", ""),
        ("pandas", ""),
        ("scikit-learn", "green"),
        ("XGBoost", "amber"),
        ("Optuna", "purple"),
        ("MLflow", "purple"),
        ("Plotly", "pink"),
        ("Streamlit", ""),
        ("FastAPI", "green"),
        ("pytest", "amber"),
        ("Great Expectations", "pink"),
    ])

    st.markdown("### Project highlights")
    h1, h2, h3 = st.columns(3)
    with h1:
        callout("ok", "Honest evaluation",
                "Stratified train/test split, 5-fold CV, and PR-AUC as the headline metric — "
                "no inflated accuracy claims on a 599:1 imbalanced problem.")
    with h2:
        callout("info", "Domain-aware features",
                "14 engineered features encoding card-testing patterns, circadian fraud signals, "
                "and PCA-component interactions that single-feature splits miss.")
    with h3:
        callout("warn", "Reproducible pipeline",
                "Every model run logged to MLflow with params, metrics, and artifacts. "
                "Best hyperparameters persisted as JSON, not hardcoded.")


# ---------------------------------------------------------------------------
# PAGE 2 — Explore the Data
# ---------------------------------------------------------------------------

def page_explore(results: dict) -> None:
    st.subheader("Explore the Data")
    st.caption(
        "Interactive view of the engineered feature dataset. "
        "Rows are stratified-sampled to keep the UI responsive — fraud cases are fully preserved."
    )

    df = load_features_sample()
    feature_cols = [c for c in df.columns if c != "Class"]

    # === Target distribution ===
    st.markdown("#### Target variable distribution")
    target_counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    cA, cB = st.columns([2, 1])
    with cA:
        fig = px.bar(
            x=target_counts.index, y=target_counts.values,
            color=target_counts.index,
            color_discrete_map={"Legitimate": PRIMARY, "Fraud": DANGER},
            labels={"x": "Class", "y": "Transactions"},
            log_y=True,
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with cB:
        callout("danger", "Severe class imbalance",
                f"{target_counts.get('Fraud', 0):,} fraud vs {target_counts.get('Legitimate', 0):,} "
                f"legitimate in this sample. Log scale used so the fraud bar is even visible.")

    st.divider()

    # === Feature explorer ===
    st.markdown("#### Feature distributions — split by class")
    pick = st.selectbox(
        "Pick a feature to inspect",
        options=feature_cols,
        index=feature_cols.index("Amount") if "Amount" in feature_cols else 0,
    )
    use_log = st.checkbox("Log-scale x-axis (helpful for skewed features like Amount)", value=(pick == "Amount"))

    plot_df = df[[pick, "Class"]].copy()
    plot_df["Class"] = plot_df["Class"].map({0: "Legitimate", 1: "Fraud"})
    if use_log and (plot_df[pick] > 0).all():
        plot_df[pick] = np.log1p(plot_df[pick])
        x_label = f"log1p({pick})"
    else:
        x_label = pick

    fig = px.histogram(
        plot_df, x=pick, color="Class", nbins=60, opacity=0.75,
        color_discrete_map={"Legitimate": PRIMARY, "Fraud": DANGER},
        marginal="box", barmode="overlay",
        labels={pick: x_label},
        histnorm="probability density",
    )
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    st.divider()

    # === Correlation heatmap ===
    st.markdown("#### Correlation heatmap")
    default_picks = [c for c in ["V14", "V12", "V10", "V17", "V16", "V4",
                                 "Amount", "amount_log", "top_signal_min",
                                 "top_signal_magnitude", "v17_x_v14", "v14_x_v12"]
                     if c in feature_cols]
    chosen = st.multiselect(
        "Choose features (8–14 looks best)",
        options=feature_cols,
        default=default_picks,
    )
    if len(chosen) >= 2:
        corr = df[chosen + ["Class"]].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pick at least two features.")

    st.divider()

    # === EDA findings ===
    st.markdown("#### Key findings from EDA")
    f1, f2 = st.columns(2)
    with f1:
        callout("warn", "599 : 1 imbalance",
                "Only 473 fraud cases in 283,726 transactions. Accuracy is meaningless — "
                "a 'predict legitimate' baseline scores 99.83%. Use PR-AUC, recall, and confusion matrices.")
        callout("info", "Amount is heavily right-skewed",
                "Skew ≈ 17 (median $22, max $25,691). A log1p transform was applied "
                "before scaling to keep linear models from being dominated by the long tail.")
    with f2:
        callout("ok", "Strongest fraud signals: V17 / V14 / V12 / V10 / V16",
                "These five PCA components carry |corr| with Class from 0.31 down to 0.19. "
                "The other 23 V-features are near-zero correlated with the target.")
        callout("info", "PCA components are mutually orthogonal",
                "Strongest inter-feature correlation is V2–Amount at 0.53. "
                "Multicollinearity is not an issue, so feature selection focused on near-constants and engineered redundancy.")


# ---------------------------------------------------------------------------
# PAGE 3 — Model Results
# ---------------------------------------------------------------------------

def model_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for key, m in results["models"].items():
        rows.append({
            "Model": m["label"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Accuracy": m["accuracy"],
        })
    return pd.DataFrame(rows)


def page_results(results: dict) -> None:
    st.subheader("Model Results")
    st.caption("All metrics computed on a stratified, held-out test set (20% of data, 56,746 transactions).")

    # === Comparison table ===
    st.markdown("#### Model comparison")
    table = model_comparison_table(results)
    st.dataframe(
        table.style
            .format({c: "{:.4f}" for c in table.columns if c != "Model"})
            .bar(subset=["ROC-AUC", "PR-AUC", "F1"], color="#4F8BF9", vmin=0.0, vmax=1.0),
        width="stretch",
        hide_index=True,
    )

    # === Why the winner ===
    st.markdown("#### Why this model won — and why I trust it")
    cL, cR = st.columns([3, 2])
    with cL:
        for point in results["winner_rationale"]:
            st.markdown(f"- {point}")
        st.caption(
            "Note: the AUC gain over a strong logistic baseline is small in absolute terms "
            "(~0.06 pp) — but PR-AUC, F1, and precision all jump dramatically. That's the "
            "right read at this imbalance: ROC-AUC is forgiving, PR-AUC is not."
        )
    with cR:
        winner = results["models"][results["winner"]]
        baseline = results["models"]["baseline"]
        st.metric("ROC-AUC (winner)", f"{winner['roc_auc']:.4f}",
                  delta=f"{(winner['roc_auc'] - baseline['roc_auc']) * 100:+.3f} pp")
        st.metric("PR-AUC (winner)", f"{winner['pr_auc']:.4f}",
                  delta=f"{(winner['pr_auc'] - baseline['pr_auc']) * 100:+.2f} pp")
        st.metric("Precision (winner)", f"{winner['precision']:.3f}",
                  delta=f"{(winner['precision'] - baseline['precision']) * 100:+.1f} pp")

    st.divider()

    # === Feature importance ===
    st.markdown("#### Feature importance — top 15")
    fi = results["feature_importances_top"]
    fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())}).head(15)
    fig = px.bar(
        fi_df.iloc[::-1], x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False,
                      yaxis_title="", xaxis_title="Gain importance")
    st.plotly_chart(fig, width="stretch")
    callout("info", "Engineered features dominate",
            "<code>top_signal_min</code> and <code>top_signal_magnitude</code> together account for ~70% "
            "of total importance — a strong validation that the EDA-driven feature engineering paid off.")

    st.divider()

    # === Confusion matrix + ROC ===
    st.markdown("#### Confusion matrix & ROC curve")
    cm_col, roc_col = st.columns(2)
    with cm_col:
        cm = results["models"][results["winner"]]["confusion_matrix"]
        cm_arr = np.array(cm)
        labels = ["Legitimate", "Fraud"]
        z_text = [[f"{v:,}" for v in row] for row in cm_arr]
        fig = go.Figure(data=go.Heatmap(
            z=cm_arr, x=labels, y=labels,
            text=z_text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(
            height=380, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Predicted", yaxis_title="Actual",
            title="Winner — confusion matrix (threshold = 0.5)",
        )
        st.plotly_chart(fig, width="stretch")
    with roc_col:
        roc = results["roc_curve_winner"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 line=dict(color=PRIMARY, width=3),
                                 name=f"AUC = {results['models'][results['winner']]['roc_auc']:.4f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(color=MUTED, dash="dash"), name="Random"))
        fig.update_layout(
            height=380, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            title="Winner — ROC curve",
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # === Try it yourself ===
    st.markdown("#### Try it yourself — live prediction")
    st.caption(
        "Adjust the inputs below and the production XGBoost model will score the transaction in real time. "
        "Defaults are set to a typical legitimate transaction; use the presets to start from a known fraud pattern."
    )
    interactive_predictor(results)


# ---------------------------------------------------------------------------
# Live predictor (Page 3)
# ---------------------------------------------------------------------------

PRESETS = {
    "Typical legitimate transaction": {
        "Amount": 60.0, "hour_of_day": 14, "is_night": 0,
        "V17": 0.0, "V14": 0.0, "V12": 0.0, "V10": 0.0, "V16": 0.0, "V4": 0.0,
    },
    "Suspicious (top-signal extremes)": {
        "Amount": 1500.0, "hour_of_day": 3, "is_night": 1,
        "V17": -7.0, "V14": -6.0, "V12": -5.5, "V10": -5.0, "V16": -4.0, "V4": 4.5,
    },
    "Card-testing (zero-amount probe)": {
        "Amount": 0.0, "hour_of_day": 4, "is_night": 1,
        "V17": -3.0, "V14": -3.5, "V12": -3.0, "V10": -2.5, "V16": -2.0, "V4": 2.0,
    },
}


def build_input_row(feature_columns: list[str], inputs: dict) -> pd.DataFrame:
    """Construct a single-row DataFrame matching the model's expected schema.

    User-controlled fields override defaults; engineered features are derived
    on the fly so the user never has to set 14 of them by hand.
    """
    amount = float(inputs["Amount"])
    hour = int(inputs["hour_of_day"])
    is_night = int(inputs["is_night"])
    v17, v14, v12, v10, v16, v4 = (float(inputs[k]) for k in ("V17", "V14", "V12", "V10", "V16", "V4"))

    # Approximate the engineered features from the training notebook.
    amount_log = float(np.log1p(amount))
    is_zero_amount = int(amount < 1.0)
    amount_rolling_mean = 88.0  # global mean is roughly $88; close enough for a demo
    amount_rolling_std = 250.0
    top_signal_magnitude = float(np.sqrt(v17**2 + v14**2 + v12**2 + v10**2 + v16**2))
    top_signal_min = float(min(v17, v14, v12, v10, v16))
    v_extreme_count = int(sum(abs(v) > 3 for v in (v17, v14, v12, v10, v16, v4)))
    v17_x_v14 = v17 * v14
    v14_x_v12 = v14 * v12
    amount_per_v17_magnitude = amount_log / (abs(v17) + 1e-3)
    night_high_amount = is_night * amount_log

    derived = {
        "Time": hour * 3600,
        "Amount": amount,
        "amount_log": amount_log,
        "is_zero_amount": is_zero_amount,
        "hour_of_day": hour,
        "is_night": is_night,
        "amount_rolling_mean": amount_rolling_mean,
        "amount_rolling_std": amount_rolling_std,
        "v_extreme_count": v_extreme_count,
        "top_signal_magnitude": top_signal_magnitude,
        "top_signal_min": top_signal_min,
        "v17_x_v14": v17_x_v14,
        "v14_x_v12": v14_x_v12,
        "amount_per_v17_magnitude": amount_per_v17_magnitude,
        "night_high_amount": night_high_amount,
        "V17": v17, "V14": v14, "V12": v12, "V10": v10, "V16": v16, "V4": v4,
    }

    row = {col: derived.get(col, 0.0) for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def interactive_predictor(results: dict) -> None:
    feature_columns = results["feature_columns"]
    model = load_production_model()

    preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Transaction amount ($)", min_value=0.0, max_value=25_000.0,
                                 value=float(preset["Amount"]), step=10.0)
        hour = st.slider("Hour of day", 0, 23, value=int(preset["hour_of_day"]))
        is_night = st.toggle("Night-time transaction (00:00–06:00)", value=bool(preset["is_night"]))
    with c2:
        v17 = st.slider("V17 (top fraud signal)", -30.0, 10.0, value=float(preset["V17"]), step=0.1)
        v14 = st.slider("V14 (2nd top signal)",   -20.0, 10.0, value=float(preset["V14"]), step=0.1)
        v12 = st.slider("V12 (3rd top signal)",   -20.0, 10.0, value=float(preset["V12"]), step=0.1)
    with c3:
        v10 = st.slider("V10", -25.0, 10.0, value=float(preset["V10"]), step=0.1)
        v16 = st.slider("V16", -15.0, 10.0, value=float(preset["V16"]), step=0.1)
        v4  = st.slider("V4",  -10.0, 15.0, value=float(preset["V4"]),  step=0.1)

    inputs = {"Amount": amount, "hour_of_day": hour, "is_night": int(is_night),
              "V17": v17, "V14": v14, "V12": v12, "V10": v10, "V16": v16, "V4": v4}

    X = build_input_row(feature_columns, inputs)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    st.markdown("##### Model output")
    g1, g2 = st.columns([1, 2])
    with g1:
        verdict = "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
        delta_color = "inverse" if pred == 1 else "normal"
        st.metric("Verdict (threshold 0.5)", verdict, delta=f"P(fraud) = {proba:.3%}",
                  delta_color=delta_color)
    with g2:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Fraud probability", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": DANGER if proba >= 0.5 else ACCENT},
                "steps": [
                    {"range": [0, 25],  "color": "#ECFDF5"},
                    {"range": [25, 50], "color": "#FEF3C7"},
                    {"range": [50, 75], "color": "#FED7AA"},
                    {"range": [75, 100], "color": "#FEE2E2"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.8, "value": 50},
            },
        ))
        gauge.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(gauge, width="stretch")

    with st.expander("See the exact feature vector sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}), width="stretch")


# ---------------------------------------------------------------------------
# PAGE 4 — How I Built This
# ---------------------------------------------------------------------------

ARCHITECTURE_DOT = """
digraph G {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11];
    edge [fontname="Inter", fontsize=10, color="#6B7280"];

    raw   [label="Raw CSV\\n(creditcard.csv)", fillcolor="#E0E7FF"];
    clean [label="Cleaning\\nsrc/data/cleaner.py", fillcolor="#DBEAFE"];
    feat  [label="Feature Engineering\\nsrc/features/engineering.py\\n(+14 features, selection)", fillcolor="#BFDBFE"];
    train [label="Train/Test Split\\nstratified 80/20", fillcolor="#93C5FD"];

    base  [label="Baseline\\nLogistic Regression", fillcolor="#FEF3C7"];
    cmp   [label="Model Comparison\\nLR / RF / XGB\\n(5-fold CV)", fillcolor="#FDE68A"];
    tune  [label="Optuna Tuning\\n30 trials × 5-fold CV", fillcolor="#FCD34D"];

    mlf   [label="MLflow Tracking\\nparams · metrics · artifacts", fillcolor="#DDD6FE"];
    prod  [label="production_model.pkl", fillcolor="#A7F3D0"];
    app   [label="Streamlit App\\n(this site)", fillcolor="#6EE7B7"];

    raw -> clean -> feat -> train;
    train -> base -> mlf;
    train -> cmp -> tune -> mlf;
    tune -> prod -> app;
    feat -> app [style=dashed, label="explore"];
}
"""

TIMELINE = [
    ("Day 0", "Project scaffolding",
     "Repo, venv, src/ package layout, requirements pinned, MLflow + Streamlit baselines installed."),
    ("Day 1", "Data ingestion & cleaning",
     "Loaded the Kaggle credit-card dataset (284k rows). Wrote a deterministic cleaner with quality checks. "
     "Result: cleaned.csv, no missing values, schema validated."),
    ("Day 2–3", "Exploratory data analysis",
     "Full EDA notebook — distributions, correlations, time-of-day patterns, target imbalance. "
     "Documented findings in README; identified V17/V14/V12/V10/V16 as top signals."),
    ("Day 4", "Feature engineering & selection",
     "Engineered 14 domain-aware features (log Amount, is_night, rolling stats, PCA interactions, "
     "extreme-count anomaly score). Variance + correlation filter to prune redundancy."),
    ("Day 5", "Baseline + model comparison",
     "Logistic Regression baseline; then RF and XGBoost on stratified 5-fold CV (ROC-AUC). "
     "XGBoost wins on PR-AUC and F1 with minimal AUC trade-off."),
    ("Day 6", "Hyperparameter tuning",
     "Optuna study (30 TPE trials) over 9 XGBoost params, 5-fold stratified CV. "
     "Best params persisted to JSON, final model retrained and saved."),
    ("Day 7", "MLflow + portfolio app",
     "Wired baseline + tuned runs into MLflow with full param/metric/artifact logging. "
     "Built this Streamlit app for public showcase."),
]

LESSONS = [
    ("Pick the right metric first.",
     "ROC-AUC barely moved between baseline and tuned model (~0.06 pp). PR-AUC jumped 17 pp. "
     "If I'd reported only AUC I'd have concluded the tuning was worthless — the opposite is true."),
    ("Engineered features can outshine PCA components.",
     "<code>top_signal_min</code> and <code>top_signal_magnitude</code> dominate XGBoost feature importance, "
     "even though they're derived from V10/V12/V14/V16/V17. Encoding domain knowledge as features still beats "
     "letting the model rediscover it."),
    ("Class-weighting beats SMOTE here.",
     "I tried both; <code>scale_pos_weight</code> on XGBoost gave equal recall with cleaner precision. "
     "SMOTE's synthetic minority points slightly hurt PR-AUC on the held-out set."),
    ("MLflow saved hours.",
     "Once tracking was wired up, comparing runs took seconds instead of grepping notebooks. "
     "Worth the 30 minutes of setup on day 1, not day 7."),
]


def page_built(results: dict) -> None:
    st.subheader("How I Built This")

    st.markdown("#### Architecture")
    st.graphviz_chart(ARCHITECTURE_DOT, width="stretch")

    st.divider()
    st.markdown("#### Build timeline")
    for day, title, body in TIMELINE:
        with st.expander(f"**{day}** — {title}", expanded=(day in {"Day 0", "Day 7"})):
            st.write(body)

    st.divider()
    st.markdown("#### Key decisions & lessons learned")
    for i, (head, body) in enumerate(LESSONS):
        col = i % 2
        if col == 0:
            cols = st.columns(2)
        with cols[col]:
            callout("info" if i % 2 == 0 else "ok", head, body)

    st.divider()
    st.markdown("#### Best hyperparameters (Optuna winner)")
    bp = results.get("best_params", {})
    if bp:
        bp_df = pd.DataFrame({"param": list(bp.keys()), "value": list(bp.values())})
        st.dataframe(bp_df, width="stretch", hide_index=True)

    st.divider()
    st.markdown(
        f"""
        #### Source code
        Everything — pipeline scripts, notebooks, MLflow setup, this app — lives in one repo:

        👉 **[github.com/MrKhaled007/my-ml-project]({GITHUB_URL})**
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAGES = {
    "🏠  Project Overview":   page_overview,
    "🔍  Explore the Data":   page_explore,
    "📊  Model Results":      page_results,
    "🛠  How I Built This":   page_built,
}


def main() -> None:
    results = load_results()

    with st.sidebar:
        st.markdown("### 🛡️ Fraud Detection")
        st.caption("Portfolio showcase")
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.markdown("##### Project at a glance")
        ds = results["dataset"]
        st.markdown(
            f"- **{ds['n_rows_total']:,}** transactions  \n"
            f"- **{ds['n_features']}** features  \n"
            f"- **{ds['imbalance_ratio']:,.0f} : 1** imbalance  \n"
            f"- **{len(results['models'])}** models compared"
        )
        st.divider()
        st.caption("Built with Streamlit")
        st.markdown(f"[GitHub repo →]({GITHUB_URL})")

    header()
    demo_banner()
    PAGES[page](results)
    footer()


if __name__ == "__main__":
    main()
else:
    main()
