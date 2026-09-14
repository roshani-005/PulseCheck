"""Survival models and survival visualizations for PulseCheck."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
except ImportError as exc:  # pragma: no cover - gives a clearer local error
    raise ImportError(
        "PulseCheck requires lifelines. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


FEATURES = ["avg_delay_3m", "emi_to_income_ratio", "rolling_delay_trend", "missed_count_3m"]


def loan_level_outcomes(panel: pd.DataFrame) -> pd.DataFrame:
    """Collapse a panel to one record per loan for nonparametric survival curves."""
    return (
        panel.sort_values(["loan_id", "month"])
        .groupby("loan_id", as_index=False)
        .tail(1)[
            [
                "loan_id",
                "borrower_type",
                "emi_to_income_ratio",
                "event",
                "time_to_event",
                "observation_end_reason",
            ]
        ]
        .reset_index(drop=True)
    )


def plot_kaplan_meier(panel: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loans = loan_level_outcomes(panel)
    paths: dict[str, Path] = {}

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.fit(loans["time_to_event"], event_observed=loans["event"], label="All loans")
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set(title="Overall loan survival", xlabel="Months since origination", ylabel="P(no default yet)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    paths["km_overall"] = output_dir / "km_overall.png"
    fig.savefig(paths["km_overall"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for borrower_type, group in loans.groupby("borrower_type"):
        kmf.fit(group["time_to_event"], event_observed=group["event"], label=borrower_type.title())
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set(title="Survival by borrower type", xlabel="Months since origination", ylabel="P(no default yet)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    paths["km_borrower_type"] = output_dir / "km_by_borrower_type.png"
    fig.savefig(paths["km_borrower_type"], dpi=160)
    plt.close(fig)

    loans["emi_tier"] = pd.cut(
        loans["emi_to_income_ratio"],
        bins=[0.0, 0.20, 0.35, np.inf],
        labels=["Low (≤20%)", "Medium (20–35%)", "High (>35%)"],
        include_lowest=True,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for tier, group in loans.groupby("emi_tier", observed=True):
        kmf.fit(group["time_to_event"], event_observed=group["event"], label=str(tier))
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set(title="Survival by EMI-to-income tier", xlabel="Months since origination", ylabel="P(no default yet)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    paths["km_emi_tier"] = output_dir / "km_by_emi_tier.png"
    fig.savefig(paths["km_emi_tier"], dpi=160)
    plt.close(fig)
    return paths


def fit_cox_model(
    train_df: pd.DataFrame,
    output_dir: str | Path,
    features: Sequence[str] = FEATURES,
) -> tuple[CoxPHFitter, pd.DataFrame, dict]:
    """Fit the required Cox PH model and persist hazard-ratio + PH diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["duration_from_landmark", "event", *features]
    fit_df = train_df[cols].dropna().copy()

    cph = CoxPHFitter(penalizer=0.02)
    cph.fit(fit_df, duration_col="duration_from_landmark", event_col="event", show_progress=False)

    summary = cph.summary.reset_index().rename(columns={"covariate": "feature"})
    keep = [
        "feature",
        "coef",
        "exp(coef)",
        "se(coef)",
        "coef lower 95%",
        "coef upper 95%",
        "exp(coef) lower 95%",
        "exp(coef) upper 95%",
        "p",
    ]
    hazard = summary[[c for c in keep if c in summary.columns]].copy()
    hazard.to_csv(output_dir / "hazard_ratios.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ordered = hazard.sort_values("exp(coef)")
    hr = ordered["exp(coef)"].to_numpy()
    low = ordered["exp(coef) lower 95%"].to_numpy()
    high = ordered["exp(coef) upper 95%"].to_numpy()
    xerr = np.vstack([hr - low, high - hr])
    ax.errorbar(hr, ordered["feature"], xerr=xerr, fmt="o", capsize=3)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.set(xlabel="Hazard ratio (95% CI)", title="Cox proportional-hazards effects")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "cox_hazard_ratios.png", dpi=160)
    plt.close(fig)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            cph.check_assumptions(fit_df, p_value_threshold=0.05, show_plots=False)
        except Exception as exc:  # keep the report explicit instead of hiding a diagnostic failure
            print(f"check_assumptions raised {type(exc).__name__}: {exc}")
    check_text = buffer.getvalue().strip()
    (output_dir / "ph_assumption_check.txt").write_text(check_text + "\n", encoding="utf-8")

    ph_test = proportional_hazard_test(cph, fit_df, time_transform="rank")
    ph_table = ph_test.summary.reset_index()
    ph_table = ph_table.rename(columns={ph_table.columns[0]: "feature"})
    ph_table.to_csv(output_dir / "ph_test_statistics.csv", index=False)
    violations = ph_table.loc[ph_table["p"] < 0.05, "feature"].astype(str).tolist()
    diagnostic = {
        "assumption_passed_at_0_05": len(violations) == 0,
        "violating_features": violations,
        "check_assumptions_output": check_text,
    }
    return cph, hazard, diagnostic


def predict_horizon_default_risk(
    cph: CoxPHFitter,
    frame: pd.DataFrame,
    horizon_months: int = 3,
    features: Sequence[str] = FEATURES,
) -> np.ndarray:
    """Predict 1-S(t) at a fixed horizon using the Cox model."""
    sf = cph.predict_survival_function(frame[list(features)], times=[horizon_months])
    return 1.0 - sf.iloc[0].to_numpy(dtype=float)


def horizon_metrics(cph: CoxPHFitter, test_df: pd.DataFrame, horizon_months: int = 3) -> dict:
    y = test_df[f"defaults_next_{horizon_months}m"].to_numpy() if f"defaults_next_{horizon_months}m" in test_df else (
        ((test_df["event"] == 1) & (test_df["duration_from_landmark"] <= horizon_months)).astype(int).to_numpy()
    )
    risk = predict_horizon_default_risk(cph, test_df, horizon_months=horizon_months)
    result = {"brier": float(brier_score_loss(y, risk))}
    result["roc_auc"] = float(roc_auc_score(y, risk)) if len(np.unique(y)) > 1 else None
    result["mean_predicted_risk"] = float(np.mean(risk))
    return result
