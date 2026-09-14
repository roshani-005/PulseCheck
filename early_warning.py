"""Dynamic warning rules, backtesting, and illustrative ROI for PulseCheck."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

from survival_model import FEATURES, predict_horizon_default_risk


def choose_alert_threshold(y_true: np.ndarray, risk: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    """Choose a train-only threshold maximizing F-beta (recall weighted when beta > 1)."""
    candidates = np.unique(np.quantile(risk, np.linspace(0.05, 0.95, 91)))
    best_threshold, best_score = float(np.median(risk)), -1.0
    for threshold in candidates:
        pred = (risk >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def score_monitoring_snapshots(cph, snapshots: pd.DataFrame, horizon_months: int = 3) -> pd.DataFrame:
    scored = snapshots.copy()
    scored["predicted_default_risk_3m"] = predict_horizon_default_risk(
        cph, scored, horizon_months=horizon_months, features=FEATURES
    )
    return scored


def backtest_early_warning(
    scored_snapshots: pd.DataFrame,
    threshold: float,
    evaluation_loan_ids: set[str] | list[str],
) -> tuple[pd.DataFrame, dict]:
    """Measure first pre-default flag lead time on a held-out set of loans."""
    ids = set(evaluation_loan_ids)
    s = scored_snapshots[scored_snapshots["loan_id"].isin(ids)].copy()
    s["flag"] = s["predicted_default_risk_3m"] >= threshold

    records: list[dict] = []
    for loan_id, g in s.groupby("loan_id"):
        g = g.sort_values("month")
        event = int(g["event"].iloc[0])
        default_month = int(g["time_to_event"].iloc[0]) if event else None
        flagged = g[g["flag"]]
        first_flag_month = int(flagged["month"].iloc[0]) if not flagged.empty else None
        predefault = flagged[flagged["month"] < default_month] if event else flagged
        first_predefault_flag = int(predefault["month"].iloc[0]) if not predefault.empty else None
        lead_time = (default_month - first_predefault_flag) if event and first_predefault_flag is not None else None
        records.append(
            {
                "loan_id": loan_id,
                "event": event,
                "default_month": default_month,
                "ever_flagged": int(first_flag_month is not None),
                "first_flag_month": first_flag_month,
                "captured_before_default": int(event and first_predefault_flag is not None),
                "first_predefault_flag_month": first_predefault_flag,
                "lead_time_months": lead_time,
            }
        )

    results = pd.DataFrame(records)
    defaults = results[results["event"] == 1]
    captured = defaults[defaults["captured_before_default"] == 1]
    nondefaults = results[results["event"] == 0]
    summary = {
        "threshold": float(threshold),
        "evaluation_loans": int(len(results)),
        "defaulted_loans": int(len(defaults)),
        "captured_before_default": int(len(captured)),
        "capture_rate": float(len(captured) / len(defaults)) if len(defaults) else None,
        "mean_lead_months": float(captured["lead_time_months"].mean()) if len(captured) else None,
        "median_lead_months": float(captured["lead_time_months"].median()) if len(captured) else None,
        "flagged_loans": int(results["ever_flagged"].sum()),
        "nondefault_flag_rate": float(nondefaults["ever_flagged"].mean()) if len(nondefaults) else None,
        "reactive_baseline_lead_months": 0.0,
    }
    return results, summary


def estimate_roi(
    backtest_results: pd.DataFrame,
    outreach_cost_inr: float = 250.0,
    full_default_recovery_cost_inr: float = 12_000.0,
    proactive_cure_rate: float = 0.25,
) -> dict:
    """Illustrative expected-value ROI. Assumptions are intentionally explicit."""
    flagged_loans = int(backtest_results["ever_flagged"].sum())
    captured_defaults = int(backtest_results["captured_before_default"].sum())
    defaulted_loans = int(backtest_results["event"].sum())

    expected_prevented_defaults = captured_defaults * proactive_cure_rate
    outreach_spend = flagged_loans * outreach_cost_inr
    avoided_recovery_cost = expected_prevented_defaults * full_default_recovery_cost_inr
    net_savings = avoided_recovery_cost - outreach_spend
    roi = net_savings / outreach_spend if outreach_spend > 0 else None
    reactive_recovery_cost = defaulted_loans * full_default_recovery_cost_inr

    return {
        "assumptions": {
            "outreach_cost_inr_per_flagged_loan": outreach_cost_inr,
            "full_default_recovery_cost_inr_per_default": full_default_recovery_cost_inr,
            "proactive_cure_rate": proactive_cure_rate,
            "note": "Illustrative operating assumptions, not an industry benchmark; excludes EAD/LGD and lost interest.",
        },
        "flagged_loans": flagged_loans,
        "captured_defaults": captured_defaults,
        "expected_prevented_defaults": expected_prevented_defaults,
        "outreach_spend_inr": outreach_spend,
        "avoided_recovery_cost_inr": avoided_recovery_cost,
        "expected_net_savings_inr": net_savings,
        "roi_multiple_on_outreach_spend": roi,
        "reactive_recovery_cost_inr": reactive_recovery_cost,
    }


def save_backtest_plots(
    scored_snapshots: pd.DataFrame,
    backtest_results: pd.DataFrame,
    threshold: float,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    captured_ids = backtest_results.loc[backtest_results["captured_before_default"] == 1, "loan_id"].head(2).tolist()
    stable_ids = backtest_results.loc[
        (backtest_results["event"] == 0) & (backtest_results["ever_flagged"] == 0), "loan_id"
    ].head(1).tolist()
    example_ids = captured_ids + stable_ids
    if len(example_ids) < 3:
        filler = [x for x in backtest_results["loan_id"].tolist() if x not in example_ids]
        example_ids += filler[: 3 - len(example_ids)]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for loan_id in example_ids:
        g = scored_snapshots[scored_snapshots["loan_id"] == loan_id].sort_values("month")
        ax.plot(g["month"], g["predicted_default_risk_3m"], marker="o", label=loan_id)
    ax.axhline(threshold, linestyle="--", linewidth=1.2, label=f"Alert threshold = {threshold:.2%}")
    ax.set(xlabel="Checkpoint month", ylabel="Predicted default risk in next 3 months", title="Live borrower risk trajectories")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "example_risk_trajectories.png", dpi=160)
    plt.close(fig)

    leads = backtest_results.loc[backtest_results["captured_before_default"] == 1, "lead_time_months"].dropna()
    if len(leads):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bins = np.arange(0.5, max(2.5, leads.max() + 1.5), 1)
        ax.hist(leads, bins=bins, rwidth=0.85)
        ax.set(xlabel="Months of warning before default", ylabel="Defaulted loans", title="Early-warning lead-time distribution")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "early_warning_lead_time.png", dpi=160)
        plt.close(fig)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
