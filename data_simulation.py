"""Synthetic repayment-panel generation for PulseCheck.

The simulator is intentionally transparent: it creates stylized salaried and gig-income
patterns, then lets repayment stress affect delays, missed payments, and default hazard.
It is not calibrated to any lender's proprietary portfolio and must not be treated as
an empirical estimate of Indian borrower behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_loans: int = 1500
    seed: int = 42
    salaried_share: float = 0.65
    min_history_months: int = 12
    max_history_months: int = 24
    min_contract_months: int = 12
    max_contract_months: int = 36


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _rolling_slope(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    x = np.arange(values.size, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def simulate_loan_panel(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """Generate a month-level loan repayment panel.

    Default is simulated only from month 4 onward so that the monitoring model can use
    a clean month-3 landmark without look-ahead. A loan exits the panel at default or
    at its observation end (closed or right-censored active loan).
    """
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    for i in range(config.n_loans):
        loan_id = f"LN{i + 1:05d}"
        borrower_type = "salaried" if rng.random() < config.salaried_share else "gig"
        is_gig = borrower_type == "gig"

        if is_gig:
            base_income = float(np.clip(rng.lognormal(np.log(40_000), 0.45), 18_000, 200_000))
            ratio = float(0.08 + 0.47 * rng.beta(2.6, 5.2))
            income_sigma = 0.24
            stress_shock_sigma = 0.58
        else:
            base_income = float(np.clip(rng.lognormal(np.log(55_000), 0.34), 22_000, 220_000))
            ratio = float(0.08 + 0.47 * rng.beta(2.3, 5.8))
            income_sigma = 0.08
            stress_shock_sigma = 0.26

        ratio = float(np.clip(ratio, 0.08, 0.55))
        emi_due = float(np.round(base_income * ratio / 50) * 50)
        contract_months = int(rng.integers(config.min_contract_months, config.max_contract_months + 1))
        planned_history = int(rng.integers(config.min_history_months, config.max_history_months + 1))
        observation_cap = min(contract_months, planned_history)

        # Some borrowers experience a gradual deterioration in cash-flow conditions.
        drift = float(rng.uniform(0.04, 0.18)) if rng.random() < (0.28 if is_gig else 0.20) else 0.0
        stress = float(rng.normal(0.0, 0.35 if is_gig else 0.22))
        risk_trait = float(rng.normal(0.0, 1.0))
        delay_history: list[float] = []
        miss_history: list[int] = []
        loan_rows: list[dict] = []
        default_month: int | None = None

        for month in range(1, observation_cap + 1):
            # Mean-preserving lognormal income noise; gig borrowers receive larger shocks.
            income_factor = float(rng.lognormal(-0.5 * income_sigma**2, income_sigma))
            realized_income = max(10_000.0, base_income * income_factor)
            effective_ratio = emi_due / realized_income

            stress = 0.65 * stress + rng.normal(0.0, stress_shock_sigma) + drift
            recent_misses = int(sum(miss_history[-2:])) if miss_history else 0
            recent_delay = float(np.mean(delay_history[-2:])) if delay_history else 0.0

            miss_logit = (
                -4.35
                + 4.6 * (effective_ratio - 0.25)
                + 0.95 * max(stress, -1.5)
                + 0.75 * recent_misses
                + 0.020 * recent_delay
                + 0.45 * risk_trait
                + (0.30 if is_gig else 0.0)
            )
            late_logit = (
                -1.85
                + 3.2 * (effective_ratio - 0.25)
                + 0.72 * max(stress, -1.5)
                + 0.35 * recent_misses
                + 0.25 * risk_trait
                + (0.35 if is_gig else 0.0)
            )
            p_miss = float(np.clip(_sigmoid(miss_logit), 0.01, 0.60))
            p_late = float(np.clip(_sigmoid(late_logit) * (1.0 - p_miss), 0.03, 0.70))

            draw = rng.random()
            if draw < p_miss:
                status_group = "missed"
                delay_days = int(np.clip(30 + 12 * recent_misses + rng.normal(8, 9), 30, 90))
                paid_fraction = float(np.clip(rng.beta(1.2, 5.0) * 0.55, 0.0, 0.55))
                emi_paid = float(np.round(emi_due * paid_fraction / 10) * 10)
                payment_status = "missed"
                missed = 1
            elif draw < p_miss + p_late:
                status_group = "late"
                scale = 5.5 if not is_gig else 8.5
                delay_days = int(np.clip(1 + rng.gamma(1.8, scale) + 2.5 * max(stress, 0), 1, 60))
                paid_fraction = float(np.clip(rng.normal(0.94, 0.08), 0.65, 1.0))
                emi_paid = float(np.round(emi_due * paid_fraction / 10) * 10)
                payment_status = f"late_{delay_days}d"
                missed = 0
            else:
                status_group = "on_time"
                delay_days = 0
                emi_paid = emi_due
                payment_status = "on_time"
                missed = 0

            delay_history.append(float(delay_days))
            miss_history.append(missed)
            avg_delay_3m = float(np.mean(delay_history[-3:]))
            missed_3m = int(sum(miss_history[-3:]))
            delay_trend = _rolling_slope(delay_history[-3:])

            # A stylized conditional default hazard. The same observable stress signals
            # later used by the model influence true risk, but randomness remains.
            if month >= 4:
                default_logit = (
                    -4.95
                    + 1.20 * missed_3m
                    + 0.040 * avg_delay_3m
                    + 3.0 * (ratio - 0.25)
                    + 0.42 * max(stress, 0)
                    + 0.65 * risk_trait
                    + (0.28 if is_gig else 0.0)
                )
                p_default = float(np.clip(_sigmoid(default_logit), 0.002, 0.30))
                if rng.random() < p_default:
                    default_month = month

            loan_rows.append(
                {
                    "loan_id": loan_id,
                    "borrower_type": borrower_type,
                    "month": month,
                    "contract_tenure_months": contract_months,
                    "monthly_income_at_origination": round(base_income, 2),
                    "realized_monthly_income": round(realized_income, 2),
                    "emi_amount_due": round(emi_due, 2),
                    "emi_amount_paid": round(emi_paid, 2),
                    "emi_to_income_ratio": round(ratio, 4),
                    "payment_status": payment_status,
                    "payment_status_group": status_group,
                    "delay_days": delay_days,
                    "rolling_avg_delay_3m": round(avg_delay_3m, 3),
                    "rolling_delay_trend": round(delay_trend, 3),
                    "missed_count_trailing_3m": missed_3m,
                }
            )
            if default_month is not None:
                break

        event = int(default_month is not None)
        observed_months = default_month if event else len(loan_rows)
        if event:
            end_reason = "default"
            time_to_event = default_month
        else:
            end_reason = "closed_paid" if observed_months >= contract_months else "active_censored"
            time_to_event = observed_months

        for row in loan_rows:
            row.update(
                {
                    "event": event,
                    "time_to_event": int(time_to_event),
                    "observation_end_reason": end_reason,
                }
            )
            rows.append(row)

    panel = pd.DataFrame(rows).sort_values(["loan_id", "month"]).reset_index(drop=True)
    return panel


def create_landmark_dataset(panel: pd.DataFrame, landmark_month: int = 3) -> pd.DataFrame:
    """Create one row per loan at a landmark month for leakage-safe model fitting."""
    eligible = panel.groupby("loan_id")["month"].max()
    eligible_ids = eligible[eligible >= landmark_month].index
    p = panel[panel["loan_id"].isin(eligible_ids)].copy()

    first_window = p[p["month"] <= landmark_month]
    first_window = first_window.copy()
    # Keep lateness intensity and missed-payment frequency as distinct signals;
    # otherwise a missed EMI (30+ delay days) mechanically duplicates the miss count.
    first_window["late_delay_for_model"] = np.where(
        first_window["payment_status_group"] == "late", first_window["delay_days"], 0.0
    )
    features = first_window.groupby("loan_id").agg(
        avg_delay_3m=("late_delay_for_model", "mean"),
        missed_count_3m=("payment_status_group", lambda s: int((s == "missed").sum())),
    )
    at_landmark = p[p["month"] == landmark_month].set_index("loan_id")
    last_rows = p.groupby("loan_id", as_index=False).tail(1).set_index("loan_id")

    landmark = features.join(
        at_landmark[["emi_to_income_ratio", "rolling_delay_trend", "borrower_type"]], how="inner"
    ).join(last_rows[["event", "time_to_event", "observation_end_reason"]], how="inner")
    landmark["duration_from_landmark"] = landmark["time_to_event"] - landmark_month
    landmark = landmark[landmark["duration_from_landmark"] > 0].copy()
    landmark["defaults_next_3m"] = (
        (landmark["event"] == 1)
        & (landmark["duration_from_landmark"] <= 3)
    ).astype(int)
    landmark = landmark.reset_index()
    return landmark


def create_monitoring_snapshots(panel: pd.DataFrame, start_month: int = 3) -> pd.DataFrame:
    """Create monthly trailing-window features compatible with the month-3 Cox schema."""
    records: list[dict] = []
    for loan_id, g in panel.groupby("loan_id", sort=False):
        g = g.sort_values("month")
        for idx in range(start_month - 1, len(g)):
            row = g.iloc[idx]
            window = g.iloc[max(0, idx - 2) : idx + 1]
            records.append(
                {
                    "loan_id": loan_id,
                    "month": int(row["month"]),
                    "borrower_type": row["borrower_type"],
                    "avg_delay_3m": float(
                        window.loc[window["payment_status_group"] == "late", "delay_days"].sum() / len(window)
                    ),
                    "emi_to_income_ratio": float(row["emi_to_income_ratio"]),
                    "rolling_delay_trend": float(row["rolling_delay_trend"]),
                    "missed_count_3m": int((window["payment_status_group"] == "missed").sum()),
                    "event": int(row["event"]),
                    "time_to_event": int(row["time_to_event"]),
                    "observation_end_reason": row["observation_end_reason"],
                }
            )
    return pd.DataFrame(records)


def save_panel(panel: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    panel.to_csv(path, index=False, compression=compression)
    return path


if __name__ == "__main__":
    data = simulate_loan_panel()
    out = save_panel(data, "data/simulated_loan_panel.csv.gz")
    print(f"Saved {len(data):,} panel rows across {data['loan_id'].nunique():,} loans to {out}")
