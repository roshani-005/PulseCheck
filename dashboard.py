"""Streamlit dashboard for PulseCheck."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter

from data_simulation import SimulationConfig, create_landmark_dataset, create_monitoring_snapshots, save_panel, simulate_loan_panel
from early_warning import choose_alert_threshold, score_monitoring_snapshots
from survival_model import FEATURES

st.set_page_config(page_title="PulseCheck", page_icon="💓", layout="wide")

DATA_PATH = Path("data/simulated_loan_panel.csv.gz")
SUMMARY_PATH = Path("outputs/summary_metrics.json")


@st.cache_data
def load_panel() -> pd.DataFrame:
    if not DATA_PATH.exists():
        panel = simulate_loan_panel(SimulationConfig(n_loans=1500, seed=42))
        save_panel(panel, DATA_PATH)
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def fit_dashboard_cox(panel_json: str) -> CoxPHFitter:
    panel = pd.read_json(panel_json, orient="split")
    landmark = create_landmark_dataset(panel, 3)
    cph = CoxPHFitter(penalizer=0.02)
    cph.fit(landmark[["duration_from_landmark", "event", *FEATURES]], "duration_from_landmark", "event")
    return cph


def km_trace(frame: pd.DataFrame, name: str) -> go.Scatter:
    km = KaplanMeierFitter()
    km.fit(frame["time_to_event"], frame["event"], label=name)
    return go.Scatter(x=km.survival_function_.index, y=km.survival_function_[name], mode="lines", name=name)


panel = load_panel()
loan_level = panel.sort_values(["loan_id", "month"]).groupby("loan_id", as_index=False).tail(1)
landmark = create_landmark_dataset(panel, 3)
cph = fit_dashboard_cox(panel.to_json(orient="split"))
snapshots = create_monitoring_snapshots(panel, 3)
scored = score_monitoring_snapshots(cph, snapshots)

if SUMMARY_PATH.exists():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    threshold = float(summary["early_warning"]["threshold"])
else:
    month3 = scored[scored["month"] == 3].merge(
        landmark[["loan_id", "defaults_next_3m"]], on="loan_id", how="inner"
    )
    threshold, _ = choose_alert_threshold(
        month3["defaults_next_3m"].to_numpy(), month3["predicted_default_risk_3m"].to_numpy()
    )

st.title("PulseCheck: Time-to-Default Monitoring Using Survival Analysis")
st.caption("Synthetic portfolio demo · survival-based monitoring during the active life of a loan")

c1, c2, c3 = st.columns(3)
c1.metric("Loans", f"{loan_level['loan_id'].nunique():,}")
c2.metric("Observed default rate", f"{loan_level['event'].mean():.1%}")
c3.metric("Alert threshold", f"{threshold:.1%}")

st.subheader("Portfolio survival")
segment = st.radio("Segment", ["Overall", "Borrower type", "EMI-to-income tier"], horizontal=True)
fig = go.Figure()
if segment == "Overall":
    fig.add_trace(km_trace(loan_level, "All loans"))
elif segment == "Borrower type":
    for key, g in loan_level.groupby("borrower_type"):
        fig.add_trace(km_trace(g, key.title()))
else:
    tmp = loan_level.copy()
    tmp["emi_tier"] = pd.cut(
        tmp["emi_to_income_ratio"], [0.0, 0.20, 0.35, np.inf], labels=["Low ≤20%", "Medium 20–35%", "High >35%"]
    )
    for key, g in tmp.groupby("emi_tier", observed=True):
        fig.add_trace(km_trace(g, str(key)))
fig.update_layout(xaxis_title="Months since origination", yaxis_title="P(no default yet)", yaxis_range=[0, 1.02])
st.plotly_chart(fig, use_container_width=True)

st.subheader("Borrower live risk trajectory")
loan_options = scored["loan_id"].drop_duplicates().tolist()
selected = st.selectbox("Loan", loan_options, index=0)
g = scored[scored["loan_id"] == selected].sort_values("month").copy()
g["flag"] = g["predicted_default_risk_3m"] >= threshold

risk_fig = go.Figure()
risk_fig.add_trace(
    go.Scatter(
        x=g["month"], y=g["predicted_default_risk_3m"], mode="lines+markers", name="3-month default risk"
    )
)
risk_fig.add_hline(y=threshold, line_dash="dash", annotation_text="outreach threshold")
risk_fig.update_layout(xaxis_title="Checkpoint month", yaxis_title="Predicted default risk in next 3 months", yaxis_range=[0, max(0.5, float(g['predicted_default_risk_3m'].max()) * 1.2)])
st.plotly_chart(risk_fig, use_container_width=True)

latest = g.iloc[-1]
status = "FLAG FOR PROACTIVE OUTREACH" if latest["flag"] else "Monitor"
st.metric("Current status", status)
st.dataframe(
    g[["month", "avg_delay_3m", "rolling_delay_trend", "missed_count_3m", "predicted_default_risk_3m", "flag"]],
    use_container_width=True,
    hide_index=True,
)

st.info(
    "The dashboard is a modeling demo. The data are synthetic and the alert threshold is optimized on simulated outcomes; production use requires lender-specific calibration, validation, fairness review, and collections policy controls."
)
