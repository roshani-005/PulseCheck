# PulseCheck: Time-to-Default Monitoring Using Survival Analysis

**PulseCheck** is an early-warning system for active loans. A traditional credit-risk model answers **“should we approve this borrower?”** at origination. PulseCheck answers a different question every month after approval: **“given what has happened so far, how much time-to-default risk is building now?”**

The project uses synthetic month-by-month repayment data, Kaplan–Meier survival curves, a Cox proportional-hazards model, a month-3 logistic baseline, dynamic monthly warning scores, a lead-time backtest, and a Streamlit dashboard.

> **Important:** every portfolio record in this repository is synthetic. The salaried/gig differences are stylized assumptions designed to create irregular-vs-stable cash-flow patterns; they are not empirical claims about Indian borrowers or an industry benchmark.

## Problem

Approval models are intentionally static: they summarize information available before the loan starts. But repayment behavior can deteriorate later because cash flow becomes volatile, EMI burden becomes harder to service, or small payment delays start compounding. A classifier can predict “default in the next 3 months,” but a one-off score discards two useful pieces of information: **censoring** (many loans have not defaulted yet) and **event timing** (default risk evolves across the active life of the loan).

PulseCheck treats default as a **time-to-event** problem. That makes it possible to estimate survival curves, compare borrower segments, quantify how observed repayment signals alter the instantaneous default hazard, and turn those signals into an operational monthly outreach trigger.

## Approach

### 1) Synthetic panel data

`data_simulation.py` generates 1,500 loans by default. Each loan has a 12–24 month observation window unless it defaults earlier. The panel contains:

- `payment_status`: `on_time`, `late_Nd`, or `missed`
- `emi_amount_due` and `emi_amount_paid`
- `delay_days`, `rolling_avg_delay_3m`, and `rolling_delay_trend`
- static `emi_to_income_ratio`
- `event` and `time_to_event`
- `borrower_type`: `salaried` or `gig`
- explicit right-censoring as `active_censored` or `closed_paid`

Gig income is simulated with higher month-to-month variance than salaried income; repayment delays remain stochastic for both groups. The default process is driven by observable repayment stress plus randomness, so the modeling task is learnable without being deterministic.

### 2) Leakage-safe survival modeling

The core Cox model is fit at a **month-3 landmark**. That is a deliberate design choice: the model only sees information available by the end of month 3, and survival duration is measured from that landmark forward. This avoids using month-3 behavior to “predict” an event that happened in month 1 or 2.

The Cox features are exactly the monitoring signals requested:

- average payment delay in the 3-month window
- EMI-to-income ratio
- rolling 3-month delay trend
- missed-payment count in the 3-month window

At month 3 these are the first-three-month features. During live monitoring, PulseCheck slides the same 3-month feature schema forward and re-scores each active loan every month. This is a **dynamic landmark-style application of a Cox PH model**, not a full time-varying-coefficient Cox model; that distinction matters and is discussed below.

`survival_model.py` also fits Kaplan–Meier curves overall and by borrower type and EMI-to-income tier. It saves Cox hazard ratios with 95% confidence intervals and runs both `lifelines.CoxPHFitter.check_assumptions(...)` and a rank-transformed proportional-hazard test. The raw diagnostic is persisted to `outputs/ph_assumption_check.txt`; violations are not suppressed.

### 3) Static baseline

`baseline_classifier.py` trains logistic regression on the same month-3 features to predict **default in months 4–6**. This is an intentionally simple comparison:

| Static classifier | Survival model |
|---|---|
| One next-3-month probability at month 3 | Models time-to-event and censoring |
| Optimized for a fixed prediction horizon | Produces a survival function over time |
| Easy to calibrate and deploy | Better suited to “when does risk emerge?” |
| Needs a new snapshot/modeling convention to update | Can be re-scored at each monthly landmark |

AUROC and Brier score are reported for both at the 3-month horizon, but those metrics are not the whole argument for survival analysis. The differentiator is the **risk trajectory and lead time**, not a guaranteed accuracy win.

### 4) Early-warning policy

`early_warning.py` converts the Cox model into a 3-month default-risk score, then selects an alert threshold **using the training set only** by maximizing F2 (recall-weighted). At every checkpoint:

```text
if predicted_default_risk_next_3m >= threshold:
    flag for proactive outreach
else:
    continue monitoring
```

The held-out backtest asks, for every loan that eventually defaults: **what was the first flag before default, and how many months of warning did it create?** A purely reactive process has 0 months of lead time by definition.

### 5) Business layer

The ROI calculation is deliberately a scenario model, not a causal claim. Default assumptions are:

- ₹250 per first proactive outreach
- ₹12,000 operating cost for full default recovery
- 25% of correctly early-flagged defaults are cured by outreach

Expected savings are:

```text
expected prevented defaults = captured defaults × cure rate
avoided recovery cost       = prevented defaults × recovery cost
net savings                 = avoided recovery cost − outreach spend
ROI                          = net savings / outreach spend
```

This excludes EAD/LGD, lost interest, contact-channel variation, and the possibility that collections treatment itself changes repayment behavior. Replace these assumptions with lender-specific evidence before using the ROI output for a real decision.

## Repository structure

```text
PulseCheck/
├── data_simulation.py          # synthetic month-level portfolio
├── survival_model.py           # KM + Cox PH + PH diagnostics
├── baseline_classifier.py      # month-3 logistic baseline
├── early_warning.py            # monthly alerts, lead-time backtest, ROI
├── dashboard.py                # Streamlit monitoring dashboard
├── run_pipeline.py             # end-to-end reproducible experiment
├── data/
│   └── simulated_loan_panel.csv.gz  # generated reference dataset
├── outputs/
│   ├── km_overall.png
│   ├── km_by_borrower_type.png
│   ├── km_by_emi_tier.png
│   ├── cox_hazard_ratios.png
│   ├── baseline_roc.png
│   ├── baseline_calibration.png
│   ├── example_risk_trajectories.png
│   ├── early_warning_lead_time.png
│   ├── hazard_ratios.csv
│   ├── ph_assumption_check.txt
│   ├── early_warning_summary.json
│   ├── roi_case.json
│   ├── summary_metrics.json
│   └── model_report.md
└── tests/
```

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --n-loans 1500 --seed 42
streamlit run dashboard.py
```

The end-to-end run regenerates the synthetic dataset, all plots, hazard ratios, proportional-hazards diagnostics, classifier metrics, warning backtest, and ROI case.

## Reading the hazard ratios

`outputs/model_report.md` turns every fitted Cox coefficient into a plain-English interpretation. Examples of the interpretation logic:

- if `missed_count_3m` has HR = 2.0, one additional missed payment in the 3-month window is associated with **2.0× the instantaneous default hazard**, conditional on the other model features;
- because EMI/income is a fraction, the report converts its coefficient into a **+10 percentage-point** effect rather than misleadingly interpreting a full +1.0 jump;
- hazard ratios are associations in this synthetic data-generating process, **not causal effects**.

## Dashboard

The Streamlit app shows the overall/segmented Kaplan–Meier curves and lets you select one loan to inspect its monthly trailing features, next-3-month Cox risk, and current outreach flag.

```bash
streamlit run dashboard.py
```

## Limitations

The project is designed to demonstrate modeling architecture, not to claim production readiness.

1. **Synthetic calibration.** Default rates, income distributions, recovery costs, and cure rates are assumed. A real deployment must re-estimate all of them on lender data.
2. **Dynamic landmark approximation.** The Cox model is fit at month 3, then the same trailing-window feature definitions are re-applied later. That creates a useful live score, but it is not mathematically identical to fitting a Cox model with fully time-varying covariates. A production extension should compare `CoxTimeVaryingFitter`, discrete-time hazard models, and landmark ensembles.
3. **PH assumption can fail.** PulseCheck checks it and writes the result instead of assuming proportionality. If a feature violates PH, options include stratification, time interactions, or a different hazard model.
4. **Treatment effects are absent.** The backtest asks when a system would have flagged, not whether outreach would truly prevent default. Cure-rate ROI is therefore scenario analysis.
5. **Operational costs are simplified.** A real collections strategy must account for contact frequency, customer experience, false-positive capacity, EAD/LGD, fairness, and policy constraints.
6. **No approval-model replacement.** PulseCheck complements an origination model; it does not replace underwriting.

## Why this project matters

The interesting question is not “can a classifier predict default?” It is **whether a lender can identify deteriorating repayment behavior early enough to act while correctly handling loans that have not yet experienced the event**. Survival analysis makes that timing explicit—and gives the business a metric a static score cannot: **months of warning before default**.
