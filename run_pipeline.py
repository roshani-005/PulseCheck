"""End-to-end reproducible experiment runner for PulseCheck."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from baseline_classifier import fit_logistic_baseline
from data_simulation import (
    SimulationConfig,
    create_landmark_dataset,
    create_monitoring_snapshots,
    save_panel,
    simulate_loan_panel,
)
from early_warning import (
    backtest_early_warning,
    choose_alert_threshold,
    estimate_roi,
    save_backtest_plots,
    save_json,
    score_monitoring_snapshots,
)
from survival_model import fit_cox_model, horizon_metrics, plot_kaplan_meier, predict_horizon_default_risk


def build_report(hazard, ph_diag, baseline_metrics, cox_metrics, warning, roi) -> str:
    lines = [
        "# PulseCheck model report",
        "",
        "> Generated from synthetic data. Metrics demonstrate the workflow; they are not estimates of real lender performance.",
        "",
        "## Cox hazard ratios",
        "",
    ]
    for _, row in hazard.iterrows():
        hr = float(row["exp(coef)"])
        feature = row["feature"]
        p = float(row["p"])
        if feature == "emi_to_income_ratio":
            interp = f"A +0.10 increase in EMI/income corresponds to roughly {hr ** 0.10:.2f}× the instantaneous default hazard, holding other model features fixed."
        elif feature == "missed_count_3m":
            interp = f"One additional missed payment in the trailing 3-month window corresponds to {hr:.2f}× the instantaneous default hazard."
        elif feature == "avg_delay_3m":
            interp = f"Each extra average day of payment delay in the 3-month window corresponds to {hr:.2f}× the instantaneous default hazard."
        else:
            interp = f"A one-unit increase corresponds to {hr:.2f}× the instantaneous default hazard."
        lines += [f"- **{feature}** — HR={hr:.3f}, p={p:.4f}. {interp}"]

    lines += [
        "",
        "## Proportional-hazards check",
        "",
        f"- Pass at p=0.05: **{ph_diag['assumption_passed_at_0_05']}**",
        f"- Flagged features: {', '.join(ph_diag['violating_features']) if ph_diag['violating_features'] else 'none'}",
        "- Full `lifelines.check_assumptions` console output is saved in `outputs/ph_assumption_check.txt`.",
        "",
        "## Baseline vs survival",
        "",
        f"- Logistic next-3-month AUROC: {baseline_metrics['roc_auc']:.3f}" if baseline_metrics["roc_auc"] is not None else "- Logistic AUROC: unavailable (single-class test set)",
        f"- Logistic Brier score: {baseline_metrics['brier']:.4f}",
        f"- Cox next-3-month AUROC: {cox_metrics['roc_auc']:.3f}" if cox_metrics["roc_auc"] is not None else "- Cox AUROC: unavailable (single-class test set)",
        f"- Cox next-3-month Brier score: {cox_metrics['brier']:.4f}",
        "- The logistic model emits one month-3 probability. Cox also models event timing and supports a survival curve; PulseCheck re-applies the fitted trailing-window schema at each monthly checkpoint to produce an updated warning trajectory.",
        "",
        "## Early-warning backtest",
        "",
        f"- Alert threshold: {warning['threshold']:.2%}",
        f"- Pre-default capture rate: {warning['capture_rate']:.1%}" if warning["capture_rate"] is not None else "- Capture rate: unavailable",
        f"- Median lead time among captured defaults: {warning['median_lead_months']:.1f} months" if warning["median_lead_months"] is not None else "- Median lead time: unavailable",
        f"- Reactive baseline lead time: {warning['reactive_baseline_lead_months']:.0f} months",
        f"- Non-default loan-level flag rate: {warning['nondefault_flag_rate']:.1%}" if warning["nondefault_flag_rate"] is not None else "- Non-default flag rate: unavailable",
        "",
        "## Illustrative ROI",
        "",
        f"- Outreach spend: ₹{roi['outreach_spend_inr']:,.0f}",
        f"- Expected avoided recovery cost: ₹{roi['avoided_recovery_cost_inr']:,.0f}",
        f"- Expected net savings: ₹{roi['expected_net_savings_inr']:,.0f}",
        f"- ROI on outreach spend: {roi['roi_multiple_on_outreach_spend']:.2f}×" if roi["roi_multiple_on_outreach_spend"] is not None else "- ROI: unavailable",
        "- This is a scenario calculation, not a causal estimate. The cure rate and cost assumptions must be replaced with lender-specific evidence before deployment.",
        "",
    ]
    return "\n".join(lines)


def main(n_loans: int = 1500, seed: int = 42) -> dict:
    data_dir = Path("data")
    output_dir = Path("outputs")
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    panel = simulate_loan_panel(SimulationConfig(n_loans=n_loans, seed=seed))
    save_panel(panel, data_dir / "simulated_loan_panel.csv.gz")
    plot_kaplan_meier(panel, output_dir)

    landmark = create_landmark_dataset(panel, landmark_month=3)
    stratify = landmark["event"] if landmark["event"].nunique() > 1 else None
    train, test = train_test_split(landmark, test_size=0.30, random_state=seed, stratify=stratify)

    cph, hazard, ph_diag = fit_cox_model(train, output_dir)
    cox_metrics = horizon_metrics(cph, test, horizon_months=3)
    _, baseline_metrics, _ = fit_logistic_baseline(train, test, output_dir)

    train_risk = predict_horizon_default_risk(cph, train, horizon_months=3)
    threshold, f2 = choose_alert_threshold(train["defaults_next_3m"].to_numpy(), train_risk, beta=2.0)

    snapshots = create_monitoring_snapshots(panel, start_month=3)
    scored = score_monitoring_snapshots(cph, snapshots, horizon_months=3)
    backtest, warning = backtest_early_warning(scored, threshold, set(test["loan_id"]))
    warning["train_threshold_f2"] = f2
    roi = estimate_roi(backtest)

    backtest.to_csv(output_dir / "early_warning_backtest.csv", index=False)
    scored[scored["loan_id"].isin(set(test["loan_id"]))].to_csv(output_dir / "test_monitoring_scores.csv", index=False)
    save_json(warning, output_dir / "early_warning_summary.json")
    save_json(roi, output_dir / "roi_case.json")
    save_backtest_plots(scored, backtest, threshold, output_dir)

    summary = {
        "n_loans": int(panel["loan_id"].nunique()),
        "n_panel_rows": int(len(panel)),
        "default_rate": float(panel.groupby("loan_id")["event"].first().mean()),
        "borrower_mix": panel.groupby("loan_id")["borrower_type"].first().value_counts(normalize=True).to_dict(),
        "baseline": baseline_metrics,
        "cox_horizon": cox_metrics,
        "early_warning": warning,
        "roi": roi,
        "ph_assumption": {
            "assumption_passed_at_0_05": ph_diag["assumption_passed_at_0_05"],
            "violating_features": ph_diag["violating_features"],
        },
    }
    save_json(summary, output_dir / "summary_metrics.json")
    (output_dir / "model_report.md").write_text(
        build_report(hazard, ph_diag, baseline_metrics, cox_metrics, warning, roi), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-loans", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(n_loans=args.n_loans, seed=args.seed)
