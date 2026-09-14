import numpy as np

from data_simulation import SimulationConfig, create_landmark_dataset, create_monitoring_snapshots, simulate_loan_panel


def test_panel_schema_and_ranges():
    panel = simulate_loan_panel(SimulationConfig(n_loans=80, seed=7))
    required = {
        "loan_id", "month", "payment_status", "emi_amount_due", "emi_amount_paid",
        "rolling_delay_trend", "emi_to_income_ratio", "event", "time_to_event"
    }
    assert required.issubset(panel.columns)
    assert panel["loan_id"].nunique() == 80
    assert panel["emi_to_income_ratio"].between(0.08, 0.55).all()
    assert (panel["emi_amount_paid"] >= 0).all()
    assert np.isfinite(panel["rolling_delay_trend"]).all()


def test_landmark_has_no_nonpositive_durations():
    panel = simulate_loan_panel(SimulationConfig(n_loans=120, seed=11))
    landmark = create_landmark_dataset(panel, 3)
    assert (landmark["duration_from_landmark"] > 0).all()
    assert landmark["loan_id"].is_unique
    assert set(landmark["defaults_next_3m"].unique()).issubset({0, 1})


def test_monitoring_windows_start_at_month_three():
    panel = simulate_loan_panel(SimulationConfig(n_loans=60, seed=13))
    snapshots = create_monitoring_snapshots(panel, 3)
    assert snapshots["month"].min() >= 3
    assert snapshots["missed_count_3m"].between(0, 3).all()
