from pathlib import Path

from sklearn.model_selection import train_test_split

from data_simulation import SimulationConfig, create_landmark_dataset, simulate_loan_panel
from survival_model import fit_cox_model, predict_horizon_default_risk


def test_cox_smoke(tmp_path: Path):
    panel = simulate_loan_panel(SimulationConfig(n_loans=220, seed=21))
    landmark = create_landmark_dataset(panel, 3)
    train, test = train_test_split(landmark, test_size=0.25, random_state=21)
    cph, hazard, diag = fit_cox_model(train, tmp_path)
    risk = predict_horizon_default_risk(cph, test)
    assert len(risk) == len(test)
    assert ((risk >= 0) & (risk <= 1)).all()
    assert set(hazard["feature"]) == {"avg_delay_3m", "emi_to_income_ratio", "rolling_delay_trend", "missed_count_3m"}
    assert "assumption_passed_at_0_05" in diag
