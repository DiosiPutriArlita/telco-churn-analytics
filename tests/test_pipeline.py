import numpy as np
import pandas as pd

from src import config, data_loader, features, leakage, modeling, prescriptive, quality
from src.config import CampaignParams
from tests.synthetic import make_telco_like


def test_clean_handles_blank_total_charges():
    raw = make_telco_like()
    assert raw["TotalCharges"].astype(str).str.strip().eq("").sum() > 0
    cleaned = data_loader.clean(raw)
    assert cleaned["TotalCharges"].notna().all()
    assert cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"].eq(0).all()
    assert set(cleaned["churn"].unique()) <= {0, 1}
    assert not cleaned["OnlineSecurity"].eq("No internet service").any()


def test_quality_rules_flag_blank_total_charges():
    raw = make_telco_like()
    rules = quality.rule_violations(raw).set_index("rule")
    assert rules.loc["TotalCharges kosong dan tenure = 0", "violations"] > 0
    assert rules.loc["TotalCharges kosong dan tenure > 0", "violations"] == 0


def test_leakage_audit_removes_known_and_planted_columns():
    cleaned = data_loader.clean(make_telco_like())
    rng = np.random.default_rng(1)
    cleaned["Churn Score"] = cleaned["churn"] * 80 + rng.integers(0, 15, len(cleaned))
    cleaned["post_event_flag"] = np.where(cleaned["churn"] == 1, "closed", "active")
    result = leakage.audit(cleaned)
    assert "Churn Score" in result["dropped_known"]
    assert "post_event_flag" in result["dropped_suspicious"]
    assert "Churn Score" not in result["dataframe"].columns
    assert "Contract" in result["dataframe"].columns


def test_stratified_split_preserves_class_ratio():
    df = features.engineer(data_loader.clean(make_telco_like()))
    X, y = data_loader.split_features_target(df)
    X_train, X_test, y_train, y_test = modeling.split(X, y)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert config.ID_COLUMN not in X.columns


def test_preprocessor_fits_only_on_training_data():
    df = features.engineer(data_loader.clean(make_telco_like()))
    X, y = data_loader.split_features_target(df)
    X_train, X_test, _, _ = modeling.split(X, y)
    prep = features.build_preprocessor().fit(X_train)
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    train_mean = X_train[features.feature_lists()[0]].mean().to_numpy()
    assert np.allclose(scaler.mean_, train_mean, atol=1e-6)
    assert prep.transform(X_test).shape[0] == len(X_test)


def test_threshold_tuning_returns_valid_probability():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.25).astype(int)
    p = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    threshold = modeling.tune_threshold(y, p)
    assert 0.05 <= threshold <= 0.95


def test_prescriptive_net_benefit_and_budget():
    params = CampaignParams(offer_cost=10, success_rate=0.5, horizon_months=10, gross_margin=0.5,
                            budget_share=0.5)
    frame = pd.DataFrame({
        "customerID": list("abcdef"),
        "MonthlyCharges": [100.0, 100.0, 100.0, 20.0, 20.0, 20.0],
        "churn_probability": [0.9, 0.5, 0.01, 0.9, 0.1, 0.01],
        "actual_churn": [1, 1, 0, 1, 0, 0],
        "Contract": ["Month-to-month"] * 6,
        "InternetService": ["Fiber optic"] * 6,
        "TechSupport": ["No"] * 6,
        "PaymentMethod": ["Electronic check"] * 6,
        "tenure": [5] * 6,
    })
    frame["customer_value"] = prescriptive.customer_value(frame["MonthlyCharges"], params)
    frame["expected_net_benefit"] = prescriptive.expected_net_benefit(
        frame["churn_probability"], frame["customer_value"], params)
    targets = prescriptive.select_targets(frame, params)
    assert len(targets) <= int(0.5 * len(frame))
    assert (targets["expected_net_benefit"] > 0).all()
    assert targets["customerID"].iloc[0] == "a"
    assert targets["recommended_action"].notna().all()
