import numpy as np
import pandas as pd


def make_telco_like(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    tenure = np.where(contract == "Month-to-month", rng.integers(0, 40, n), rng.integers(10, 73, n))
    phone = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n)

    def addon():
        values = rng.choice(["Yes", "No"], n)
        return np.where(internet == "No", "No internet service", values)

    tech = addon()
    monthly = np.round(
        np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 50, 22))
        + rng.normal(0, 8, n), 2).clip(18.25, 119)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)

    logit = (-1.0 + 1.4 * (contract == "Month-to-month") - 0.03 * tenure
             + 0.8 * (internet == "Fiber optic") + 0.6 * (payment == "Electronic check")
             - 0.5 * (tech == "Yes") + rng.normal(0, 0.5, n))
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
        "OnlineSecurity": addon(),
        "OnlineBackup": addon(),
        "DeviceProtection": addon(),
        "TechSupport": tech,
        "StreamingTV": addon(),
        "StreamingMovies": addon(),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(object),
        "Churn": np.where(churn, "Yes", "No"),
    })
    zero = df["tenure"] == 0
    df.loc[zero, "TotalCharges"] = " "
    return df
