import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.config import CampaignParams

ACTION_RULES = [
    (
        "Bundel dukungan teknis dan keamanan gratis 3 bulan",
        lambda r: r["Contract"] == "Month-to-month"
        and r["InternetService"] == "Fiber optic"
        and r["TechSupport"] == "No",
    ),
    (
        "Program onboarding dan diskon untuk kontrak 12 bulan",
        lambda r: r["Contract"] == "Month-to-month" and r["tenure"] <= 12,
    ),
    (
        "Insentif upgrade ke kontrak 1 tahun",
        lambda r: r["Contract"] == "Month-to-month",
    ),
    (
        "Insentif pindah ke pembayaran otomatis",
        lambda r: r["PaymentMethod"] == "Electronic check",
    ),
]
DEFAULT_ACTION = "Panggilan retensi oleh tim layanan pelanggan"


def customer_value(monthly_charges: pd.Series, params: CampaignParams) -> pd.Series:
    return monthly_charges * params.horizon_months * params.gross_margin


def expected_net_benefit(probability: pd.Series, value: pd.Series, params: CampaignParams) -> pd.Series:
    return probability * params.success_rate * value - params.offer_cost


def break_even_probability(value: pd.Series, params: CampaignParams) -> pd.Series:
    return params.offer_cost / (params.success_rate * value)


def assign_action(row: pd.Series) -> str:
    for action, condition in ACTION_RULES:
        if condition(row):
            return action
    return DEFAULT_ACTION


def build_scoring_frame(features: pd.DataFrame, probabilities: np.ndarray, customer_ids: pd.Series,
                        y: np.ndarray, params: CampaignParams) -> pd.DataFrame:
    frame = features.copy()
    frame.insert(0, config.ID_COLUMN, customer_ids.to_numpy())
    frame["churn_probability"] = probabilities
    frame["actual_churn"] = y
    frame["customer_value"] = customer_value(frame["MonthlyCharges"], params)
    frame["expected_net_benefit"] = expected_net_benefit(
        frame["churn_probability"], frame["customer_value"], params)
    frame["break_even_probability"] = break_even_probability(frame["customer_value"], params)
    return frame


def select_targets(frame: pd.DataFrame, params: CampaignParams) -> pd.DataFrame:
    budget = int(np.floor(params.budget_share * len(frame)))
    eligible = frame[frame["expected_net_benefit"] > 0]
    targets = eligible.sort_values("expected_net_benefit", ascending=False).head(budget).copy()
    targets["recommended_action"] = targets.apply(assign_action, axis=1)
    return targets


def realised_benefit(selected: pd.DataFrame, params: CampaignParams) -> float:
    saved = (selected["actual_churn"] * params.success_rate * selected["customer_value"]).sum()
    return float(saved - params.offer_cost * len(selected))


def compare_strategies(frame: pd.DataFrame, targets: pd.DataFrame, params: CampaignParams,
                       draws: int = 500) -> pd.DataFrame:
    n_selected = len(targets)
    rng = np.random.default_rng(config.RANDOM_STATE)
    random_benefit, random_reached = [], []
    for _ in range(draws):
        sample = frame.iloc[rng.choice(len(frame), n_selected, replace=False)]
        random_benefit.append(realised_benefit(sample, params))
        random_reached.append(sample["actual_churn"].sum())
    by_probability = frame.sort_values("churn_probability", ascending=False).head(n_selected)
    unconstrained = frame[frame["expected_net_benefit"] > 0]
    rows = [
        ("tanpa kampanye", 0, 0.0, 0.0),
        ("hubungi semua pelanggan", len(frame), realised_benefit(frame, params),
         float(frame["actual_churn"].sum())),
        ("acak (ukuran sama)", n_selected, float(np.mean(random_benefit)),
         float(np.mean(random_reached))),
        ("top probabilitas churn", n_selected, realised_benefit(by_probability, params),
         float(by_probability["actual_churn"].sum())),
        ("top expected net benefit (dengan anggaran)", n_selected, realised_benefit(targets, params),
         float(targets["actual_churn"].sum())),
        ("semua expected benefit positif (tanpa anggaran)", len(unconstrained),
         realised_benefit(unconstrained, params), float(unconstrained["actual_churn"].sum())),
    ]
    return pd.DataFrame(rows, columns=["strategy", "customers_contacted",
                                       "realised_net_benefit", "churners_reached"])


def sensitivity(frame: pd.DataFrame, success_rates=(0.1, 0.2, 0.3, 0.4, 0.5),
                offer_costs=(5, 15, 30, 50), base: CampaignParams = CampaignParams()) -> pd.DataFrame:
    rows = []
    for success in success_rates:
        for cost in offer_costs:
            params = CampaignParams(offer_cost=cost, success_rate=success,
                                    horizon_months=base.horizon_months,
                                    gross_margin=base.gross_margin, budget_share=1.0)
            benefit = expected_net_benefit(frame["churn_probability"],
                                           customer_value(frame["MonthlyCharges"], params), params)
            positive = benefit[benefit > 0]
            rows.append({"success_rate": success, "offer_cost": cost,
                         "customers_targeted": len(positive),
                         "expected_net_benefit": float(positive.sum())})
    return pd.DataFrame(rows)


def action_summary(targets: pd.DataFrame) -> pd.DataFrame:
    summary = targets.groupby("recommended_action").agg(
        customers=("customerID", "size"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        expected_net_benefit=("expected_net_benefit", "sum"),
    )
    return summary.sort_values("expected_net_benefit", ascending=False).reset_index()


def plot_strategies(table: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    colors = ["#999999", "#bbbbbb", "#bbbbbb", "#4c78a8", "#e45756", "#54a24b"]
    ax.barh(table["strategy"], table["realised_net_benefit"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net benefit pada data uji")
    ax.set_title("Perbandingan strategi kampanye retensi")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sensitivity(table: pd.DataFrame, path):
    pivot = table.pivot(index="success_rate", columns="offer_cost", values="expected_net_benefit")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    image = ax.imshow(pivot.values, cmap="Greens", aspect="auto", origin="lower")
    ax.set_xticks(range(pivot.shape[1]), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{s:.0%}" for s in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    ax.set(xlabel="Biaya penawaran per pelanggan", ylabel="Tingkat keberhasilan retensi",
           title="Sensitivitas expected net benefit")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(frame: pd.DataFrame, params: CampaignParams = CampaignParams()) -> dict:
    targets = select_targets(frame, params)
    strategies = compare_strategies(frame, targets, params)
    sens = sensitivity(frame, base=params)
    actions = action_summary(targets)

    targets.to_csv(config.TABLES_DIR / "prescriptive_target_list.csv", index=False)
    strategies.to_csv(config.TABLES_DIR / "prescriptive_strategy_comparison.csv", index=False)
    sens.to_csv(config.TABLES_DIR / "prescriptive_sensitivity.csv", index=False)
    actions.to_csv(config.TABLES_DIR / "prescriptive_action_summary.csv", index=False)

    plot_strategies(strategies, config.FIGURES_DIR / "09_strategy_comparison.png")
    plot_sensitivity(sens, config.FIGURES_DIR / "10_sensitivity.png")
    return {"targets": targets, "strategies": strategies, "sensitivity": sens, "actions": actions}
