import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.descriptive import add_tenure_band


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def categorical_tests(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        table = pd.crosstab(df[column], df["churn"])
        chi2, p_value, dof, _ = stats.chi2_contingency(table)
        rows.append(
            {"feature": column, "chi2": chi2, "dof": dof, "p_value": p_value,
             "cramers_v": cramers_v(table)}
        )
    result = pd.DataFrame(rows)
    result["p_bonferroni"] = (result["p_value"] * len(result)).clip(upper=1.0)
    return result.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_tests(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        churned = df.loc[df["churn"] == 1, column]
        retained = df.loc[df["churn"] == 0, column]
        u_stat, p_value = stats.mannwhitneyu(churned, retained, alternative="two-sided")
        effect = 1 - 2 * u_stat / (len(churned) * len(retained))
        rows.append(
            {"feature": column, "median_churned": churned.median(),
             "median_retained": retained.median(), "u_stat": u_stat,
             "p_value": p_value, "rank_biserial": -effect}
        )
    result = pd.DataFrame(rows)
    result["p_bonferroni"] = (result["p_value"] * len(result)).clip(upper=1.0)
    return result.sort_values("rank_biserial", key=np.abs, ascending=False).reset_index(drop=True)


def segment_combinations(df: pd.DataFrame, features: list[str], min_customers: int = 80,
                         top: int = 15) -> pd.DataFrame:
    base = df["churn"].mean()
    frames = []
    for pair in itertools.combinations(features, 2):
        grouped = df.groupby(list(pair))["churn"].agg(customers="size", churned="sum",
                                                      churn_rate="mean").reset_index()
        grouped["segment"] = grouped[pair[0]].astype(str) + " | " + grouped[pair[1]].astype(str)
        grouped["features"] = " x ".join(pair)
        frames.append(grouped[["features", "segment", "customers", "churned", "churn_rate"]])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["customers"] >= min_customers].copy()
    combined["lift"] = combined["churn_rate"] / base
    return combined.sort_values("churn_rate", ascending=False).head(top).reset_index(drop=True)


def addon_effect(df: pd.DataFrame) -> pd.DataFrame:
    internet = df[df["InternetService"] != "No"]
    rows = []
    for column in config.ADDON_COLUMNS:
        rates = internet.groupby(column)["churn"].mean()
        rows.append({"service": column,
                     "churn_with": rates.get("Yes", np.nan),
                     "churn_without": rates.get("No", np.nan)})
    result = pd.DataFrame(rows)
    result["difference"] = result["churn_with"] - result["churn_without"]
    return result.sort_values("difference")


def stratified_effect(df: pd.DataFrame, feature: str, stratum: str) -> pd.DataFrame:
    grouped = df.groupby([stratum, feature])["churn"].agg(customers="size", churn_rate="mean")
    return grouped.reset_index()


def plot_effect_sizes(table: pd.DataFrame, path):
    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top["feature"], top["cramers_v"], color="#4c78a8")
    ax.set_xlabel("Cramer's V")
    ax.set_title("Kekuatan asosiasi fitur kategorikal dengan Churn")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_contract_tenure_heatmap(df: pd.DataFrame, path):
    pivot = df.pivot_table(index="Contract", columns="tenure_band", values="churn", aggfunc="mean")
    pivot = pivot[[c for c in config.TENURE_LABELS if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    image = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=max(0.5, np.nanmax(pivot.values)))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns)
    ax.set_yticks(range(pivot.shape[0]), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center")
    ax.set_title("Churn rate: kontrak x lama berlangganan")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    banded = add_tenure_band(df)
    categorical = config.CATEGORICAL_COLUMNS + ["tenure_band"]
    cat_table = categorical_tests(banded, categorical)
    num_table = numeric_tests(banded, config.NUMERIC_COLUMNS)
    combos = segment_combinations(banded, ["Contract", "InternetService", "PaymentMethod",
                                           "tenure_band", "TechSupport", "PaperlessBilling"])
    addons = addon_effect(banded)
    contract_by_tenure = stratified_effect(banded, "Contract", "tenure_band")

    cat_table.to_csv(config.TABLES_DIR / "diagnostic_categorical_tests.csv", index=False)
    num_table.to_csv(config.TABLES_DIR / "diagnostic_numeric_tests.csv", index=False)
    combos.to_csv(config.TABLES_DIR / "diagnostic_top_segments.csv", index=False)
    addons.to_csv(config.TABLES_DIR / "diagnostic_addon_effect.csv", index=False)
    contract_by_tenure.to_csv(config.TABLES_DIR / "diagnostic_contract_by_tenure.csv", index=False)

    plot_effect_sizes(cat_table, config.FIGURES_DIR / "04_effect_sizes.png")
    plot_contract_tenure_heatmap(banded, config.FIGURES_DIR / "05_contract_tenure_heatmap.png")
    return {"categorical": cat_table, "numeric": num_table, "segments": combos, "addons": addons}
