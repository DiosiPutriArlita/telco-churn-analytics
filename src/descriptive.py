import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tenure_band"] = pd.cut(
        out["tenure"], bins=config.TENURE_BINS, labels=config.TENURE_LABELS
    ).astype(str)
    return out


def overview(df: pd.DataFrame) -> dict:
    return {
        "customers": len(df),
        "churned": int(df["churn"].sum()),
        "churn_rate": float(df["churn"].mean()),
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()),
        "avg_monthly_charges_churned": float(df.loc[df["churn"] == 1, "MonthlyCharges"].mean()),
        "avg_monthly_charges_retained": float(df.loc[df["churn"] == 0, "MonthlyCharges"].mean()),
        "median_tenure_churned": float(df.loc[df["churn"] == 1, "tenure"].median()),
        "median_tenure_retained": float(df.loc[df["churn"] == 0, "tenure"].median()),
        "monthly_revenue_at_risk": float(df.loc[df["churn"] == 1, "MonthlyCharges"].sum()),
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("churn")[config.NUMERIC_COLUMNS].describe().T


def churn_by_segment(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    base = df["churn"].mean()
    frames = []
    for column in columns:
        grouped = df.groupby(column)["churn"].agg(customers="size", churned="sum", churn_rate="mean")
        grouped["lift"] = grouped["churn_rate"] / base
        grouped.insert(0, "feature", column)
        grouped = grouped.reset_index().rename(columns={column: "level"})
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def plot_class_balance(df: pd.DataFrame, path=None):
    counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4c78a8", "#e45756"])
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value}\n({value / counts.sum():.1%})",
                ha="center", va="bottom")
    ax.set_ylabel("Pelanggan")
    ax.set_title("Distribusi kelas Churn")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(path or config.FIGURES_DIR / "01_class_balance.png", dpi=150)
    plt.close(fig)


def plot_rate_by_feature(df: pd.DataFrame, column: str, path):
    rates = df.groupby(column)["churn"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(rates.index.astype(str), rates.values, color="#e45756")
    ax.axhline(df["churn"].mean(), color="black", linestyle="--", linewidth=1, label="rata-rata")
    ax.set_ylabel("Churn rate")
    ax.set_title(f"Churn rate berdasarkan {column}")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_numeric_distributions(df: pd.DataFrame, path=None):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, config.NUMERIC_COLUMNS):
        for label, color in ((0, "#4c78a8"), (1, "#e45756")):
            ax.hist(df.loc[df["churn"] == label, column], bins=30, alpha=0.6, density=True,
                    color=color, label="Churn" if label else "Bertahan")
        ax.set_title(column)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path or config.FIGURES_DIR / "03_numeric_distributions.png", dpi=150)
    plt.close(fig)


def run(df: pd.DataFrame) -> dict:
    banded = add_tenure_band(df)
    segment_columns = ["Contract", "InternetService", "PaymentMethod", "tenure_band",
                       "PaperlessBilling", "SeniorCitizen", "TechSupport"]
    segments = churn_by_segment(banded, segment_columns)
    segments.to_csv(config.TABLES_DIR / "descriptive_churn_by_segment.csv", index=False)
    numeric_summary(df).to_csv(config.TABLES_DIR / "descriptive_numeric_summary.csv")

    plot_class_balance(df)
    plot_rate_by_feature(banded, "Contract", config.FIGURES_DIR / "02_churn_by_contract.png")
    plot_rate_by_feature(banded, "tenure_band", config.FIGURES_DIR / "02_churn_by_tenure_band.png")
    plot_numeric_distributions(df)
    return {"overview": overview(df), "segments": segments}
