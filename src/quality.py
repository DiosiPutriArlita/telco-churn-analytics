import pandas as pd

from src import config


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    blank = df.apply(lambda s: s.astype(str).str.strip().eq("").sum())
    report = pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "blank_string": blank,
            "dtype": df.dtypes.astype(str),
            "n_unique": df.nunique(),
        }
    )
    return report.sort_values(["missing", "blank_string"], ascending=False)


def rule_violations(df: pd.DataFrame) -> pd.DataFrame:
    total_blank = df["TotalCharges"].astype(str).str.strip().eq("")
    total_num = pd.to_numeric(df["TotalCharges"], errors="coerce")
    addons_active = df[config.ADDON_COLUMNS].eq("Yes").any(axis=1)

    rules = {
        "TotalCharges kosong": total_blank,
        "TotalCharges kosong dan tenure = 0": total_blank & df["tenure"].eq(0),
        "TotalCharges kosong dan tenure > 0": total_blank & df["tenure"].gt(0),
        "tenure di luar 0-72": ~df["tenure"].between(0, 72),
        "MonthlyCharges <= 0": df["MonthlyCharges"].le(0),
        "TotalCharges < MonthlyCharges (tenure > 0)": total_num.lt(df["MonthlyCharges"])
        & df["tenure"].gt(0),
        "Layanan tambahan aktif tanpa internet": df["InternetService"].eq("No") & addons_active,
        "MultipleLines aktif tanpa PhoneService": df["PhoneService"].eq("No")
        & df["MultipleLines"].eq("Yes"),
    }
    rows = [
        {"rule": name, "violations": int(mask.sum()), "share": float(mask.mean())}
        for name, mask in rules.items()
    ]
    return pd.DataFrame(rows)


def category_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in config.CATEGORICAL_COLUMNS + [config.TARGET]:
        if column not in df.columns:
            continue
        raw = df[column].astype(str)
        normalized = raw.str.strip().str.lower()
        rows.append(
            {
                "column": column,
                "levels": raw.nunique(),
                "levels_after_normalize": normalized.nunique(),
                "values": ", ".join(sorted(raw.unique())),
            }
        )
    return pd.DataFrame(rows)


def duplicate_report(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_id": int(df[config.ID_COLUMN].duplicated().sum()),
    }


def build_quality_report(df: pd.DataFrame) -> dict:
    return {
        "duplicates": duplicate_report(df),
        "missing": missing_report(df),
        "rules": rule_violations(df),
        "categories": category_inconsistencies(df),
    }
