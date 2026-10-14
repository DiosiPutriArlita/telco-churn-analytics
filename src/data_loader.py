import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def download_dataset(path: Path = config.DATA_PATH, url: str = config.DATA_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_raw(path: Path = config.DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        download_dataset(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=config.ID_COLUMN, keep="first")

    total = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    new_customer = out["tenure"].eq(0)
    total = total.where(~(new_customer & total.isna()), 0.0)
    fallback = out["tenure"] * out["MonthlyCharges"]
    out["TotalCharges"] = total.fillna(fallback).astype(float)

    if pd.api.types.is_numeric_dtype(out["SeniorCitizen"]):
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    for column in config.ADDON_COLUMNS:
        out[column] = out[column].replace({"No internet service": "No"})
    out["MultipleLines"] = out["MultipleLines"].replace({"No phone service": "No"})

    for column in config.CATEGORICAL_COLUMNS + [config.TARGET]:
        out[column] = out[column].astype(str).str.strip()

    out["churn"] = (out[config.TARGET] == "Yes").astype(int)
    return out.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    features = df.drop(columns=[config.TARGET, "churn", config.ID_COLUMN], errors="ignore")
    return features, df["churn"].to_numpy(dtype=np.int64)
