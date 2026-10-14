import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src import config

ENGINEERED_NUMERIC = ["n_addons", "price_ratio"]
ENGINEERED_CATEGORICAL = ["autopay"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["n_addons"] = out[config.ADDON_COLUMNS].eq("Yes").sum(axis=1)
    out["autopay"] = np.where(out["PaymentMethod"].str.contains("automatic", case=False), "Yes", "No")
    historical = out["TotalCharges"] / out["tenure"].replace(0, np.nan)
    out["price_ratio"] = (out["MonthlyCharges"] / historical).replace([np.inf, -np.inf], np.nan)
    out["price_ratio"] = out["price_ratio"].fillna(1.0)
    return out


def feature_lists() -> tuple[list[str], list[str]]:
    numeric = config.NUMERIC_COLUMNS + ENGINEERED_NUMERIC
    categorical = config.CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL
    return numeric, categorical


def build_preprocessor() -> ColumnTransformer:
    numeric, categorical = feature_lists()
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        verbose_feature_names_out=False,
    )
