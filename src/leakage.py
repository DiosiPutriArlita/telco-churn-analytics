import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src import config


def drop_post_outcome_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    present = [c for c in config.POST_OUTCOME_COLUMNS if c in df.columns]
    return df.drop(columns=present), present


def _oof_target_encoding(series: pd.Series, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    encoded = np.zeros(len(series), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prior = y.mean()
    for train_idx, valid_idx in splitter.split(series, y):
        means = pd.Series(y[train_idx]).groupby(series.iloc[train_idx].to_numpy()).mean()
        encoded[valid_idx] = series.iloc[valid_idx].map(means).fillna(prior).to_numpy()
    return encoded


def single_feature_auc(features: pd.DataFrame, y: np.ndarray, folds: int = 5,
                       seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rows = []
    for column in features.columns:
        series = features[column]
        if pd.api.types.is_numeric_dtype(series):
            score = series.fillna(series.median()).to_numpy(dtype=float)
        else:
            score = _oof_target_encoding(series.astype(str), y, folds, seed)
        auc = roc_auc_score(y, score)
        rows.append({"feature": column, "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def audit(df: pd.DataFrame, threshold: float = config.LEAKAGE_AUC_THRESHOLD) -> dict:
    cleaned, dropped = drop_post_outcome_columns(df)
    candidates = cleaned.drop(columns=[config.TARGET, "churn", config.ID_COLUMN], errors="ignore")
    table = single_feature_auc(candidates, cleaned["churn"].to_numpy())
    suspicious = table.loc[table["auc"] >= threshold, "feature"].tolist()
    return {
        "dataframe": cleaned.drop(columns=suspicious),
        "dropped_known": dropped,
        "dropped_suspicious": suspicious,
        "auc_table": table,
    }
