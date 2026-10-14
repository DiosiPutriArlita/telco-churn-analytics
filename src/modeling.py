import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor


def make_models() -> dict[str, Pipeline]:
    seed = config.RANDOM_STATE
    estimators = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg_unweighted": LogisticRegression(max_iter=2000, random_state=seed),
        "logreg_balanced": LogisticRegression(max_iter=2000, class_weight="balanced",
                                              random_state=seed),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "hist_gradient_boosting_balanced": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_depth=4, l2_regularization=1.0,
            class_weight="balanced", random_state=seed),
    }
    return {
        name: Pipeline([("prep", build_preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def split(features: pd.DataFrame, y: np.ndarray):
    return train_test_split(features, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def cross_validate_models(models: dict[str, Pipeline], X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, pipeline in models.items():
        result = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for metric in scoring:
            values = result[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)


def best_model_name(cv_table: pd.DataFrame) -> str:
    candidates = cv_table[cv_table["model"] != "dummy_majority"]
    return candidates.iloc[0]["model"]


def calibrated(pipeline: Pipeline) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=config.CV_FOLDS)


def out_of_fold_probabilities(model, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE + 1)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def tune_threshold(y: np.ndarray, probabilities: np.ndarray, beta: float = config.THRESHOLD_BETA) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y, probabilities >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "f2": fbeta_score(y, predicted, beta=2.0, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def compare_calibration(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    raw = pipeline.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    cal = calibrated(pipeline).fit(X_train, y_train).predict_proba(X_test)[:, 1]
    rows = []
    for label, probs in (("uncalibrated", raw), ("calibrated_sigmoid", cal)):
        rows.append({"variant": label, "brier": brier_score_loss(y_test, probs),
                     "mean_predicted": probs.mean(), "observed_rate": y_test.mean()})
    return pd.DataFrame(rows)


def feature_importance(model, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring="average_precision", n_repeats=10,
                                    random_state=config.RANDOM_STATE, n_jobs=1)
    table = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean,
                          "std": result.importances_std})
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_curves(y: np.ndarray, probabilities: np.ndarray, path):
    fpr, tpr, _ = roc_curve(y, probabilities)
    precision, recall, _ = precision_recall_curve(y, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#4c78a8")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                title=f"ROC (AUC = {roc_auc_score(y, probabilities):.3f})")
    axes[1].plot(recall, precision, color="#e45756")
    axes[1].axhline(y.mean(), color="black", linestyle="--", linewidth=1, label="baseline")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall (AP = {average_precision_score(y, probabilities):.3f})")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y: np.ndarray, raw: np.ndarray, calibrated_probs: np.ndarray, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, probs, color in (("sebelum kalibrasi", raw, "#999999"),
                                ("sesudah kalibrasi", calibrated_probs, "#e45756")):
        frac, mean_pred = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="Probabilitas prediksi", ylabel="Proporsi churn aktual", title="Kurva kalibrasi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(table: pd.DataFrame, path, top: int = 12):
    top_rows = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(top_rows["feature"], top_rows["importance"], xerr=top_rows["std"], color="#4c78a8")
    ax.set_xlabel("Penurunan PR-AUC saat fitur diacak")
    ax.set_title("Permutation importance (data uji)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_model(model, name: str):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODELS_DIR / f"{name}.joblib")
