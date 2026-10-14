import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src import config, data_loader, descriptive, diagnostic, features, leakage, modeling
from src import prescriptive, quality
from src.config import CampaignParams

logger = logging.getLogger("pipeline")


def prepare_directories():
    for directory in (config.FIGURES_DIR, config.TABLES_DIR, config.MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def stage_quality(raw: pd.DataFrame):
    report = quality.build_quality_report(raw)
    report["missing"].to_csv(config.TABLES_DIR / "quality_missing.csv")
    report["rules"].to_csv(config.TABLES_DIR / "quality_rules.csv", index=False)
    report["categories"].to_csv(config.TABLES_DIR / "quality_categories.csv", index=False)
    logger.info("Duplikat: %s", report["duplicates"])
    logger.info("Pelanggaran aturan:\n%s", report["rules"].to_string(index=False))


def stage_leakage(df: pd.DataFrame) -> pd.DataFrame:
    result = leakage.audit(df)
    result["auc_table"].to_csv(config.TABLES_DIR / "leakage_single_feature_auc.csv", index=False)
    logger.info("Kolom pasca-outcome dibuang: %s", result["dropped_known"])
    logger.info("Kolom mencurigakan dibuang: %s", result["dropped_suspicious"])
    logger.info("AUC satu fitur tertinggi:\n%s", result["auc_table"].head(5).to_string(index=False))
    return result["dataframe"]


def stage_predictive(df: pd.DataFrame):
    engineered = features.engineer(df)
    X, y = data_loader.split_features_target(engineered)
    X_train, X_test, y_train, y_test = modeling.split(X, y)
    logger.info("Train %d, test %d, churn rate train %.3f, test %.3f",
                len(X_train), len(X_test), y_train.mean(), y_test.mean())

    models = modeling.make_models()
    cv_table = modeling.cross_validate_models(models, X_train, y_train)
    cv_table.to_csv(config.TABLES_DIR / "predictive_cv_results.csv", index=False)
    logger.info("Hasil cross-validation:\n%s", cv_table[[
        "model", "accuracy_mean", "recall_mean", "precision_mean", "roc_auc_mean", "pr_auc_mean"
    ]].round(3).to_string(index=False))

    best_name = modeling.best_model_name(cv_table)
    logger.info("Model terpilih (PR-AUC CV): %s", best_name)
    pipeline = modeling.make_models()[best_name]

    calibration = modeling.compare_calibration(pipeline, X_train, y_train, X_test, y_test)
    calibration.to_csv(config.TABLES_DIR / "predictive_calibration.csv", index=False)
    logger.info("Kalibrasi:\n%s", calibration.round(4).to_string(index=False))

    model = modeling.calibrated(modeling.make_models()[best_name]).fit(X_train, y_train)
    oof = modeling.out_of_fold_probabilities(modeling.calibrated(modeling.make_models()[best_name]),
                                             X_train, y_train)
    threshold = modeling.tune_threshold(y_train, oof)
    probabilities = model.predict_proba(X_test)[:, 1]

    baseline = modeling.make_models()["dummy_majority"].fit(X_train, y_train)
    rows = {
        "baseline_majority": modeling.evaluate(y_test, baseline.predict_proba(X_test)[:, 1], 0.5),
        f"{best_name}_threshold_0.5": modeling.evaluate(y_test, probabilities, 0.5),
        f"{best_name}_threshold_tuned": modeling.evaluate(y_test, probabilities, threshold),
    }
    metrics = pd.DataFrame(rows).T
    metrics.to_csv(config.TABLES_DIR / "predictive_test_metrics.csv")
    logger.info("Metrik data uji:\n%s", metrics[[
        "threshold", "accuracy", "precision", "recall", "f2", "roc_auc", "pr_auc", "brier"
    ]].round(3).to_string())

    raw_probs = modeling.make_models()[best_name].fit(X_train, y_train).predict_proba(X_test)[:, 1]
    modeling.plot_curves(y_test, probabilities, config.FIGURES_DIR / "06_roc_pr_curves.png")
    modeling.plot_calibration(y_test, raw_probs, probabilities, config.FIGURES_DIR / "07_calibration.png")
    importance = modeling.feature_importance(model, X_test, y_test)
    importance.to_csv(config.TABLES_DIR / "predictive_feature_importance.csv", index=False)
    modeling.plot_importance(importance, config.FIGURES_DIR / "08_feature_importance.png")
    modeling.save_model(model, "churn_model")
    return {"X_test": X_test, "y_test": y_test, "probabilities": probabilities,
            "threshold": threshold, "best_name": best_name}


def stage_prescriptive(df: pd.DataFrame, predictive: dict, params: CampaignParams):
    X_test = predictive["X_test"]
    ids = df.loc[X_test.index, config.ID_COLUMN]
    frame = prescriptive.build_scoring_frame(X_test, predictive["probabilities"], ids,
                                             predictive["y_test"], params)
    result = prescriptive.run(frame, params)
    logger.info("Pelanggan ditargetkan: %d dari %d", len(result["targets"]), len(frame))
    logger.info("Perbandingan strategi:\n%s", result["strategies"].round(1).to_string(index=False))
    logger.info("Ringkasan aksi:\n%s", result["actions"].round(2).to_string(index=False))
    return result


def main(data_path: Path = config.DATA_PATH, params: CampaignParams = CampaignParams()):
    prepare_directories()
    raw = data_loader.load_raw(data_path)
    logger.info("Data dimuat: %d baris, %d kolom", *raw.shape)

    stage_quality(raw)
    df = data_loader.clean(raw)
    df = stage_leakage(df)

    overview = descriptive.run(df)["overview"]
    logger.info("Ringkasan deskriptif: %s", json.dumps(overview, indent=2))

    diag = diagnostic.run(df)
    logger.info("Asosiasi terkuat:\n%s", diag["categorical"].head(6)[
        ["feature", "cramers_v", "p_bonferroni"]].round(4).to_string(index=False))
    logger.info("Segmen berisiko tertinggi:\n%s", diag["segments"].head(5).round(3).to_string(index=False))

    predictive = stage_predictive(df)
    stage_prescriptive(df, predictive, params)


def parse_args():
    parser = argparse.ArgumentParser(description="Analisis churn pelanggan Telco")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--offer-cost", type=float, default=CampaignParams.offer_cost)
    parser.add_argument("--success-rate", type=float, default=CampaignParams.success_rate)
    parser.add_argument("--horizon-months", type=int, default=CampaignParams.horizon_months)
    parser.add_argument("--gross-margin", type=float, default=CampaignParams.gross_margin)
    parser.add_argument("--budget-share", type=float, default=CampaignParams.budget_share)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    main(args.data, CampaignParams(args.offer_cost, args.success_rate, args.horizon_months,
                                   args.gross_margin, args.budget_share))
