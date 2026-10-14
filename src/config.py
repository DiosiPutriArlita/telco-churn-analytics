from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"
MODELS_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD_BETA = 2.0

TARGET = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
ADDON_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    *ADDON_COLUMNS,
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

POST_OUTCOME_COLUMNS = [
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Churn Category",
    "CLTV",
    "Customer Status",
]
LEAKAGE_AUC_THRESHOLD = 0.90

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]


@dataclass(frozen=True)
class CampaignParams:
    offer_cost: float = 15.0
    success_rate: float = 0.30
    horizon_months: int = 12
    gross_margin: float = 0.40
    budget_share: float = 0.20
