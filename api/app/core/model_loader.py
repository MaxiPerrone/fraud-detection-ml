import joblib
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
APP_DIR = CURRENT_DIR.parent
MODELS_DIR = APP_DIR / "models"

def load_models():
    """
    Carga los 3 modelos entrenados desde app/models.
    """
    return {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "svm": joblib.load(MODELS_DIR / "svm.pkl"),
    }


def load_scaler():
    """
    Carga el StandardScaler entrenado desde app/models/scaler.pkl.
    """
    return joblib.load(MODELS_DIR / "scaler.pkl")
