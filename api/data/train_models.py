import pickle
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).resolve().parent 
API_DIR = DATA_DIR.parent
APP_DIR = API_DIR / "app"
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "X_test.pkl", "rb") as f:
    X = pickle.load(f)

with open(DATA_DIR / "y_test.pkl", "rb") as f:
    y = pickle.load(f)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)


# Logistic Regression
log_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_valid_scaled)
print(f"Logistic regression accuracy: {accuracy_score(y_valid, log_pred):.3f}")

# Random Forest
rf_model = RandomForestClassifier(
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_valid)
print(f"Random forest accuracy: {accuracy_score(y_valid, rf_pred):.3f}")

sample_size = min(40000, len(X_train_scaled))
X_svm_train = X_train_scaled[:sample_size]
y_svm_train = y_train.iloc[:sample_size]


svm_classifier = SVC(kernel = "linear", probability=True, random_state=42)
calibrated_svm = CalibratedClassifierCV(svm_classifier)
calibrated_svm.fit(X_svm_train, y_svm_train)

calibrated_svm.fit(X_svm_train, y_svm_train)
svm_pred = calibrated_svm.predict(X_valid_scaled)
print(f"SVM accuracy: {accuracy_score(y_valid, svm_pred):.3f}")

joblib.dump(log_model, MODELS_DIR / "logistic_regression.pkl")
joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
joblib.dump(calibrated_svm, MODELS_DIR / "svm.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

print(f"Modelos y scaler guardados en {MODELS_DIR}")
