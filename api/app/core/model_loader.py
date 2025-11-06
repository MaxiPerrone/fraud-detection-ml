import joblib

def load_models():
    models = {
        "logistic_regression": joblib.load("models/logistic_regression.pkl"),
        "random_forest": joblib.load("models/random_forest.pkl"),
        "svm": joblib.load("models/svm.pkl")
    }
    return models
