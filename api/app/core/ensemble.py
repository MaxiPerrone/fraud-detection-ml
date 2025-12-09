from collections import Counter

def ensemble_predict(models: dict, X_df):
    """
    Recibe un dict de modelos (sin scaler) y un X (1 fila ya escalada).
    Hace majority vote sobre las predicciones.
    """
    preds = [model.predict(X_df)[0] for model in models.values()]
    vote = Counter(preds).most_common(1)[0][0]
    return vote
