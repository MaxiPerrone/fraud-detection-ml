import numpy as np
from collections import Counter

def ensemble_predict(models, X):
    preds = [m.predict(X)[0] for m in models.values()]
    vote = Counter(preds).most_common(1)[0][0]
    return vote
