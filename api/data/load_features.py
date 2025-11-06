import pandas as pd
import kagglehub
import os
import pickle

dataset_path = kagglehub.dataset_download("dhanushnarayananr/credit-card-fraud")
csv_file = os.path.join(dataset_path, "card_transdata.csv")

df = pd.read_csv(csv_file)

X = df.drop("fraud", axis=1)
y = df["fraud"].copy()

with open("X_test.pkl", "wb") as f:
    pickle.dump(X, f)
print("X_test saved to 'X_test.pkl'.")

with open("y_test.pkl", "wb") as f:
    pickle.dump(y, f)
print("y_test saved to 'y_test.pkl'.")