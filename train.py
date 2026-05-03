# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import pickle
# import random
# from sklearn.metrics import accuracy_score



# print("Accuracy:", acc)
# data = pd.read_csv("data_v2.csv")
# X = data[["hours"]]
# y = data["pass"]

# random.seed(45)
# # X = np.random.rand(100, 1)
# # y = 3 * X + np.random.rand(100, 1)
# model = LinearRegression()
# model.fit(X, y)
# print("done")

# with open("model_v2.pkl", "wb") as f:
#     pickle.dump(model, f)
    
# with open("metrics.txt", "a") as f:
#     f.write(f"Model: {MODEL_PATH}, Accuracy: {acc}\n")
    
# preds = model.predict(X)
# acc = accuracy_score(y, preds)


# print("Model trained and saved on real data!")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import random

# -------- CONFIG --------
DATA_PATH = "data_v2.csv"
MODEL_PATH = "model_v2.pkl"
BEST_METRIC_FILE = "best_metrics.txt"

# -------- LOAD DATA --------
data = pd.read_csv(DATA_PATH)
X = data[["hours"]]
y = data["pass"]
# random.seed(45)
# -------- TRAIN --------
model = LogisticRegression()
model.fit(X, y)

# -------- EVALUATE --------
preds = model.predict(X)
acc = accuracy_score(y, preds)

print("Current Accuracy:", acc)

# -------- LOAD BEST --------
best_acc = 0

if os.path.exists(BEST_METRIC_FILE):
    with open(BEST_METRIC_FILE, "r") as f:
        best_acc = float(f.read())

print("Best Accuracy:", best_acc)

# -------- DECISION --------
if acc > best_acc:
    print("New model is better. Saving...")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(BEST_METRIC_FILE, "w") as f:
        f.write(str(acc))

else:
    print("Model not better. Skipping save.")

# -------- LOGGING --------
with open("metrics.txt", "a") as f:
    f.write(f"{MODEL_PATH} -> {acc}\n")

print("Done")