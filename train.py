import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import random


data = pd.read_csv("data.csv")
X = data[["hours"]]
y = data["pass"]

random.seed(45)
# X = np.random.rand(100, 1)
# y = 3 * X + np.random.rand(100, 1)
model = LinearRegression()
model.fit(X, y)
print("done")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved on real data!")