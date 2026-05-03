import pickle
import numpy as np

with open("model_v2.pkl", "rb") as f:
    model = pickle.load(f)

hours = float(input("Enter hours studied: "))

# model expects 2D input
X = np.array([[hours]])

prediction = model.predict(X)

if prediction[0] == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")