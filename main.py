import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [20, 30, 40, 50, 60]
})

X = data[["Hours"]]
y = data["Marks"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict marks for 6 study hours
prediction = model.predict(pd.DataFrame({"Hours": [6]}))

print("Predicted marks:", prediction[0])
