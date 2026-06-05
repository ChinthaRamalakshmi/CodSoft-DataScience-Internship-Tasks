# 📊 Sales Prediction using Simple Linear Regression
# CodSoft Internship - Task 4

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 📌 Load dataset
df = pd.read_csv("advertising.csv")

print("Dataset Head:")
print(df.head())

# 📌 Features & Target
X = df[['TV']]   # only TV (Simple Linear Regression)
y = df['Sales']

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100
)

# 📌 Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Prediction
y_pred = model.predict(X_test)

# 📌 Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# 📌 Model equation
print("\n📌 Model Equation:")
print("Sales =", model.intercept_, "+", model.coef_[0], "* TV")

# 📌 Sample predictions
print("\n🎯 Sample Predictions:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {y_pred[i]:.2f}")

# 📌 Visualization
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.title("TV vs Sales Prediction")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.legend()
plt.show()