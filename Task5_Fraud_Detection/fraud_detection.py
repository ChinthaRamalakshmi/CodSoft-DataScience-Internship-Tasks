import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv("creditcard.csv")
except FileNotFoundError:
    print("Dataset not found.")
    print("Please download 'creditcard.csv' from the Kaggle dataset link provided in README.md and place it in this folder.")
    exit()

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Features & target
X = df.drop('Class', axis=1)
y = df['Class']

# IMPORTANT: scale Amount + Time
scaler = StandardScaler()
X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

# Train-test split (stratify important)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()
