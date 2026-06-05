import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("movies.csv", encoding="latin1")

# Select required columns
df = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']]

# Remove rows with missing values
df = df.dropna()

# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Remove rows where Rating is missing
df = df.dropna(subset=['Rating'])

# Reset index
df = df.reset_index(drop=True)

# Label Encoding
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor1 = LabelEncoder()
le_actor2 = LabelEncoder()
le_actor3 = LabelEncoder()

df['Genre'] = le_genre.fit_transform(df['Genre'])
df['Director'] = le_director.fit_transform(df['Director'])
df['Actor 1'] = le_actor1.fit_transform(df['Actor 1'])
df['Actor 2'] = le_actor2.fit_transform(df['Actor 2'])
df['Actor 3'] = le_actor3.fit_transform(df['Actor 3'])

# Features and Target
X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = df['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)

print("\nMovie Rating Prediction Completed Successfully!")
print("Mean Absolute Error:", round(mae, 2))

# Sample Predictions
print("\nSample Predictions:")
for i in range(min(5, len(y_test))):
    print(
        f"Actual Rating: {y_test.iloc[i]:.1f} | Predicted Rating: {y_pred[i]:.1f}"
    )