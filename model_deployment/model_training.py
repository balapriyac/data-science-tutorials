# model_training.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load the dataset
data = fetch_california_housing(as_frame=True)
df = data['data']
target = data['target']

# Select a few features
selected_features = ['MedInc', 'AveRooms', 'AveOccup']
X = df[selected_features]
y = target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Create a 'model' folder to save the trained model
os.makedirs('model', exist_ok=True)

# Save the trained model using pickle
with open('model/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")
