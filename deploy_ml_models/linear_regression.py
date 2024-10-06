import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the California Housing dataset
data = fetch_california_housing(as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Define features and target variable
X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]  # Selecting some example features
y = df['target']  # Target variable is the housing price (scaled)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
score = model.score(X_test, y_test)
print(f"Model R-squared: {score:.4f}")

# Save the model using pickle
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)
