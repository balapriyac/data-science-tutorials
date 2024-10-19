from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

def load_wine_data():
    wine_data = load_wine()
    df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
    df['target'] = wine_data.target  # Adding the target (wine quality class)
    return df

def preprocess_data(df):
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target (wine quality)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model using pickle
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    df = load_wine_data()
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)
