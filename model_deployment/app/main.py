# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# Define the input data schema using Pydantic
class InputData(BaseModel):
    MedInc: float
    AveRooms: float
    AveOccup: float

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")

# Load the model during startup
model_path = os.path.join("model", "linear_regression_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: InputData):
    # Prepare the data for prediction
    input_features = [[data.MedInc, data.AveRooms, data.AveOccup]]
    
    # Make prediction using the loaded model
    prediction = model.predict(input_features)
    
    # Return the prediction result
    return {"predicted_house_price": prediction[0]}
