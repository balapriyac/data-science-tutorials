from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Load the saved model
model = joblib.load('linear_regression_model.pkl')

# Create the FastAPI app
app = FastAPI()

# Define the input schema for the API
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveOccup: float

# Define the prediction endpoint
@app.post('/predict')
def predict_price(data: HousingData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return the predicted price
    return {"Predicted Price": prediction[0]}
