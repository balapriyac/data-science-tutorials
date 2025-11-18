from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np

app = FastAPI(
    title="Wine Quality Predictor",
    description="Predict wine quality based on chemical properties",
    version="1.0.0"
)

# Load model and scaler at startup
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Model files not found. Please train the model first.")

# Wine class names
wine_classes = ['Class 0', 'Class 1', 'Class 2']


class WineFeatures(BaseModel):
    alcohol: float = Field(..., description="Alcohol content (%)")
    malic_acid: float = Field(..., description="Malic acid (g/L)")
    ash: float = Field(..., description="Ash content (g/L)")
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash")
    magnesium: float = Field(..., description="Magnesium (mg/L)")
    total_phenols: float = Field(..., description="Total phenols")
    flavanoids: float = Field(..., description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., description="Non-flavanoid phenols")
    proanthocyanins: float = Field(..., description="Proanthocyanins")
    color_intensity: float = Field(..., description="Color intensity")
    hue: float = Field(..., description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., description="OD280/OD315 of diluted wines")
    proline: float = Field(..., description="Proline (mg/L)")

    class Config:
        json_schema_extra = {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 2.77,
                "ash": 2.51,
                "alcalinity_of_ash": 18.5,
                "magnesium": 96.0,
                "total_phenols": 2.45,
                "flavanoids": 2.53,
                "nonflavanoid_phenols": 0.29,
                "proanthocyanins": 1.54,
                "color_intensity": 5.0,
                "hue": 1.04,
                "od280_od315_of_diluted_wines": 3.47,
                "proline": 920.0
            }
        }


@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Wine Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict")
def predict(features: WineFeatures):
    """
    Predict wine quality based on chemical properties
    
    Returns the predicted class, confidence score, and probabilities for all classes
    """
    try:
        # Convert input to array
        input_data = np.array([[
            features.alcohol,
            features.malic_acid,
            features.ash,
            features.alcalinity_of_ash,
            features.magnesium,
            features.total_phenols,
            features.flavanoids,
            features.nonflavanoid_phenols,
            features.proanthocyanins,
            features.color_intensity,
            features.hue,
            features.od280_od315_of_diluted_wines,
            features.proline
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        return {
            "prediction": wine_classes[prediction[0]],
            "prediction_index": int(prediction[0]),
            "confidence": float(probabilities[0][prediction[0]]),
            "all_probabilities": {
                wine_classes[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


