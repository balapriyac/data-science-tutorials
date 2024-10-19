from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "A Simple Prediction API"}

def load_model():
    with open('model/classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

class WineFeatures(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.post("/predict")
def predict_wine(features: WineFeatures):
    model = load_model()
    input_data = [[
        features.alcohol, features.malic_acid, features.ash, features.alcalinity_of_ash,
        features.magnesium, features.total_phenols, features.flavanoids,
        features.nonflavanoid_phenols, features.proanthocyanins, features.color_intensity,
        features.hue, features.od280_od315_of_diluted_wines, features.proline
    ]]
    
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}
