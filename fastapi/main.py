# Create a FastAPI app
# Root endpoint returns the app description

from fastapi import FastAPI

app = FastAPI()

# Define a function to return a description of the app
def get_app_description():
    return (
        "Welcome to the Iris Species Prediction API!"
        "This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
        "Use the '/predict/' endpoint with a POST request to make predictions."
        "Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
    )

# Define the root endpoint to return the app description
@app.get("/")
async def root():
    return {"message": get_app_description()}


# Build a logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    return iris.target_names[prediction[0]]

# Define the Pydantic model for your input data
from pydantic import BaseModel

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create API endpoint
@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
    species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
    return {"species": species}

