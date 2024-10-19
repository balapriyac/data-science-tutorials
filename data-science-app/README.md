## Building a Simple Data Science App

Install the required libraries in a virtual environment for the project:
```
$ pip3 install fastapi uvicorn scikit-learn pandas
```

Run `model_training.py` to train the logistic regression model.

To run the FastAPI app, use:
```
$ uvicorn app:app --reload
```
Use curl to send POST requests to the `/predict` endpoint.
