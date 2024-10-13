## Deploying ML Models

```
project-directory/
│
├── app/
│   ├── __init__.py  # Empty file
│   └── main.py      # FastAPI logic
│
├── model/
│   └── linear_regression_model.pkl  # Saved model (after running model_training.py)
│
├── model_training.py  # Model training code
├── requirements.txt  # Python dependencies
└── Dockerfile  # Docker configuration
```
In your project environment, create and activate a virtual environment:

```
$ python3 -m venv v1
$ source v1/bin/activate
```
Install these required packages using pip:

```
$ pip3 install pandas scikit-learn fastapi uvicorn
```

