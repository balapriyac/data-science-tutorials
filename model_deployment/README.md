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

Run the script to train the model and save it:

```
$ python3 model_training.py
```

You should be able to find the .pkl file (`linear_regression_model.pkl`) in the `model/` directory.

Use FastAPI to build an API to serve model predictions and containerize it using Docker.

### Building the Docker Image 

Build the Docker image by running the following `docker build` command:

```
$ docker build -t house-price-prediction-api .
```

Next run the Docker container:

```
$ docker run -d -p 80:80 house-price-prediction-api
```

### Tagging and Pushing the Image to Docker Hub

First, login to Docker Hub:

```
$ docker login
```

Tag the Docker image:

```
$ docker tag house-price-prediction-api your_username/house-price-prediction-api:v1
```

Push the image to Docker Hub:

```
$ docker push your_username/house-price-prediction-api:v1
```

Other developers can now pull and run the image like so: 

```
$ docker pull your_username/house-price-prediction-api:v1
$ docker run -d -p 80:80 your_username/house-price-prediction-api:v1
```




