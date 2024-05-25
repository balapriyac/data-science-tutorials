## Getting Started

Create and activate a dedicated venv for the project:

```bash
$ python3 -m venv v1
$ source v1/bin/activate
```
Install FastAPI and Uvicorn with `pip`:

```bash
$ pip3 install fastapi uvicorn
```
Also install scikit-learn:

```bash
$ pip3 install scikit-learn
```
Check [main.py](https://github.com/balapriyac/data-science-tutorials/blob/main/fastapi/main.py) for the complete code.

## Run the App

Run the following command:

```bash
$ uvicorn main:app --reload
```

## Query the `/predict/` Endpoint

Example POST request (using cURL):

```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'

```


