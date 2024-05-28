## Create Minimal Docker Image for a Sample Python Application

Your project directory should look like so:

```
<project-dir>/
├── app.py
├── Dockerfile
├── requirements.txt
```

Build the docker image for the Flask app:

```sh
$ docker build -t inventory-app:slim .
```
