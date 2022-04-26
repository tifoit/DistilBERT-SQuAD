<!-- Space: PDP -->
<!-- Parent: Technical Documentation -->
<!-- Parent: Distilbert Squad -->
<!-- Title: Distilbert Squad -->

# Distilbert Squad

The functionalities that can be found in this repository are:
* Add vocabulary to a hugging face model
* Question and answering using text extraction
* Masked language modeling training
* Get from text words not found in a model's vocabulary with IDF


# Dependencies

Dependencies for this repository can be found in the requirements.txt file, library names along with their required versions. Before attempting to run app.py make sure that every requirement found in the file are satisfied.
These requirements can be installed with the command: 

```bash
pip install -r requirements.txt
```

# Start Up the Service

Execute the `app.py` script. At startup, this will load all models found in a specific path. 

NOTE: This path defaults to `./models` but it can be overriden with an environment variable.

# Docker

## Models

Models need to be downloaded to the `models` folder within the project (at the same level as the `Dockerfile`). These will be copied to the image.

NOTE: This is acceptable for local Docker testing only. When running this in a Kubernetes cluster, models will be loaded to a NFS and the pod will read them from there.



## Build the container

```bash
docker build -t distilbert-squad-flask .
```

## Run the container

```bash
docker run -dp 8080:8080 -e WORKERS=2 -e THREADS=8 -e TIMEOUT=900 -e MODELS_PATH="./models" distilbert-squad-flask
```

# Run Tests

This repository currently has no formal tests.