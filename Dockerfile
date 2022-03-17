# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

ENV WORKERS=1
ENV THREADS=8
ENV TIMEOUT=900
ENV MODELS_PATH='./models'

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.

COPY requirements.txt .

RUN pip install -r requirements.txt

# Don't download the model. It's better to get it once and mount it multiple times.
# RUN python model.py

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.

EXPOSE 8080

ENTRYPOINT gunicorn --bind 0.0.0.0:8080 --workers ${WORKERS} --threads ${THREADS} app:app --timeout ${TIMEOUT}
