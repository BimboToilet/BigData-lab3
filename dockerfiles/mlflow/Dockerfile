FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000