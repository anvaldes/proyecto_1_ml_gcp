FROM python:3.10-slim

WORKDIR /app

# Instalar build tools necesarios para compilar paquetes como pickle5
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gcsfs joblib

COPY . .

ENTRYPOINT ["python"]
