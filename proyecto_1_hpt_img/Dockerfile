FROM python:3.10-slim

WORKDIR /app

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

USER 0

# Configurar como entrypoint
ENTRYPOINT ["python", "train.py"]

