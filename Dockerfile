FROM python:3.10-slim-bookworm

# Añadimos 'git' a la lista de instalación
RUN apt-get update && apt-get install -y \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Dependencias de sistema para OpenCV y Wget
RUN apt-get update && apt-get install -y \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar el modelo v8.4.0
RUN wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26x-seg.pt -O yoloe-26x-seg.pt

COPY handler.py .

# En Serverless no necesitas EXPOSE 8000, 
# la comunicación es interna vía la librería runpod.

# Ejecución directa del worker
CMD ["python", "-u", "handler.py"]
