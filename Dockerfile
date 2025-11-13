# =============================================================
#  TrueNest - MLflow + FastAPI Inference Image (GeoPandas-ready)
# =============================================================

FROM python:3.10

# Avoid interactive prompts during apt installs
ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------
# Install system dependencies for geopandas, shapely, fiona, gdal, pyproj
# ------------------------------------------------
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    proj-bin \
    proj-data \
    libproj-dev \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------
# Environment for GDAL, PROJ
# ------------------------------------------------
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# ------------------------------------------------
# Prevent buffering and enforce UTF-8
# ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

WORKDIR /app

# ------------------------------------------------
# Copy your inference code and model metadata
# ------------------------------------------------
COPY ./main.py /app/main.py
COPY ./handler.py /app/handler.py
COPY ./reports /app/reports
COPY ./requirements.txt /app/requirements.txt

# ------------------------------------------------
# Install Python dependencies (GeoPandas supported)
# ------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# ------------------------------------------------
# Hugging Face requires port 7860
# ------------------------------------------------
EXPOSE 7860

# ------------------------------------------------
# Create startup script
# ------------------------------------------------
RUN printf '%s\n' \
  '#!/bin/bash' \
  'set -e' \
  'echo "🚀 Starting TrueNest GeoPandas-enabled inference container"' \
  '' \
  'if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then' \
  '  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/creds.json' \
  '  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/creds.json' \
  '  echo "✅ GCP credentials configured"' \
  'else' \
  '  echo "⚠️ No GOOGLE_APPLICATION_CREDENTIALS_JSON provided. GCS access may fail."' \
  'fi' \
  '' \
  '# Preload MLflow model' \
  'python <<EOF' \
  'import mlflow, json' \
  'info = json.load(open("/app/reports/last_run_info.json"))' \
  'uri = info.get("pipeline_model_uri") or info.get("model_uri")' \
  'print(f"Loading MLflow model: {uri}")' \
  'mlflow.pyfunc.load_model(uri)' \
  'print("MLflow model loaded successfully")' \
  'EOF' \
  '' \
  'echo "🚀 Starting FastAPI server"' \
  'exec uvicorn main:app --host 0.0.0.0 --port 7860' \
  > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]

