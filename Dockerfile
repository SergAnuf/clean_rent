# =============================================================
# TrueNest - MLflow + FastAPI Inference Image (GeoPandas-ready)
# =============================================================

FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------
# Install system deps for GeoPandas stack
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

ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8

WORKDIR /app

# ------------------------------------------------
# Copy inference code & metadata
# ------------------------------------------------
COPY ./main.py /app/main.py
COPY ./handler.py /app/handler.py
COPY ./reports /app/reports
COPY ./requirements.txt /app/requirements.txt

# ------------------------------------------------
# NEW — Copy source code for MLflow unpickling
# ------------------------------------------------
COPY ./src /app/src
ENV PYTHONPATH="/app"

# ------------------------------------------------
# Install Python deps
# ------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir uvicorn fastapi python-dotenv
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 7860

# ------------------------------------------------
# Startup script with GCP credential fix
# ------------------------------------------------
RUN printf '%s\n' \
  '#!/bin/bash' \
  'set -e' \
  'echo "🚀 Starting TrueNest GeoPandas-enabled inference container"' \
  '' \
  '# -------------------------------' \
  '# Configure GCP credentials' \
  '# -------------------------------' \
  'if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then' \
  '  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/creds.json' \
  '  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/creds.json' \
  '  export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/creds.json' \
  '  echo "✅ GCP credentials written to /tmp/creds.json"' \
  'else' \
  '  echo "⚠️ No GOOGLE_APPLICATION_CREDENTIALS_JSON provided. GCS access will fail."' \
  'fi' \
  '' \
  '# -------------------------------' \
  '# Force GCP credential load BEFORE MLflow loads model' \
  'echo "🔐 Verifying GCP credentials..."' \
  'python <<EOF' \
  'import os' \
  'from google.oauth2 import service_account' \
  'from google.cloud import storage' \
  'p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")' \
  'if not p or not os.path.exists(p):' \
  '    print("❌ No valid Google credentials found:", p)' \
  'else:' \
  '    creds = service_account.Credentials.from_service_account_file(p)' \
  '    client = storage.Client(credentials=creds)' \
  '    try:' \
  '        client.list_buckets(max_results=1)' \
  '        print("🔐 GCP credentials verified successfully")' \
  '    except Exception as e:' \
  '        print("❌ GCP credential test failed:", e)' \
  'EOF' \
  '' \
  '# -------------------------------' \
  '# Preload MLflow pipeline model' \
  '# -------------------------------' \
  'python <<EOF' \
  'import mlflow, json' \
  'info = json.load(open("/app/reports/last_run_info.json"))' \
  'uri = info.get("pipeline_model_uri") or info.get("model_uri")' \
  'print(f"📦 Loading MLflow model: {uri}")' \
  'mlflow.pyfunc.load_model(uri)' \
  'print("✅ MLflow model loaded successfully")' \
  'EOF' \
  '' \
  'echo "🚀 Launching FastAPI server"' \
  'exec uvicorn main:app --host 0.0.0.0 --port 7860' \
  > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
