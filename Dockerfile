# =============================================================
#  TrueNest - MLflow + FastAPI Inference Image
#  Author: Dr. Sergii Anufriev
# =============================================================

FROM python:3.10-slim

# Prevent buffering and enforce UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8

# ------------------------------------------------
# Set working directory
# ------------------------------------------------
WORKDIR /app

# ------------------------------------------------
# Copy FastAPI inference files (flat structure)
# ------------------------------------------------
COPY ./main.py /app/main.py
COPY ./handler.py /app/handler.py
COPY ./reports /app/reports

# ------------------------------------------------
# Copy requirements.txt (bundled in deploy/)
# ------------------------------------------------
COPY ./requirements.txt /app/requirements.txt

# ------------------------------------------------
# Install dependencies
# ------------------------------------------------
RUN pip install --no-cache-dir -r /app/requirements.txt

# ------------------------------------------------
# Hugging Face Spaces expects port 7860
# ------------------------------------------------
EXPOSE 7860

# ------------------------------------------------
# Startup script: configure secrets, load MLflow model, run app
# ------------------------------------------------
RUN echo '#!/bin/bash
set -e

echo "🚀 Starting TrueNest inference container"

# --- Setup Google Cloud credentials ---
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/creds.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/creds.json
  echo "✅ GCP credentials configured"
else
  echo "⚠️ No GOOGLE_APPLICATION_CREDENTIALS_JSON provided. GCS access may fail."
fi

# --- Preload MLflow model from GCS ---
python <<EOF
import mlflow, json
info = json.load(open("/app/reports/last_run_info.json"))
uri = info.get("pipeline_model_uri") or info.get("model_uri")

print(f"🔗 Preloading MLflow model: {uri}")
mlflow.pyfunc.load_model(uri)
print("✅ Model and artifacts loaded successfully")
EOF

# --- Start FastAPI ---
echo "🚀 Starting API on port 7860"
exec uvicorn main:app --host 0.0.0.0 --port 7860
' > /app/start.sh \
    && chmod +x /app/start.sh

# ------------------------------------------------
# Default command
# ------------------------------------------------
CMD ["/app/start.sh"]
