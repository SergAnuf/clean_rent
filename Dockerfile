# ================================================================
#  TrueNest - MLflow + FastAPI Inference Image
#  Author: Dr. Sergii Anufriev
# ================================================================

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
# Copy FastAPI app + model metadata
# ------------------------------------------------
COPY ./src/app /app
COPY ./reports/last_run_info.json /app/reports/last_run_info.json

# ------------------------------------------------
# Copy MLflow-generated requirements (from model artifact)
# ------------------------------------------------
# You should have downloaded it locally with:
# gsutil cp gs://rent_price_bucket/mlflow/runs/<run_id>/pipeline_model/requirements.txt ./requirements.txt
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
# Startup script: configure secrets, preload model, run app
# ------------------------------------------------
RUN echo '#!/bin/bash\n\
set -e\n\
# --- Setup Google credentials ---\n\
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then \n\
  echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/creds.json; \n\
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/creds.json; \n\
  echo "✅ GCP credentials configured"; \n\
else \n\
  echo "⚠️ No GOOGLE_APPLICATION_CREDENTIALS_JSON provided (GCS access may fail)"; \n\
fi;\n\
# --- Preload MLflow model from GCS ---\n\
python - <<EOF\n\
import mlflow, json;\n\
info=json.load(open(\"/app/reports/last_run_info.json\"));\n\
print(f\"🔗 Preloading MLflow model from {info['pipeline_model_uri']} ...\");\n\
mlflow.pyfunc.load_model(info[\"pipeline_model_uri\"]);\n\
print(\"✅ Model and artifacts loaded successfully\");\n\
EOF\n\
# --- Start FastAPI ---\n\
exec uvicorn main:app --host 0.0.0.0 --port 7860' > /app/start.sh \
    && chmod +x /app/start.sh

# ------------------------------------------------
# Default command
# ------------------------------------------------
CMD ["/app/start.sh"]
