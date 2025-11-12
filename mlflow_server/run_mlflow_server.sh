#!/bin/bash
# ====================================
# Run MLflow server (PostgreSQL backend + GCS artifacts)
# ====================================

# Exit on any error
set -e

# Load environment variables from project root .env
set -o allexport
source "$(dirname "$0")/../.env"
set +o allexport

# Validate required environment variables
REQUIRED_VARS=(
  "DB_DESTINATION_USER"
  "DB_DESTINATION_PASSWORD"
  "DB_DESTINATION_HOST"
  "DB_DESTINATION_PORT"
  "DB_DESTINATION_NAME"
  "MLFLOW_ARTIFACT_ROOT"
  "GOOGLE_APPLICATION_CREDENTIALS"
)

for VAR in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR}" ]; then
    echo "❌ Missing required environment variable: $VAR"
    exit 1
  fi
done

# Print summary (but avoid printing secrets)
echo "🚀 Starting MLflow server..."
echo "   Backend: postgresql://${DB_DESTINATION_USER}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}"
echo "   Artifact root: $MLFLOW_ARTIFACT_ROOT"
echo "   Host: 127.0.0.1"
echo "   Port: 5000"
echo

# Ensure GOOGLE_APPLICATION_CREDENTIALS is readable
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
  echo "❌ GCP credentials file not found at $GOOGLE_APPLICATION_CREDENTIALS"
  exit 1
fi

# Start MLflow Tracking Server
exec mlflow server \
  --backend-store-uri "postgresql+psycopg2://${DB_DESTINATION_USER}:${DB_DESTINATION_PASSWORD}@${DB_DESTINATION_HOST}:${DB_DESTINATION_PORT}/${DB_DESTINATION_NAME}?sslmode=require" \
  --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
  --host 127.0.0.1 \
  --port 5000
