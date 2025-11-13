import os, sys
import json
import mlflow
import pandas as pd
import requests
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import Timer

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TEST_PATH = "data/processed/test.parquet"
RUN_INFO_PATH = "reports/last_run_info.json"
METRICS_PATH = "reports/eval_metrics.json"

# Tracking server info (should match training script)
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000


def main():
    # ---------------------------------------------------------------
    # 1. Load metadata
    # ---------------------------------------------------------------
    if not os.path.exists(RUN_INFO_PATH):
        raise FileNotFoundError(f"❌ {RUN_INFO_PATH} not found. Run training first.")

    print("📄 Loading last MLflow run info...")
    with open(RUN_INFO_PATH) as f:
        run_info = json.load(f)

    run_id = run_info["run_id"]
    model_uri = run_info["pipeline_model_uri"]

    print(f"🔍 Run ID: {run_id}")
    print(f"🔄 Model URI: {model_uri}")
    print()

    # ---------------------------------------------------------------
    # Ensure Google Cloud credentials are set
    # ---------------------------------------------------------------
    GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not GOOGLE_CREDS or not os.path.isfile(GOOGLE_CREDS):
        raise FileNotFoundError(
            f"❌ GOOGLE_APPLICATION_CREDENTIALS not set or invalid: {GOOGLE_CREDS}"
        )
    print(f"🔐 Using Google credentials: {GOOGLE_CREDS}")

    # ---------------------------------------------------------------
    # 2. Connect to MLflow tracking server
    # ---------------------------------------------------------------
    try:
        r = requests.get(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}", timeout=3)
        if r.status_code != 200:
            raise requests.exceptions.RequestException
    except requests.exceptions.RequestException:
        raise ConnectionError(
            f"❌ MLflow tracking server not reachable at "
            f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}. "
            f"Start the server before evaluation."
        )

    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
    mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

    print(f"🔗 Connected to MLflow: {mlflow.get_tracking_uri()}")
    print(f"   Using run ID: {run_id}")
    print()

    # ---------------------------------------------------------------
    # 3. Load model from GCS
    # ---------------------------------------------------------------
    with Timer("Load MLflow model"):
        model = mlflow.pyfunc.load_model(model_uri)

    # ---------------------------------------------------------------
    # 4. Load test data
    # ---------------------------------------------------------------
    print("📦 Loading test data...")
    df_test = pd.read_parquet(TEST_PATH)
    X_test = df_test.drop(columns=["price"])
    y_test = df_test["price"]

    # ---------------------------------------------------------------
    # 5. Run inference
    # ---------------------------------------------------------------
    print("⚙️ Running inference on test set...")
    with Timer("Model inference"):
        preds = model.predict(X_test)

    # ---------------------------------------------------------------
    # 6. Compute metrics
    # ---------------------------------------------------------------
    print("📊 Computing metrics...")
    metrics = {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "mape": round(mean_absolute_percentage_error(y_test, preds), 4),
    }

    # ---------------------------------------------------------------
    # 7. Log metrics to MLflow (same run)
    # ---------------------------------------------------------------
    print("📝 Logging metrics to MLflow...")
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.end_run()

    # ---------------------------------------------------------------
    # 8. Save metrics locally for DVC
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation complete!")
    print(json.dumps(metrics, indent=2))
    print(f"🔗 MLflow UI: {run_info['mlflow_ui_link']}")


if __name__ == "__main__":
    main()

