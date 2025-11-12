import os
import mlflow
from dotenv import load_dotenv

# === Load secrets ===
load_dotenv()

# === Configure GCS credentials ===
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# === Configure MLflow server connection ===
TRACKING_SERVER_HOST = "127.0.0.1"   # or external IP if remote access enabled
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "GCS_INTEGRATION_TEST"

# Tell MLflow to talk to the server — not directly to Postgres!
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

print("Tracking URI:", mlflow.get_tracking_uri())

# === Create or set experiment ===
mlflow.set_experiment(EXPERIMENT_NAME)

# === Example run ===
with mlflow.start_run(run_name="test_integration_gcs"):
    mlflow.log_param("alpha", 0.05)
    mlflow.log_metric("rmse", 0.82)

    # Log a text artifact (will go to GCS!)
    mlflow.log_text("Privet sergey !!!", "hello_world.txt")

print("✅ Run logged successfully — check MLflow UI for GCS path")

