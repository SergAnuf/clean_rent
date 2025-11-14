import os, sys
import json
import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv

# Load .env BEFORE anything else
load_dotenv()

# Ensure the project root (which contains 'src') is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class FastApiHandler:
    """Handler for rent price prediction using MLflow pipeline model."""

    def __init__(
        self,
        run_info_path: str = "reports/last_run_info.json",
    ):
        self.run_info_path = run_info_path
        self.model = None
        self.run_id = None
        self.model_uri = None

        self._configure_gcp_credentials()
        self.load_model()  # Load once at startup

    # -----------------------------------------------------------
    # Configure Google Cloud authentication
    # -----------------------------------------------------------
    def _configure_gcp_credentials(self):
        """Loads GCP credentials from HF ENV or system ENV."""

        # Hugging Face Spaces: JSON secret
        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

        if creds_json:
            print("🔐 Configuring GCP credentials from ENV JSON...")
            with open("/tmp/gcp_creds.json", "w") as f:
                f.write(creds_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp_creds.json"

        # Local dev or Docker with .env
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print("🔐 Using GOOGLE_APPLICATION_CREDENTIALS from environment")

        else:
            print("⚠️ WARNING: No GCP credentials provided! GCS model loading may fail.")

    # -----------------------------------------------------------
    # Load the MLflow model
    # -----------------------------------------------------------
    def load_model(self):
        if not os.path.exists(self.run_info_path):
            raise FileNotFoundError(
                f"❌ {self.run_info_path} not found — train the model first."
            )

        with open(self.run_info_path) as f:
            info = json.load(f)

        self.run_id = info.get("run_id")
        self.model_uri = info.get("pipeline_model_uri")

        print(f"🔗 Loading MLflow model: {self.model_uri}")

        # MLflow resolves GCS path automatically from runs:/ URI
        self.model = mlflow.pyfunc.load_model(self.model_uri)

        print(f"✅ Model loaded successfully (run_id={self.run_id})")

    # -----------------------------------------------------------
    # Predict
    # -----------------------------------------------------------
    def predict(self, model_params: dict) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        df = pd.DataFrame([model_params])
        preds = self.model.predict(df)
        return float(preds[0])

    # -----------------------------------------------------------
    # FastAPI-compatible handler
    # -----------------------------------------------------------
    def handle(self, params: dict) -> dict:
        if "model_params" not in params:
            return {"error": "Missing 'model_params' in request"}

        try:
            prediction = self.predict(params["model_params"])
        except Exception as e:
            return {"error": str(e)}

        return {
            "prediction": prediction,
            "inputs": params["model_params"],
            "run_id": self.run_id,
        }
