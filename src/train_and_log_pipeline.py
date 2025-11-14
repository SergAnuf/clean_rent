import os, sys
import yaml
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from time import perf_counter
from datetime import datetime
import json
import requests
from dotenv import load_dotenv

# Add project root to sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import build_features
from src.rent_price_pipeline import RentPricePipeline
from src.utils import Timer


# ============================================================
# 0. Setup: environment, credentials, MLflow connection
# ============================================================
overall_start = perf_counter()
load_dotenv()

GOOGLE_CREDS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDS or not os.path.isfile(GOOGLE_CREDS):
    raise FileNotFoundError(f"❌ GOOGLE_APPLICATION_CREDENTIALS invalid: {GOOGLE_CREDS}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS

TRACKING_SERVER_HOST = "127.0.0.1"   # or external IP if remote
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = "Rent_Price_Pipeline"

# ---- Verify MLflow server ----
try:
    r = requests.get(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}", timeout=3)
    if r.status_code != 200:
        raise requests.exceptions.RequestException
except requests.exceptions.RequestException:
    raise ConnectionError(
        f"❌ MLflow tracking server not reachable at http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}. "
        f"Please start it before running this script."
    )

# ---- Configure MLflow ----
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"🔗 Connected to MLflow tracking server: {mlflow.get_tracking_uri()}")
print(f"Experiment: {EXPERIMENT_NAME}")
print()

# ============================================================
# 1. Load parameters and prepare data
# ============================================================
with Timer("Load parameters"):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

train_params = params["train"]
model_meta = params["model"]
TARGET = model_meta["target"]
NUMERIC = model_meta["numerical_features"]
CATEGORICAL = model_meta["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL
NOT_USED_COLUMNS = model_meta["not_used_features"]

print("📦 Loading training data...")
with Timer("Load training data"):
    train_df = pd.read_parquet("data/processed/train.parquet")
    train_df = train_df.drop(columns=NOT_USED_COLUMNS)

print("🧩 Feature engineering...")
with Timer("Feature engineering"):
    train_df = build_features(train_df, geo_dir="data/geo")
    X_train, y_train = train_df[FEATURES], train_df[TARGET]

# ============================================================
# 2. Train CatBoost model
# ============================================================
print("🚀 Training CatBoost model...")
with Timer("Training CatBoost"):
    model = CatBoostRegressor(
        iterations=train_params["iterations"],
        depth=train_params["depth"],
        learning_rate=train_params["learning_rate"],
        l2_leaf_reg=train_params["l2_leaf_reg"],
        bagging_temperature=train_params["bagging_temperature"],
        cat_features=CATEGORICAL,
        verbose=False,
    )
    model.fit(X_train, y_train)

# ============================================================
# 3. Save model and prepare signatures
# ============================================================
with Timer("Save base model"):
    os.makedirs("models", exist_ok=True)
    cbm_path = "models/catboost_model_v1.cbm"
    model.save_model(cbm_path)

with Timer("Infer signature for CatBoost"):
    signature_catboost = infer_signature(X_train, model.predict(X_train[:5]))

print("🧠 Preparing wrapper pipeline...")
with Timer("Prepare wrapper and raw input example"):
    wrapped = RentPricePipeline(cb_model_path=cbm_path, geo_dir="data/geo")
    wrapped.model = CatBoostRegressor()
    wrapped.model.load_model(cbm_path)

    raw_df = pd.read_parquet("data/processed/train.parquet")
    raw_df = raw_df.drop(columns=NOT_USED_COLUMNS)
    input_example = raw_df.sample(1, random_state=42).drop(columns=[TARGET])

with Timer("Infer wrapper signature"):
    pred_example = wrapped.predict(None, input_example)
    signature_pipeline = infer_signature(input_example, pred_example)

# ============================================================
# 4. Log everything to MLflow
# ============================================================
print("📝 Logging models to MLflow...")
with Timer("Log to MLflow"):
    with mlflow.start_run(run_name=f"{model_meta['type']}_v1") as run:
        # ---- Log CatBoost base model ----
        mlflow.catboost.log_model(
            cb_model=model,
            name="catboost_model",
            input_example=X_train.sample(1, random_state=42),
            signature=signature_catboost,
        )
        base_uri = f"runs:/{run.info.run_id}/catboost_model"

        # ---- Log wrapper pipeline ----
        logged = mlflow.pyfunc.log_model(
            name="pipeline_model",
            python_model=wrapped,
            code_paths=[
                "src/features/geo_features.py",
                "src/features/build_features.py",
                "src/rent_price_pipeline.py",
            ],
            artifacts={
                "catboost_model": cbm_path,
                "geo_dir": "data/geo",
            },
            signature=signature_pipeline,
            input_example=input_example,
        )

        # ---- Tags for traceability ----
        mlflow.set_tags({
            "type": "rent_price_pipeline",
            "base_model_uri": base_uri,
            "features_version": "v1",
            "input_schema": "raw_property_data",
        })

        # ---- Save run metadata for DVC ----
        print("🧩 Saving MLflow run metadata for DVC linkage...")
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        # ============================================================
        # Capture run ID (MLflow-native)
        # ============================================================
        run_id = run.info.run_id

        # MLflow-native model URI (recommended for all environments)
        pipeline_model_uri = logged.model_uri

        # ============================================================
        # Save metadata to JSON — clean and portable
        # ============================================================
        run_info = {
            "run_id": run_id,
            "pipeline_model_uri": pipeline_model_uri,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mlflow_experiment": mlflow.get_experiment(run.info.experiment_id).name,
            "mlflow_ui_link": (
                f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}/#/experiments/"
                f"{run.info.experiment_id}/runs/{run_id}"
            ),
        }

        with open(os.path.join(reports_dir, "last_run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        with open(os.path.join(reports_dir, "last_run_id.txt"), "w") as f:
            f.write(run_id)

        print("   📄 Saved run metadata to reports/last_run_info.json")
        print("   Run ID:", run_id)
        print("   Pipeline model URI:", pipeline_model_uri)
        print("   MLflow UI:", run_info["mlflow_ui_link"])

# ============================================================
# 5. Completion
# ============================================================
print("✅ Training and remote logging completed!")
print("   Base model URI:", base_uri)
print("   Wrapper pipeline logged as: pipeline_model/")
print(f"🏁 Total script time: {perf_counter() - overall_start:.2f}s")