import os
import yaml
import mlflow
import pandas as pd
from catboost import CatBoostRegressor
from mlflow.models import infer_signature
from time import perf_counter
from datetime import datetime
import json

from src.features.build_features import build_features
from src.rent_price_pipeline import RentPricePipeline
from src.utils import Timer


overall_start = perf_counter()

# ---------- Load parameters ----------
with Timer("Load parameters"):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

train_params = params["train"]
model_meta = params["model"]
TARGET = model_meta["target"]
NUMERIC = model_meta["numerical_features"]
CATEGORICAL = model_meta["categorical_features"]
FEATURES = NUMERIC + CATEGORICAL

# ---------- Prepare training data ----------
print("📦 Loading training data...")
with Timer("Load training data"):
    train_df = pd.read_parquet("data/processed/train.parquet")

print("Feature engineering...")
with Timer("Feature engineering"):
    train_df = build_features(train_df, geo_dir="data/geo")
    X_train, y_train = train_df[FEATURES], train_df[TARGET]

# ---------- Train CatBoost ----------
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

# ---------- Save base model ----------
with Timer("Save base model"):
    os.makedirs("models", exist_ok=True)
    cbm_path = "models/catboost_model_v1.cbm"
    model.save_model(cbm_path)

with Timer("Infer signature for CatBoost"):
    signature_catboost = infer_signature(X_train, model.predict(X_train[:5]))

# ---------- Prepare wrapper + raw input example ----------
print("🧠 Preparing wrapper and raw input example...")
with Timer("Prepare wrapper and raw input example"):
    wrapped = RentPricePipeline(cb_model_path=cbm_path, geo_dir="data/geo")

    # manually load the CatBoost model for local inference
    wrapped.model = CatBoostRegressor()
    wrapped.model.load_model(cbm_path)

    raw_df = pd.read_parquet("data/processed/train.parquet")
    input_example = raw_df.sample(1, random_state=42).drop(columns=[TARGET])

with Timer("Infer wrapper signature"):
    pred_example = wrapped.predict(None, input_example)
    signature_pipeline = infer_signature(input_example, pred_example)

# ---------- Log to MLflow ----------
print("📝 Logging models to MLflow...")
with Timer("Log to MLflow"):
    with mlflow.start_run(run_name=f"{model_meta['type']}_v1") as run:
        # 1. Log base CatBoost model
        mlflow.catboost.log_model(
            cb_model=model,
            name="catboost_model",
            input_example=X_train.sample(1, random_state=42),
            signature=signature_catboost,
        )
        base_uri = f"runs:/{run.info.run_id}/catboost_model"

        # 2. Log wrapper pipeline model
        mlflow.pyfunc.log_model(
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

        # 3. Add linkage tags
        mlflow.set_tags({
            "type": "rent_price_pipeline",
            "base_model_uri": base_uri,
            "features_version": "v1",
            "input_schema": "raw_property_data",
        })

        # ---------- Save run metadata for DVC and evaluation ----------
        print("🧩 Saving MLflow run metadata for DVC linkage...")
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        run_info = {
            "run_id": run.info.run_id,
            "pipeline_model_uri": f"runs:/{run.info.run_id}/pipeline_model",
            "catboost_model_uri": f"runs:/{run.info.run_id}/catboost_model",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mlflow_experiment": mlflow.get_experiment(run.info.experiment_id).name,
            "mlflow_ui_link": f"http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
        }

        with open(os.path.join(reports_dir, "last_run_info.json"), "w") as f:
            json.dump(run_info, f, indent=2)

        with open(os.path.join(reports_dir, "last_run_id.txt"), "w") as f:
            f.write(run.info.run_id)

        print("   📄 Saved run metadata to reports/last_run_info.json")
        print("   Run ID:", run.info.run_id)
        print("   MLflow UI:", run_info["mlflow_ui_link"])


print("✅ Training and logging complete!")
print("   Base model URI:", base_uri)
print("   Wrapper pipeline logged at: pipeline_model/")

total_elapsed = perf_counter() - overall_start
print(f"🏁 Total script time: {total_elapsed:.2f}s")
