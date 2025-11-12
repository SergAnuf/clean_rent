# scripts/evaluate.py
import os
import json
import mlflow
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from src.utils import Timer

TEST_PATH = "data/processed/test.parquet"
RUN_INFO_PATH = "reports/last_run_info.json"
METRICS_PATH = "reports/eval_metrics.json"


def main():
    if not os.path.exists(RUN_INFO_PATH):
        raise FileNotFoundError(f"{RUN_INFO_PATH} not found. Run training first.")

    print("📄 Loading last MLflow run info...")
    with open(RUN_INFO_PATH) as f:
        run_info = json.load(f)

    run_id = run_info["run_id"]
    model_uri = run_info["pipeline_model_uri"]  # evaluate the wrapper model
    print(f"🔄 Loading model from {model_uri}")

    # ---------- Load MLflow model ----------
    with Timer("Load MLflow model"):
        model = mlflow.pyfunc.load_model(model_uri)

    # ---------- Load test data ----------
    print("📦 Loading test data...")
    df_test = pd.read_parquet(TEST_PATH)
    X_test = df_test.drop(columns=["price"])
    y_test = df_test["price"]

    # ---------- Run predictions ----------
    print("⚙️ Running inference on test set...")
    with Timer("Model inference"):
        preds = model.predict(X_test)

    # ---------- Compute metrics ----------
    print("📊 Computing metrics...")
    metrics = {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "mape": round(mean_absolute_percentage_error(y_test, preds), 2),
    }

    # ---------- Log metrics to MLflow ----------
    print("📝 Logging metrics to MLflow...")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    # ---------- Save metrics locally ----------
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation complete!")
    print(json.dumps(metrics, indent=2))
    print(f"🔗 MLflow UI: {run_info['mlflow_ui_link']}")


if __name__ == "__main__":
    main()
