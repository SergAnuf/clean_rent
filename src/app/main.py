from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

app = FastAPI()

# Load model (bundled with geo logic & files)
model = mlflow.pyfunc.load_model("models:/RentPricePipeline/Production")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    y_pred = model.predict(df)
    return {"predicted_price": float(y_pred[0])}
