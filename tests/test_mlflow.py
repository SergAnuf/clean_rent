import mlflow
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

MODEL_URI =  "models:/m-201aa76a32bc4a4883fc9702dad34373"

print("🔗 Loading MLflow model from:")
print(MODEL_URI)

model = mlflow.pyfunc.load_model(MODEL_URI)

print("✅ Model loaded successfully")

df = pd.DataFrame([{
    "latitude": 51.50,
    "longitude": -0.13,
    "bedrooms": 2,
    "bathrooms": 1,
    "propertyType": "Flat",          # required
    "deposit": True,                 # required
    "letType": "Long Let",           # required
    "furnishType": "Furnished",      # required
}])


print(model.predict(df))

