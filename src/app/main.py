from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from handler import FastApiHandler

app = FastAPI(title="TrueNest Rent Prediction API")
handler = None


# ---------- Request schema with example ----------

class PredictRequest(BaseModel):
    model_params: dict = Field(
        ...,
        json_schema_extra={
            "example": {
                "bathrooms": 1,
                "bedrooms": 2,
                "propertyType": "Flat",
                "deposit": False,
                "letType": "Long term",
                "furnishType": "Furnished",
                "latitude": 51.49199,
                "longitude": -0.17134
            }
        },
    )



# ---------- Startup: load model once ----------
@app.on_event("startup")
def load_model_once():
    global handler
    handler = FastApiHandler()
    print("✅ MLflow model loaded at startup")


# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "🏡 Rent Prediction API is running", "run_id": handler.run_id}


@app.post("/predict")
def predict(req: PredictRequest):
    result = handler.handle(req.dict())
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
