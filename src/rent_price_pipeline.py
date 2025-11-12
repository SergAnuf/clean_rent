import mlflow.pyfunc
import pandas as pd
from catboost import CatBoostRegressor
from src.features.build_features import build_features

class RentPricePipeline(mlflow.pyfunc.PythonModel):
    """
    MLflow-wrapped pipeline that:
    - loads CatBoost model and geo datasets
    - uses build_features() to compute all features
    - predicts rent prices
    """

    def __init__(self, cb_model_path=None, geo_dir=None):
        self.cb_model_path = cb_model_path
        self.geo_dir = geo_dir


    def load_context(self, context):
        """Load CatBoost model and geo datasets from MLflow artifacts."""
        model_path = context.artifacts.get("catboost_model", self.cb_model_path)
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        # ✅ Prefer MLflow artifact path if available
        self.geo_dir = context.artifacts.get("geo_dir", self.geo_dir or "data/geo")


    def predict(self, context, model_input):
        """Compute features and predict rent price."""
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        enriched = build_features(model_input, geo_dir=self.geo_dir)

        numerical_features = [
            "latitude", "longitude",
            "distance_to_center", "angle_from_center",
            "distance_to_station1", "distance_to_station2", "distance_to_station3"
        ]
        categorical_features = [
            "bedrooms","bathrooms","deposit","zone","borough","propertyType",
            "furnishType","NoiseClass","letType","TFL1","TFL2","TFL3",
            "RAIL1","RAIL2","RAIL3"
        ]
        features = numerical_features + categorical_features
        return self.model.predict(enriched[features])
