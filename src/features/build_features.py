import pandas as pd
from src.features.geo_features import LondonPropertyGeoFeatures
# Optional:
# from src.features.feature_engineering import add_non_geo_features
# from src.features.nlp_features import PropertyTextEncoder

def is_numeric_and_true(value):
    return isinstance(value, (int, float)) and bool(value)


def build_features(df: pd.DataFrame, geo_dir: str = "data/geo") -> pd.DataFrame:
    """
    Compute all features used by the rent price model.
    Combines geospatial, engineered, and NLP-derived features.
    """

    # 1. Geospatial engineered features
    geo = LondonPropertyGeoFeatures(geo_dir)
    df = geo.add_features_to_df(df)

    # 2. Other engineered features (optional)
    # df = add_non_geo_features(df)

    # 3. NLP / embeddings (optional)
    # encoder = PropertyTextEncoder()
    # df = encoder.add_nlp_embeddings(df, text_column="description")

    df["deposit"] = df["deposit"].apply(is_numeric_and_true)

    return df


