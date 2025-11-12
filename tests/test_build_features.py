import pandas as pd
from src.features.build_features import build_features
import yaml

train_df = pd.read_parquet("data/processed/train.parquet")

df_sample = train_df.sample(10)
df_features = build_features(df_sample)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train_params = params["train"]
model_meta   = params["model"]
TARGET       = model_meta["target"]

NUMERIC      = model_meta["numerical_features"]
CATEGORICAL  = model_meta["categorical_features"]
FEATURES     = NUMERIC + CATEGORICAL


print(df_features[FEATURES].columns)