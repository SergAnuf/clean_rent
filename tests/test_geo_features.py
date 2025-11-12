from src.features.geo_features import LondonPropertyGeoFeatures


geo = LondonPropertyGeoFeatures("data/geo")

print(geo.extract_geo_features( 51.37364,0.09737))