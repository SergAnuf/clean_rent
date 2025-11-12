# --- Imports ---
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import pandas as pd

# --- Constants ---
CITY_CENTER = (51.5072, -0.1276)  # London
EPSG = "EPSG:4326"


class LondonPropertyGeoFeatures:
    """Extract London property geo features for model inference."""

    def __init__(self, geo_dir):
        self.CITY_CENTER = CITY_CENTER
        self.EPSG = EPSG
        self.geo_dir = geo_dir
        self.load_datasets()
        self.prepare_station_tree()

    def load_datasets(self):
        """Load and prepare geographic datasets."""
        self.london_boundaries = gpd.read_file(f"{self.geo_dir}/london_boroughs.geojson").to_crs(self.EPSG)
        self.hex_gdf = gpd.read_parquet(f"{self.geo_dir}/noize.parquet").to_crs(self.EPSG)
        self.zone_fares = gpd.read_parquet(f"{self.geo_dir}/zone_fares.parquet").to_crs(self.EPSG)
        self.stations = gpd.read_parquet(f"{self.geo_dir}/rail_tfl.parquet").to_crs(self.EPSG)


    def prepare_station_tree(self):
        """Prepare BallTree for fast station distance queries."""
        # Convert stations to UTM for accurate distance calculations
        self.stations_utm = self.stations.to_crs(self.stations.estimate_utm_crs())
        station_coords = np.array([[p.x, p.y] for p in self.stations_utm.geometry])
        self.station_tree = BallTree(station_coords, leaf_size=15, metric='euclidean')
        self.station_names = self.stations_utm['CommonName'].values
        self.station_tfl = self.stations_utm['TFL'].values
        self.station_rail = self.stations_utm['RAIL'].values

    def _create_point_gdf(self, lat, lon):
        """Create a GeoDataFrame for the point (internal helper)."""
        point = Point(lon, lat)
        return gpd.GeoDataFrame(geometry=[point], crs=self.EPSG)

    def borough(self, lat, lon):
        """Return the London borough name containing the given coordinates."""
        prop_gdf = self._create_point_gdf(lat, lon)
        joined = gpd.sjoin(prop_gdf, self.london_boundaries, how="left", predicate="within")
        return joined.iloc[0].get("name", None)

    def compute_angle(self, lat, lon):
        """Compute angle (in radians) of a point relative to London center."""
        lat1, lon1 = np.radians(self.CITY_CENTER[0]), np.radians(self.CITY_CENTER[1])
        lat2, lon2 = np.radians(lat), np.radians(lon)

        dlon = lon2 - lon1
        x = np.cos(lat2) * np.sin(dlon)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        return np.arctan2(x, y)

    def distance_to_center(self, lat, lon):
        """Return distance from city center (in miles)."""
        return geodesic((lat, lon), self.CITY_CENTER).miles

    def noize_class(self, lat, lon):
        """Return noise class for given coordinates."""
        prop_gdf = self._create_point_gdf(lat, lon)
        joined = gpd.sjoin(prop_gdf, self.hex_gdf, how="left", predicate="within")
        return joined.iloc[0].get("NoiseClass", None)

    def zone_fare(self, lat, lon):
        """Return transport fare zone for given coordinates."""
        prop_gdf = self._create_point_gdf(lat, lon)
        joined = gpd.sjoin(prop_gdf, self.zone_fares, how="left", predicate="within")
        zone_name = joined.iloc[0].get("Name", None)
        # Extract just the zone number if format is "Zone X"
        if zone_name and "Zone" in zone_name:
            return zone_name.split(" ")[-1]
        return zone_name

    def find_nearest_stations(self, lat, lon, k=3, max_distance_meters=50000):
        """
        Find k nearest stations with distances and TFL/RAIL flags.
        Returns distances in miles.
        """
        prop_gdf = self._create_point_gdf(lat, lon)
        prop_utm = prop_gdf.to_crs(self.stations_utm.crs)

        # Query the BallTree
        prop_coords = np.array([[p.x, p.y] for p in prop_utm.geometry])
        distances_m, indices = self.station_tree.query(prop_coords, k=k)

        results = []
        for dist_m, idx in zip(distances_m[0], indices[0]):
            if dist_m <= max_distance_meters:
                station_data = {
                    'distance_miles': dist_m / 1609.34,
                    'name': self.station_names[idx],
                    'TFL': bool(self.station_tfl[idx]),
                    'RAIL': bool(self.station_rail[idx])
                }
                results.append(station_data)

        return results


    def extract_geo_features(self, lat, lon):
        """
        Extract all GEO features for model inference in the required format.
        """
        # Geographic features
        borough_name = self.borough(lat, lon)
        angle = self.compute_angle(lat, lon)
        center_distance = self.distance_to_center(lat, lon)
        noise_class = self.noize_class(lat, lon)
        zone = self.zone_fare(lat, lon)

        # Station features
        nearest_stations = self.find_nearest_stations(lat, lon, k=3)

        # Prepare station features with proper naming
        station_features = {}
        for i, station in enumerate(nearest_stations[:3], 1):
            station_features[f'distance_to_station{i}'] = round(station['distance_miles'], 6)
            station_features[f'TFL{i}'] = station['TFL']
            station_features[f'RAIL{i}'] = station['RAIL']

        # Fill missing stations with default values
        for i in range(len(nearest_stations) + 1, 4):
            station_features[f'distance_to_station{i}'] = None
            station_features[f'TFL{i}'] = False
            station_features[f'RAIL{i}'] = False

        geo_features = {
                "distance_to_center": round(center_distance, 6),
                "angle_from_center": round(angle, 6),
                "zone": zone,
                "borough": borough_name,
                "NoiseClass": noise_class,
                **station_features
            }

        return geo_features


    def add_features_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized feature extraction for a full DataFrame."""
        features = df.apply(
            lambda row: pd.Series(self.extract_geo_features(row["latitude"], row["longitude"])),
            axis=1
        )
        return pd.concat([df, features], axis=1)




