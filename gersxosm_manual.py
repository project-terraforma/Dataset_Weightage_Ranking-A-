import geopandas as gpd
from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pandas as pd
import joblib
from rapidfuzz.fuzz import token_set_ratio

# Load trained model
model = joblib.load("name_match_xgm.pkl")

# Load data
osm_gdf = gpd.read_file("osm_data/sf_osm_features.geojson")
gers_gdf = gpd.read_file("overture_data/sf_gers_places.geojson")

# Ensure CRS
osm_gdf.set_crs(epsg=4326, inplace=True)
gers_gdf.set_crs(epsg=4326, inplace=True)
osm_gdf = osm_gdf.to_crs(epsg=3857)
gers_gdf = gers_gdf.to_crs(epsg=3857)

# Match parameters
max_distance_meters = 200
match_count = 0
abstain_count = 0
total_count = len(osm_gdf)

# Spatial index
gers_sindex = gers_gdf.sindex

# Thresholds
LOW = 0.25
HIGH = 0.75

# Matching loop
for idx, osm_row in osm_gdf.iterrows():
    osm_point = osm_row.geometry
    osm_name = str(osm_row.get("name", "")).lower()

    # Nearby GERS candidates
    possible_matches_idx = list(gers_sindex.intersection(osm_point.buffer(max_distance_meters).bounds))
    candidates = gers_gdf.iloc[possible_matches_idx]
    candidates = candidates[candidates.distance(osm_point) <= max_distance_meters]

    matched = False

    for _, gers_row in candidates.iterrows():
        gers_name = str(gers_row.get("names", "")).lower()
        dist = osm_point.distance(gers_row.geometry)
        # Get point for OSM geometry (handle polygons)
        osm_centroid = osm_point.centroid

        # Feature extraction
        features = pd.DataFrame([{
            "distance_m": dist,
            "token_overlap": len(set(osm_name.split()) & set(gers_name.split())),
            "name_length_diff": abs(len(osm_name) - len(gers_name)),
            "first_word_match": int(osm_name.split()[0] == gers_name.split()[0]),
            "fuzzy_score": token_set_ratio(osm_name, gers_name),
            "latitude_1_x": osm_centroid.y,
            "longitude_1_x": osm_centroid.x,
            "latitude_2_y": gers_row.geometry.y,
            "longitude_2_y": gers_row.geometry.x
        }])
        prob = model.predict_proba(features)[0][1]
        # Predict with abstain logic
        osm_abstained = True  # assume all were uncertain

        for _, gers_row in candidates.iterrows():
            ...
            if prob >= HIGH:
                matched = True
                osm_abstained = False
                break
            elif prob <= LOW:
                osm_abstained = False  # confident non-match

        if matched:
            match_count += 1
        elif osm_abstained:
            abstain_count += 1


# Final metrics
match_percentage = (match_count / total_count) * 100
abstain_percentage = (abstain_count / total_count) * 100
mismatch_percentage = 100 - match_percentage - abstain_percentage

print(f"\n📊 Matching Summary:")
print(f"Total OSM Places Checked: {total_count}")
print(f"Matches Found: {match_count}")
print(f"Abstained: {abstain_count}")
print(f"Match %: {match_percentage:.2f}%")
print(f"Abstain %: {abstain_percentage:.2f}%")
print(f"Mismatch (Predicted Non-Match) %: {mismatch_percentage:.2f}%")

# GERS stats
unique_gers_ids = gers_gdf['id'].nunique()
total_gers_places = len(gers_gdf)

print(f"\n--- Overture Dataset Stats ---")
print(f"Total GERS Places: {total_gers_places}")
print(f"Total Unique GERS IDs: {unique_gers_ids}")
