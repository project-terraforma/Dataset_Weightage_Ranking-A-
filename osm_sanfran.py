import os
import osmnx as ox
import geopandas as gpd

# Define the area of interest
place_name = "San Francisco, California, USA"

# Specify the tags for the features you want to retrieve
tags = {'amenity': True, 'shop': True, 'tourism': True}

# Retrieve the data using the updated function
print(f"Downloading OSM features for {place_name}...")
gdf = ox.features_from_place(place_name, tags=tags)

# Check what we got
print(f"Downloaded {len(gdf)} features.")
print(gdf.head())

# Save to local files
output_folder = "osm_data"
os.makedirs(output_folder, exist_ok=True)

geojson_path = os.path.join(output_folder, "sf_osm_features.geojson")
csv_path = os.path.join(output_folder, "sf_osm_features.csv")

print(f"Saving to {geojson_path} and {csv_path}...")
gdf.to_file(geojson_path, driver="GeoJSON")
gdf.drop(columns='geometry').to_csv(csv_path, index=False)

print("Done! Files saved locally.")
