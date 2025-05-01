import geopandas as gpd
import os
from overturemaps import core
from lonboard import Map, ScatterplotLayer
import osmnx as ox

# Step 1: Define rough bounding box for San Francisco
bbox = (-123.25, 37.55, -122.20, 38.00)

# Step 2: Download GERS Places inside bounding box
print("Downloading San Francisco GERS places (bounding box)...")
gdf = core.geodataframe("place", bbox=bbox)
gdf.set_crs(epsg=4326, inplace=True)
print(f"Downloaded {len(gdf)} places in bounding box.")

# Step 3: Get the real San Francisco city polygon from OSM
print("Fetching real San Francisco boundary polygon...")
sf_boundary_gdf = ox.geocode_to_gdf("San Francisco, California, USA")

# Step 4: Clip (filter) GERS places to be only inside the city polygon
print("Filtering GERS places inside official SF boundary...")
gdf = gdf.to_crs(sf_boundary_gdf.crs)  # Make sure both in same CRS
gdf_clipped = gpd.sjoin(gdf, sf_boundary_gdf, how="inner", predicate="within")
print(f"Places after clipping: {len(gdf_clipped)}")

# Step 5: Save clipped data
output_folder = "overture_data"
os.makedirs(output_folder, exist_ok=True)

gdf_clipped.to_file(os.path.join(output_folder, "sf_gers_places.geojson"), driver="GeoJSON")
gdf_clipped[['id', 'names', 'geometry']].to_csv(os.path.join(output_folder, "sf_gers_places.csv"), index=False)

print("Saved clipped data to overture_data/")

# Step 6: (Optional) Visualize
layer = ScatterplotLayer.from_geopandas(
    gdf_clipped,
    get_fill_color=[0, 128, 128],
    radius_min_pixels=1.5,
)

view_state = {
    "longitude": -122.4194,  # Centered roughly on SF downtown
    "latitude": 37.7749,
    "zoom": 12,
    "pitch": 45,
}
m = Map(layer, view_state=view_state)
m
