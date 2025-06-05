import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import time
import argparse
from overturemaps import core # For Overture
import ast # For parsing stringified Python literals from Overture
import re  # For regex in Overture address cleaning
import numpy as np # For Overture address processing
import requests # For Overpass API fallback
from shapely.geometry import Point, LineString, Polygon # For Overpass fallback

# --- Utility Functions ---
def generate_bbox_filename_suffix(bbox_coords):
    """Generates a string suffix for filenames based on bbox coordinates."""
    return f"bbox_{bbox_coords[1]:.2f}_{bbox_coords[0]:.2f}_{bbox_coords[3]:.2f}_{bbox_coords[2]:.2f}"

def robust_literal_eval(val):
    """Safely evaluate a string containing a Python literal (list, dict)."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError):
            array_pattern_match = re.match(r"array\((.*),\s*dtype=object\)", val)
            if array_pattern_match:
                list_str = array_pattern_match.group(1)
                try:
                    return ast.literal_eval(list_str)
                except (ValueError, SyntaxError, TypeError):
                    return None
            return None
    return val

# --- OSM Data: Overpass API Fallback Function ---
def fetch_osm_data_via_overpass_api(osm_bbox_nswe, tags_dict, timeout=90):
    """
    Fetches OSM data using a direct Overpass API query as a fallback.
    Args:
        osm_bbox_nswe (tuple): Bounding box as (north, south, east, west).
        tags_dict (dict): Dictionary of tags to retrieve (keys are used).
        timeout (int): Timeout for the API request in seconds.
    Returns:
        geopandas.GeoDataFrame or None: Processed data or None if failed.
    """
    north, south, east, west = osm_bbox_nswe
    # Overpass API expects bbox as: (south, west, north, east)
    overpass_bbox_str = f"{south},{west},{north},{east}"
    print(f"Overpass API Fallback: Using bbox {overpass_bbox_str}")

    # Construct query parts for each tag key
    query_parts = []
    for key in tags_dict.keys():
        # Fetch nodes, ways, and relations that have the specified tag key
        query_parts.append(f'node["{key}"]({overpass_bbox_str});')
        query_parts.append(f'way["{key}"]({overpass_bbox_str});')
        query_parts.append(f'relation["{key}"]({overpass_bbox_str});')
    
    full_query_parts_str = "\n  ".join(query_parts)

    overpass_query = f"""
    [out:json][timeout:{timeout}];
    (
      {full_query_parts_str}
    );
    out geom;
    """
    # print(f"Overpass Query:\n{overpass_query}") # For debugging

    overpass_url = "https://overpass-api.de/api/interpreter"
    try:
        print("Overpass API Fallback: Sending request...")
        response = requests.post(overpass_url, data=overpass_query, timeout=timeout)
        response.raise_for_status()  # Raises an exception for HTTP errors
        data = response.json()
        print(f"Overpass API Fallback: Received {len(data.get('elements', []))} elements.")
    except requests.exceptions.RequestException as e:
        print(f"Overpass API Fallback: Request failed: {e}")
        return None
    except ValueError as e: # Includes JSONDecodeError
        print(f"Overpass API Fallback: Failed to decode JSON response: {e}")
        return None

    features = []
    if not data or 'elements' not in data:
        print("Overpass API Fallback: No 'elements' in response.")
        return None

    for element in data['elements']:
        props = element.get('tags', {})
        props['osmid'] = element.get('id')
        props['element_type'] = element.get('type')
        geom = None

        if element['type'] == 'node' and 'lat' in element and 'lon' in element:
            geom = Point(element['lon'], element['lat'])
        elif element['type'] == 'way' and 'geometry' in element:
            coords = [(pt['lon'], pt['lat']) for pt in element['geometry']]
            if len(coords) >= 2:
                # Basic check for polygon: closed way with area-like tags or explicit area=yes
                # This is a simplification; osmnx has more robust logic.
                is_polygon = False
                if coords[0] == coords[-1] and len(coords) >= 4: # At least 3 distinct points for a polygon
                    if 'area' in props and props['area'] == 'yes':
                        is_polygon = True
                    elif any(tag in props for tag in ['building', 'landuse', 'natural', 'leisure']):
                        # Common tags that imply an area if the way is closed
                        is_polygon = True
                
                if is_polygon:
                    try:
                        geom = Polygon(coords)
                    except Exception: # Handle invalid polygon geometry (e.g. self-intersecting)
                        geom = LineString(coords) # Fallback to LineString
                else:
                    geom = LineString(coords)
        # Relations are more complex to convert to simple geometries directly from `out geom;`
        # For simplicity, we'll skip converting relation geometries unless `out geom;` provides a direct one.
        # `osmnx` handles multipolygons from relations, which is non-trivial.
        elif element['type'] == 'relation' and 'bounds' in element and 'center' in element:
            # As a very rough representation, create a point for the relation's center
            # Or use bounds to create a bounding box polygon (less accurate for feature shape)
            # geom = Point(element['center']['lon'], element['center']['lat'])
            # For now, we'll primarily rely on nodes and ways for geometry in this fallback
            pass


        if geom:
            feature_dict = {'geometry': geom}
            # Flatten tags into properties, prefixing with 'tag_' to avoid clashes if needed,
            # but for now, direct assignment. OSMnx does this flattening.
            for k, v in props.items():
                feature_dict[k.replace(':', '_')] = v # Sanitize column names
            features.append(feature_dict)

    if not features:
        print("Overpass API Fallback: No geometric features could be processed.")
        return gpd.GeoDataFrame([], crs="EPSG:4326") # Return empty GeoDataFrame

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    
    # Ensure common columns exist, similar to what downstream processing might expect
    # This part might need adjustment based on how osmnx structures its GDF.
    # For example, osmnx might ensure 'name' column exists, etc.
    # The current analysis code iterates row attributes, so tags being columns is important.

    return gdf


# --- OSM Data Processing Section (Main) ---
# Helper functions (get_osm_place_name, etc.) remain the same
def get_osm_place_name(row):
    if 'name' in row and pd.notna(row['name']) and isinstance(row['name'], str) and row['name'].strip():
        return row['name']
    return None

def is_osm_category_defined(row, category_tags=['amenity', 'shop', 'tourism', 'office', 'leisure', 'craft', 'public_transport', 'historic', 'man_made']):
    for tag in category_tags:
        if tag in row and pd.notna(row[tag]):
            if isinstance(row[tag], str) and row[tag].strip() != "": return True
            elif not isinstance(row[tag], str): return True
    return False

def is_osm_website_populated(row):
    tag = 'website'
    return tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != ""

def is_osm_socials_populated(row, social_tags=['contact:facebook', 'contact:instagram', 'contact:twitter', 'contact:youtube', 'facebook', 'instagram', 'twitter']):
    for tag in social_tags:
        if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": return True
    return False

def is_osm_email_populated(row, email_tags=['email', 'contact:email']):
    for tag in email_tags:
        if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": return True
    return False

def is_osm_phone_populated(row, phone_tags=['phone', 'contact:phone']):
    for tag in phone_tags:
        if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": return True
    return False

def is_osm_brand_populated(row):
    tag = 'brand'
    return tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != ""

def is_osm_address_details_populated(row):
    has_street, has_city_or_postcode = False, False
    street_tags = ['addr_street', 'addr_housenumber'] # Note: using underscore due to potential sanitization
    for tag in street_tags:
        if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": has_street = True; break
    city_tags, postcode_tags = ['addr_city'], ['addr_postcode']
    for tag in city_tags:
        if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": has_city_or_postcode = True; break
    if not has_city_or_postcode:
        for tag in postcode_tags:
            if tag in row and pd.notna(row[tag]) and isinstance(row[tag], str) and row[tag].strip() != "": has_city_or_postcode = True; break
    return has_street and has_city_or_postcode

OSM_CONCEPTUAL_ATTRIBUTES = {
    'has_category': is_osm_category_defined, 'has_website': is_osm_website_populated,
    'has_socials': is_osm_socials_populated, 'has_email': is_osm_email_populated,
    'has_phone': is_osm_phone_populated, 'has_brand': is_osm_brand_populated,
    'has_address_details': is_osm_address_details_populated
}


def process_osm_data_for_bbox(bbox_coords, output_dir):
    """Downloads and processes OSM data for a given bounding box, with Overpass API fallback."""
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    osm_bbox_nswe_tuple = (max_lat, min_lat, max_lon, min_lon) 
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Processing OSM Data for BBox: {bbox_coords} ---")
    print(f"Output directory: {output_dir}")
    
    try:
        print(f"Using OSMnx version: {ox.__version__}")
    except Exception:
        print("Could not determine OSMnx version.")

    tags_to_retrieve = {
        'amenity': True, 'shop': True, 'tourism': True, 'office': True,
        'leisure': True, 'craft': True, 'public_transport': True,
        'historic': True, 'man_made': True, 'name': True, 'website': True,
        'contact:facebook': True, 'contact:instagram': True, 'contact:twitter': True, 'contact:youtube': True,
        'facebook': True, 'instagram': True, 'twitter': True, 'email': True, 'contact:email': True,
        'phone': True, 'contact:phone': True, 'brand': True,
        'addr:street': True, 'addr:housenumber': True, 'addr:city': True, 'addr:postcode': True, 'addr:country': True
    }

    gdf = None
    source_api = "OSMnx"

    try:
        print(f"Attempting to download OSM features using OSMnx (bbox tuple: {osm_bbox_nswe_tuple})...")
        try:
            gdf = ox.features_from_bbox(osm_bbox_nswe_tuple, tags=tags_to_retrieve)
        except AttributeError:
            print("ox.features_from_bbox not found, trying ox.features.features_from_bbox...")
            gdf = ox.features.features_from_bbox(osm_bbox_nswe_tuple, tags=tags_to_retrieve)
        
        if gdf is None or gdf.empty: # Explicitly check if gdf is None or empty after call
             raise ValueError("OSMnx returned None or empty GeoDataFrame.")
        print(f"Successfully downloaded {len(gdf)} features using OSMnx.")

    except Exception as osmnx_error:
        print(f"OSMnx download failed: {osmnx_error}")
        print("Attempting to download OSM data using direct Overpass API query as a fallback...")
        source_api = "OverpassAPI"
        gdf = fetch_osm_data_via_overpass_api(osm_bbox_nswe_tuple, tags_to_retrieve)
        
        if gdf is None or gdf.empty:
            print("Overpass API fallback also failed or returned no usable data.")
            return None
        else:
            print(f"Successfully fetched {len(gdf)} features using Overpass API fallback.")

    print(f"Downloaded {len(gdf)} OSM features using {source_api}.")
    if gdf.empty: # Should be caught earlier, but double check
        print("No OSM features found for the given bounding box.")
        return None

    # Clean column names (important for both OSMnx and Overpass GDFs)
    # OSMnx might have already done some cleaning.
    # The Overpass fallback GDF creation already replaces ':' with '_'.
    if isinstance(gdf.columns, pd.MultiIndex): # Should primarily apply to OSMnx GDFs
        gdf.columns = ['_'.join(str(cp) for cp in col).strip('_') if isinstance(col, tuple) else str(col) for col in gdf.columns.values]
    
    # General cleaning for any remaining problematic characters from tags
    new_cols = {}
    for col in gdf.columns:
        new_col_name = str(col).replace(':', '_').replace('-', '_').replace('.', '_')
        # Ensure unique column names if cleaning creates duplicates (less likely with simple replacement)
        # For simplicity, direct replacement here.
        new_cols[col] = new_col_name
    gdf.rename(columns=new_cols, inplace=True)


    # 1. Per-Place Attribute Presence
    per_place_data = []
    # gdf_copy for analysis - ensure it has a unique ID and name if possible
    gdf_copy = gdf.copy()

    # Ensure 'osmid' and 'element_type' exist if possible, for consistency
    if 'osmid' not in gdf_copy.columns and 'id' in gdf_copy.columns: # Common after Overpass
        gdf_copy.rename(columns={'id': 'osmid'}, inplace=True)
    
    # Unify ID for per-place analysis
    # This logic might need refinement depending on actual columns from Overpass GDF
    if 'osmid_unified' not in gdf_copy.columns:
        if 'osmid' in gdf_copy.columns and 'element_type' in gdf_copy.columns:
            gdf_copy['osmid_unified'] = gdf_copy['element_type'].astype(str) + "_" + gdf_copy['osmid'].astype(str)
        elif 'osmid' in gdf_copy.columns:
            gdf_copy['osmid_unified'] = gdf_copy['osmid'].astype(str)
        elif 'id' in gdf_copy.columns: # Fallback if only 'id' exists
             gdf_copy['osmid_unified'] = gdf_copy['id'].astype(str)
        else: # Last resort
            gdf_copy = gdf_copy.reset_index()
            gdf_copy['osmid_unified'] = gdf_copy.index.astype(str)


    for index_val, row in gdf_copy.iterrows(): 
        place_info = {'osmid': row.get('osmid_unified', index_val), 'name': get_osm_place_name(row)}
        for attr_key, checker_func in OSM_CONCEPTUAL_ATTRIBUTES.items():
            place_info[attr_key] = checker_func(row) # Assumes tags are columns
        per_place_data.append(place_info)
    
    per_place_df = pd.DataFrame(per_place_data)
    per_place_csv_path = os.path.join(output_dir, "osm_features_per_place_presence.csv")
    per_place_df.to_csv(per_place_csv_path, index=False)
    print(f"Saved OSM per-place attribute presence to: {per_place_csv_path}")

    # 2. All Columns CSV and GeoJSON
    all_columns_csv_path = os.path.join(output_dir, "osm_features_all_columns.csv")
    
    # Prepare GDF for CSV: drop geometry, ensure simple index if osmid is not a column
    df_for_csv = gdf.drop(columns=['geometry'], errors='ignore').copy()
    if isinstance(df_for_csv.index, pd.MultiIndex): df_for_csv = df_for_csv.reset_index()
    # If 'osmid' (or 'id' from Overpass) was the main ID and is now an index after some ops
    if 'osmid' not in df_for_csv.columns and df_for_csv.index.name == 'osmid':
        df_for_csv = df_for_csv.reset_index()
    elif 'id' not in df_for_csv.columns and df_for_csv.index.name == 'id': # from Overpass
        df_for_csv = df_for_csv.reset_index()

    df_for_csv.to_csv(all_columns_csv_path, index=False)
    print(f"Saved all OSM features (no geometry) to: {all_columns_csv_path}")

    geojson_path = os.path.join(output_dir, "osm_features.geojson")
    try:
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"Saved OSM features (with geometry) to: {geojson_path}")
    except Exception as e:
        print(f"Error saving GeoJSON: {e}. Ensure all geometries are valid.")
        print("The Overpass API fallback might produce mixed or invalid geometries more easily than OSMnx.")


    # 3. Aggregate Analysis (from the CSV just saved)
    try:
        df_from_csv = pd.read_csv(all_columns_csv_path, low_memory=False, keep_default_na=False, na_values=['', 'None', 'nan', 'NaN', '<NA>'])
        if not df_from_csv.empty:
            attribute_counts_agg = {key: 0 for key in OSM_CONCEPTUAL_ATTRIBUTES.keys()}
            total_analyzed = len(df_from_csv)
            for _, row in df_from_csv.iterrows():
                for attr_key, checker_func in OSM_CONCEPTUAL_ATTRIBUTES.items():
                    if checker_func(row): attribute_counts_agg[attr_key] += 1
            
            agg_results = []
            for key, count in attribute_counts_agg.items():
                perc = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
                agg_results.append({
                    'osm_attribute_concept': key,
                    'percentage_populated': round(perc, 2),
                    'count_populated': count,
                    'total_analyzed': total_analyzed
                })
            agg_df = pd.DataFrame(agg_results)
            agg_csv_path = os.path.join(output_dir, "osm_attributes_aggregate_population.csv")
            agg_df.to_csv(agg_csv_path, index=False)
            print(f"Saved OSM aggregate attribute population to: {agg_csv_path}")
    except Exception as e:
        print(f"Warning: Could not perform OSM aggregate analysis from CSV: {e}")

    return all_columns_csv_path


# --- Overture Places (GERS) Data Processing Section ---
# (This section remains unchanged)
def is_overture_categories_populated(cat_data):
    if not isinstance(cat_data, dict): cat_data = robust_literal_eval(cat_data)
    if not isinstance(cat_data, dict): return False
    return isinstance(cat_data.get('primary'), str) and cat_data['primary'].strip() != ""

def is_overture_list_of_strings_populated(list_data):
    if not isinstance(list_data, list): list_data = robust_literal_eval(list_data)
    if not isinstance(list_data, list) or not list_data: return False
    return any(isinstance(item, str) and item.strip() != "" for item in list_data)

def is_overture_brand_populated(brand_data):
    if not isinstance(brand_data, dict): brand_data = robust_literal_eval(brand_data)
    if not isinstance(brand_data, dict): return False
    if isinstance(brand_data.get('wikidata'), str) and brand_data['wikidata'].strip() != "": return True
    names = brand_data.get('names')
    if isinstance(names, dict):
        return isinstance(names.get('primary'), str) and names['primary'].strip() != ""
    return False

def is_overture_addresses_populated(addr_list_data):
    if not isinstance(addr_list_data, list): addr_list_data = robust_literal_eval(addr_list_data)
    if not isinstance(addr_list_data, list) or not addr_list_data: return False
    for addr in addr_list_data:
        if isinstance(addr, dict):
            if (isinstance(addr.get('freeform'), str) and addr['freeform'].strip() != "") or \
               (isinstance(addr.get('street'), str) and addr['street'].strip() != "") or \
               (isinstance(addr.get('locality'), str) and addr['locality'].strip() != ""):
                return True
    return False

OVERTURE_CONCEPTUAL_ATTRIBUTES = {
    'categories': is_overture_categories_populated, 'websites': is_overture_list_of_strings_populated,
    'socials': is_overture_list_of_strings_populated, 'emails': is_overture_list_of_strings_populated,
    'phones': is_overture_list_of_strings_populated, 'brand': is_overture_brand_populated,
    'addresses': is_overture_addresses_populated
}

def process_overture_places_for_bbox(bbox_coords, output_dir):
    """Downloads and processes Overture Places data for a given bounding box."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Processing Overture Places Data for BBox: {bbox_coords} ---")
    print(f"Output directory: {output_dir}")

    bbox_overture_format = (bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])

    print(f"Downloading Overture 'place' features for bbox: {bbox_overture_format}...")
    try:
        gdf = core.geodataframe("place", bbox=bbox_overture_format)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True) 
    except Exception as e:
        print(f"ERROR: Could not download Overture Places features: {e}")
        return None, None

    print(f"Downloaded {len(gdf)} Overture Places features.")
    if gdf.empty:
        print("No Overture Places features found for the given bounding box.")
        return None, None

    all_columns_csv_path = os.path.join(output_dir, "overture_places_all_columns.csv")
    gdf.to_csv(all_columns_csv_path, index=False)
    print(f"Saved all Overture Places features to: {all_columns_csv_path}")

    geojson_path = os.path.join(output_dir, "overture_places.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved Overture Places features (with geometry) to: {geojson_path}")

    try:
        df_from_csv = pd.read_csv(all_columns_csv_path, keep_default_na=False, na_values=['', 'None', 'nan', 'NaN', 'NULL', '{}', '[]'])
        
        if not df_from_csv.empty:
            attribute_counts_agg = {key: 0 for key in OVERTURE_CONCEPTUAL_ATTRIBUTES.keys()}
            total_analyzed = len(df_from_csv)

            for _, row in df_from_csv.iterrows():
                for attr_key, checker_func in OVERTURE_CONCEPTUAL_ATTRIBUTES.items():
                    if attr_key in row:
                        raw_val = row[attr_key]
                        if checker_func(raw_val): 
                            attribute_counts_agg[attr_key] += 1
            
            agg_results = []
            for key, count in attribute_counts_agg.items():
                perc = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
                agg_results.append({
                    'overture_attribute_concept': key,
                    'percentage_populated': round(perc, 2),
                    'count_populated': count,
                    'total_analyzed': total_analyzed
                })
            agg_df = pd.DataFrame(agg_results)
            agg_csv_path = os.path.join(output_dir, "overture_places_attributes_aggregate_population.csv")
            agg_df.to_csv(agg_csv_path, index=False)
            print(f"Saved Overture Places aggregate attribute population to: {agg_csv_path}")
    except Exception as e:
        print(f"Warning: Could not perform Overture Places aggregate analysis from CSV: {e}")
        
    return all_columns_csv_path, gdf

# --- Overture Address Extraction Section ---
# (This section remains unchanged)
def extract_overture_address_components(overture_gdf, output_dir):
    """Extracts address components from Overture GeoDataFrame."""
    if overture_gdf is None or overture_gdf.empty:
        print("No Overture GeoDataFrame provided for address extraction.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Extracting Address Components from Overture Data ---")
    print(f"Output directory: {output_dir}")

    if 'id' not in overture_gdf.columns or 'addresses' not in overture_gdf.columns:
        print("ERROR: Overture GDF missing 'id' or 'addresses' columns for address extraction.")
        return None

    extracted_addresses = []
    processed_count = 0
    total_places = len(overture_gdf)

    for _, place_row in overture_gdf.iterrows():
        place_id = place_row['id']
        addresses_data_raw = place_row['addresses']
        
        parsed_addresses = addresses_data_raw if isinstance(addresses_data_raw, list) else robust_literal_eval(addresses_data_raw)

        if isinstance(parsed_addresses, (list, np.ndarray)):
            for i, addr_item in enumerate(parsed_addresses):
                if isinstance(addr_item, dict):
                    raw_freeform = addr_item.get('freeform')
                    cleaned_address_freeform = None
                    if isinstance(raw_freeform, str):
                        temp_cleaned = re.sub(r'\d+', '', raw_freeform) 
                        temp_cleaned = re.sub(r'[^\w\s]', '', temp_cleaned) 
                        cleaned_address_freeform = ' '.join(temp_cleaned.split()).strip().lower() 
                        if not cleaned_address_freeform:
                            cleaned_address_freeform = None
                    
                    extracted_info = {
                        'overture_place_id': place_id,
                        'address_index': i,
                        'address_freeform_cleaned': cleaned_address_freeform,
                        'raw_freeform': raw_freeform,
                        'street': addr_item.get('street'),
                        'locality': addr_item.get('locality'),
                        'district': addr_item.get('district'),
                        'region': addr_item.get('region'), 
                        'postcode': addr_item.get('postcode'),
                        'country': addr_item.get('country')
                    }
                    extracted_addresses.append(extracted_info)
        
        processed_count += 1
        if processed_count % 1000 == 0 or processed_count == total_places:
            print(f"  Processed {processed_count}/{total_places} places for address extraction...")

    if not extracted_addresses:
        print("No address components could be extracted.")
        return None

    output_df = pd.DataFrame(extracted_addresses)
    output_csv_path = os.path.join(output_dir, "overture_extracted_address_components.csv")
    
    column_order = [
        'overture_place_id', 'address_index', 'address_freeform_cleaned', 'raw_freeform', 
        'street', 'locality', 'district', 'region', 'postcode', 'country'
    ]
    final_columns = [col for col in column_order if col in output_df.columns]
    output_df = output_df[final_columns]
    
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully extracted and saved {len(output_df)} address entries to: {output_csv_path}")
    return output_csv_path

# --- Main Orchestration Function ---
# (This section remains unchanged)
def main_downloader(bbox_coords, base_output_dir="downloaded_geodata"):
    """Main function to orchestrate the download and processing of geographic data."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bbox_suffix = generate_bbox_filename_suffix(bbox_coords)
    run_output_dir_name = f"run_{timestamp}_{bbox_suffix}"
    run_output_path = os.path.join(base_output_dir, run_output_dir_name)
    
    print(f"Starting data download and processing run.")
    print(f"Bounding Box: {bbox_coords}")
    print(f"All outputs will be saved under: {run_output_path}")
    os.makedirs(run_output_path, exist_ok=True)

    osm_output_dir = os.path.join(run_output_path, "osm_data")
    overture_places_output_dir = os.path.join(run_output_path, "overture_places_data")
    overture_addresses_output_dir = os.path.join(run_output_path, "overture_address_components")

    osm_main_csv = process_osm_data_for_bbox(bbox_coords, osm_output_dir)
    if osm_main_csv:
        print(f"OSM data processing complete. Main CSV: {osm_main_csv}")
    else:
        print("OSM data processing failed or yielded no data.")

    overture_places_main_csv, overture_gdf = process_overture_places_for_bbox(bbox_coords, overture_places_output_dir)
    if overture_places_main_csv:
        print(f"Overture Places data processing complete. Main CSV: {overture_places_main_csv}")
    else:
        print("Overture Places data processing failed or yielded no data.")

    if overture_gdf is not None and not overture_gdf.empty:
        overture_addresses_csv = extract_overture_address_components(overture_gdf, overture_addresses_output_dir)
        if overture_addresses_csv:
            print(f"Overture address component extraction complete. CSV: {overture_addresses_csv}")
        else:
            print("Overture address component extraction failed or yielded no data.")
    else:
        print("Skipping Overture address extraction as Overture Places data was not available.")

    print(f"\n--- Unified Data Download and Processing Finished ---")
    print(f"All files for this run are in: {run_output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and process OpenStreetMap and Overture Maps data for a given bounding box.")
    parser.add_argument("min_lon", type=float, help="Minimum longitude (West bound) of the bounding box.")
    parser.add_argument("min_lat", type=float, help="Minimum latitude (South bound) of the bounding box.")
    parser.add_argument("max_lon", type=float, help="Maximum longitude (East bound) of the bounding box.")
    parser.add_argument("max_lat", type=float, help="Maximum latitude (North bound) of the bounding box.")
    parser.add_argument("--output_dir", type=str, default="downloaded_geodata", help="Base directory to save the output run folders.")
    
    args = parser.parse_args()
    
    bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    
    main_downloader(bbox_coords=bbox, base_output_dir=args.output_dir)
