import pandas as pd
from overturemaps import core
import geopandas as gpd
import os
import ast
import re
import numpy as np

BBOX = (-123.25, 37.55, -122.20, 38.00)
OUTPUT_CSV_FILE = "overture_extracted_address_components_custom.csv"

def robust_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError):
            return None
    return val

def extract_primary_name(names_data):
    names_data_eval = robust_literal_eval(names_data)
    if isinstance(names_data_eval, dict):
        primary = names_data_eval.get('primary')
        if isinstance(primary, str):
            return primary.strip()
        common_names = names_data_eval.get('common')
        if isinstance(common_names, list) and len(common_names) > 0:
            if isinstance(common_names[0], str):
                return common_names[0].strip()
        return ""
    elif isinstance(names_data, str):
        match = re.search(r"(?:'primary':\s*'([^']*)'|\"primary\":\s*\"([^\"]*)\")", names_data)
        if match:
            return (match.group(1) if match.group(1) else match.group(2)).strip()
        if not (names_data.startswith('{') and names_data.endswith('}')):
            return names_data.strip()
    return ""

def main():
    print(f"--- Overture Address Component Extractor (Custom Columns with Cleaned Freeform) ---")
    print(f"Using Bounding Box: {BBOX}")

    print("Downloading Overture 'place' data...")
    try:
        gdf = core.geodataframe("place", bbox=BBOX)
        print(f"Downloaded {len(gdf)} places from Overture.")
    except Exception as e:
        print(f"Error downloading Overture data: {e}")
        return

    if gdf.empty:
        print("No data downloaded for the specified bounding box.")
        return

    if 'id' not in gdf.columns or 'addresses' not in gdf.columns or 'names' not in gdf.columns:
        print("Error: Downloaded data is missing essential 'id', 'names', or 'addresses' columns.")
        print(f"Available columns: {list(gdf.columns)}")
        return

    extracted_addresses = []
    processed_count = 0
    total_places = len(gdf)
    
    print(f"Processing {total_places} places...")

    for index, place_row in gdf.iterrows():
        place_id = place_row['id']
        
        addresses_data = place_row['addresses']
        parsed_addresses = robust_literal_eval(addresses_data)

        if isinstance(parsed_addresses, (list, np.ndarray)):
            for i, addr_item in enumerate(parsed_addresses):
                if isinstance(addr_item, dict):
                    raw_freeform = addr_item.get('freeform')
                    cleaned_address_freeform = None
                    if isinstance(raw_freeform, str):
                        temp_cleaned = re.sub(r'\d+', '', raw_freeform)
                        cleaned_address_freeform = ' '.join(temp_cleaned.split()).strip()
                        if not cleaned_address_freeform:
                            cleaned_address_freeform = None
                    
                    extracted_info = {
                        'address_freeform': cleaned_address_freeform,
                        'street': addr_item.get('street'),
                        'locality': addr_item.get('locality'),
                        'district': addr_item.get('district'),
                        'country': addr_item.get('country')
                    }
                    extracted_addresses.append(extracted_info)
                else:
                    print(f"Warning: Place ID {place_id}, address item at index {i} is not a dictionary: {type(addr_item)}")
        elif parsed_addresses is not None:
            print(f"Warning: Place ID {place_id} has 'addresses' field that is not a list or NumPy array after parsing: {type(parsed_addresses)}")

        processed_count += 1
        if processed_count % 500 == 0 or processed_count == total_places : # Adjusted progress reporting
            print(f"  Processed {processed_count}/{total_places} places...")

    if not extracted_addresses:
        print("No address components could be extracted from the downloaded places.")
        return

    output_df = pd.DataFrame(extracted_addresses)
    
    column_order = [
        'address_freeform', 'street', 'locality', 'district', 'country'
    ]
    
    final_columns = [col for col in column_order if col in output_df.columns]
    output_df = output_df[final_columns]
    
    try:
        output_df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
        print(f"\nSuccessfully extracted and saved {len(output_df)} address entries to '{OUTPUT_CSV_FILE}'")
        print("\nPreview of the first 5 extracted address entries:")
        print(output_df.head())
        print("\nSummary of non-null values for extracted components:")
        output_df.info()

    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main()