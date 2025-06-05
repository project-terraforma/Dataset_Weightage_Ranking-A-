import os
import sys
import subprocess
import time 
import re 
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS 
import pandas as pd # For CSV handling in /save_verification
import csv # For more controlled CSV writing
import traceback

app = Flask(__name__)
print("Flask app object created.") 
CORS(app) 

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DOWNLOADER_SCRIPT_PATH = os.path.join(BASE_DIR, "download.py") 
MATCHER_SCRIPT_PATH = os.path.join(BASE_DIR, "matcher_scripty.py")
DEFAULT_OUTPUT_DIR_NAME = "downloaded_geo_data_from_web"
DEFAULT_OUTPUT_DIR_PATH = os.path.join(BASE_DIR, DEFAULT_OUTPUT_DIR_NAME)
USER_VERIFICATIONS_CSV = os.path.join(DEFAULT_OUTPUT_DIR_PATH, "user_verifications.csv") # Path for user verifications

# --- Helper Function ---
# generate_bbox_suffix_for_run_dir can remain the same if used by download.py for naming

# --- Routes ---
@app.route('/')
def index():
    try:
        return render_template('map_interface_backend_comms.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        return "Error: Could not load the map interface.", 500

@app.route('/trigger_download', methods=['POST']) 
def trigger_download_and_match(): 
    print("Received request on /trigger_download") 
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided in request."}), 400

        min_lon = data.get('min_lon')
        min_lat = data.get('min_lat')
        max_lon = data.get('max_lon')
        max_lat = data.get('max_lat')

        if None in [min_lon, min_lat, max_lon, max_lat]:
            return jsonify({"status": "error", "message": "Missing one or more bounding box coordinates."}), 400
        try:
            bbox_coords_for_suffix = {
                'min_lon': float(min_lon), 'min_lat': float(min_lat),
                'max_lon': float(max_lon), 'max_lat': float(max_lat)
            }
        except ValueError:
            return jsonify({"status": "error", "message": "Coordinates must be valid numbers."}), 400
            
        print(f"Received BBox: {bbox_coords_for_suffix}")
        os.makedirs(DEFAULT_OUTPUT_DIR_PATH, exist_ok=True)
        
        python_executable = sys.executable
        if not os.path.exists(DATA_DOWNLOADER_SCRIPT_PATH):
            msg = f"Server config error: Downloader script '{os.path.basename(DATA_DOWNLOADER_SCRIPT_PATH)}' not found."
            print(f"ERROR: {msg}")
            return jsonify({"status": "error", "message": msg}), 500

        download_cmd = [
            python_executable, DATA_DOWNLOADER_SCRIPT_PATH,
            str(bbox_coords_for_suffix['min_lon']), str(bbox_coords_for_suffix['min_lat']),
            str(bbox_coords_for_suffix['max_lon']), str(bbox_coords_for_suffix['max_lat']),
            "--output_dir", DEFAULT_OUTPUT_DIR_PATH
        ]
        
        print(f"Executing synchronous download: {' '.join(download_cmd)}")
        specific_run_output_path = None
        try:
            download_process = subprocess.run(download_cmd, cwd=BASE_DIR, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            
            if download_process.returncode != 0:
                print(f"ERROR: Download script failed (Code: {download_process.returncode}).")
                print(f"Download STDOUT:\n{download_process.stdout}")
                print(f"Download STDERR:\n{download_process.stderr}")
                return jsonify({"status": "error", "message": f"Download script failed. STDERR: {download_process.stderr[:200]}..."}), 500
            
            print("Download script completed successfully.")
            subfolders = [f.path for f in os.scandir(DEFAULT_OUTPUT_DIR_PATH) if f.is_dir() and f.name.startswith("run_")]
            if not subfolders:
                print("ERROR: No run output directory found after download.")
                return jsonify({"status": "error", "message": "Could not determine output directory."}), 500
            specific_run_output_path = max(subfolders, key=os.path.getmtime)
            print(f"Determined run output path: {specific_run_output_path}")

        except Exception as e:
            print(f"Error during download subprocess: {e}")
            return jsonify({"status": "error", "message": f"Download process error: {str(e)}"}), 500

        if specific_run_output_path is None:
             return jsonify({"status": "error", "message": "Failed to get data path for matching."}), 500

        if not os.path.exists(MATCHER_SCRIPT_PATH):
            msg = f"Server config error: Matcher script '{os.path.basename(MATCHER_SCRIPT_PATH)}' not found."
            print(f"ERROR: {msg}")
            return jsonify({"status": "error", "message": msg}), 500

        match_cmd = [python_executable, MATCHER_SCRIPT_PATH, specific_run_output_path]
        print(f"Executing async matching: {' '.join(match_cmd)}")
        try:
            subprocess.Popen(match_cmd, cwd=BASE_DIR) # Async
            run_directory_basename = os.path.basename(specific_run_output_path)
            print(f"Matching process initiated for data in: {run_directory_basename}")
            return jsonify({
                "status": "success", 
                "message": f"Download complete. Matching initiated for '{run_directory_basename}'.",
                "run_directory": run_directory_basename # Send this back to frontend
            }), 202
        except Exception as e:
            print(f"Error starting matching subprocess: {e}")
            return jsonify({"status": "error", "message": f"Failed to start matching: {str(e)}"}), 500

    except Exception as e:
        print(f"Unexpected error in /trigger_download: {e}"); traceback.print_exc()
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500

@app.route('/get_match_results', methods=['GET'])
def get_match_results():
    """Serves the matches_with_confidence.csv file from a specific run directory."""
    run_dir_basename = request.args.get('run_dir')
    if not run_dir_basename:
        return jsonify({"status": "error", "message": "Missing 'run_dir' parameter."}), 400
    
    # Basic validation for run_dir_basename to prevent directory traversal issues
    if not re.match(r'^run_[\w.-]+$', run_dir_basename): # Allow alphanumeric, underscore, dot, hyphen
        return jsonify({"status": "error", "message": "Invalid 'run_dir' format."}), 400

    file_path = os.path.join(DEFAULT_OUTPUT_DIR_PATH, run_dir_basename, "matches_with_confidence.csv")
    
    if not os.path.exists(file_path):
        # Check if the directory itself exists, maybe the CSV isn't generated yet
        if not os.path.exists(os.path.dirname(file_path)):
            return jsonify({"status": "error", "message": f"Run directory '{run_dir_basename}' not found."}), 404
        return jsonify({"status": "pending", "message": "Match results CSV not found yet. Matching may still be in progress or failed."}), 404 # Or 202 if you want to imply pending

    try:
        # Send the file directly. Frontend will use PapaParse or similar.
        return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=False, mimetype='text/csv')
    except Exception as e:
        print(f"Error sending match results file '{file_path}': {e}")
        return jsonify({"status": "error", "message": "Could not retrieve match results."}), 500

@app.route('/save_verification', methods=['POST'])
def save_verification():
    """Saves user verification to a CSV file."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided."}), 400

        osm_id = data.get('osm_id')
        overture_id = data.get('overture_id')
        osm_name = data.get('osm_name_original') # Use original names for clarity in verification log
        overture_name = data.get('overture_name_original')
        verified_status = data.get('verified_status') # e.g., "match", "non-match"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if not all([osm_id, overture_id, osm_name is not None, overture_name is not None, verified_status]): # Check osm_name and overture_name for None explicitly
            return jsonify({"status": "error", "message": "Missing required verification data."}), 400

        # Ensure the main output directory exists
        os.makedirs(DEFAULT_OUTPUT_DIR_PATH, exist_ok=True)
        
        fieldnames = ['timestamp', 'osm_id', 'overture_id', 'osm_name', 'overture_name', 'verified_status']
        
        # Check if file exists to write header
        file_exists = os.path.isfile(USER_VERIFICATIONS_CSV)
        
        with open(USER_VERIFICATIONS_CSV, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader() # Write header only if file is new
            writer.writerow({
                'timestamp': timestamp,
                'osm_id': osm_id,
                'overture_id': overture_id,
                'osm_name': osm_name,
                'overture_name': overture_name,
                'verified_status': verified_status
            })
        
        return jsonify({"status": "success", "message": "Verification saved."}), 200

    except Exception as e:
        print(f"Error saving verification: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": "Failed to save verification."}), 500


if __name__ == '__main__':
    os.makedirs(DEFAULT_OUTPUT_DIR_PATH, exist_ok=True)
    templates_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)

    print(f"Flask app running. Access at http://127.0.0.1:5000/")
    print(f"HTML templates expected in: {templates_dir}")
    print(f"Data Downloader script: {DATA_DOWNLOADER_SCRIPT_PATH}")
    print(f"Matcher script: {MATCHER_SCRIPT_PATH}")
    print(f"User verifications will be saved to: {USER_VERIFICATIONS_CSV}")
    app.run(debug=True, port=5000, use_reloader=False)
