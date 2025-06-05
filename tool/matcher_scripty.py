import pandas as pd
import geopandas as gpd
import numpy as np
import os
import re
import time
import argparse
import ast # For literal_eval
from rapidfuzz.fuzz import WRatio, token_set_ratio 
from sentence_transformers import SentenceTransformer, util
import torch
import joblib # For loading XGBoost model
import traceback # For more detailed error printing

# For Qwen LLM (adapt imports based on your final setup from queneval.py)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
MAX_DISTANCE_METERS = 200
MIN_FUZZY_THRESHOLD = 30
SENTENCE_MODEL_CONFIDENCE_THRESHOLD = 0.73330253 # Cosine similarity threshold
XGBOOST_PREDICTION_THRESHOLD = 0.5 # Assuming XGBoost outputs a probability; 0.5 for binary classification
XGBOOST_CONFIDENT_THRESHOLD = 0.75 # XGBoost probability to be considered "confident" for Green matches

# --- Model Paths (Updated based on user's file structure) ---
FINE_TUNED_SENTENCE_TRANSFORMER_PATH = './fine_tuned_name_matcher_online_contrastive' # For ST similarity score
XGBOOST_EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' # For XGBoost's embedding_cosine_similarity feature
XGBOOST_MODEL_PATH = "./name_match_xgboost_csv_with_embeddings.pkl"

# Qwen LLM Paths
QWEN_BASE_MODEL_FOR_TOKENIZER = 'Qwen/Qwen-7B-Chat' # Base model for tokenizer configuration
QWEN_MERGED_MODEL_PATH = './qwen_7b_chat_merged_final' # Path to your merged Qwen model

# --- Global Model Variables ---
device = "cuda" if torch.cuda.is_available() else "cpu"
sentence_model = None # This will be the fine-tuned one for direct ST similarity
xgboost_feature_embedding_model = None # This will be all-MiniLM-L6-v2 for XGBoost feature
xgboost_model = None
llm_model = None
llm_tokenizer = None

# XGBoost Feature Names (must match training order from trainname3.py)
XGBOOST_FEATURE_NAMES = [
    'token_overlap',
    'name_length_diff',
    'first_word_match',
    'fuzzy_score', 
    'embedding_cosine_similarity'
]


# Qwen-specific Prompts
QWEN_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
QWEN_USER_INSTRUCTION_PROMPT_TEMPLATE = """You are a highly accurate assistant specialized in matching place names. Carefully determine if the two names below refer to the exact same place. The names have been pre-processed to remove common locality information.

Be stricter when comparing names for restaurants, banks, cafes, schools, or other common businesses and institutions.

Answer with only "Yes" or "No". Do not provide any explanation or any other text.

OSM Name: "{osm_name}"
GERS Name: "{gers_name}"."""


# --- Helper Functions ---
def clean_name(name):
    if pd.isna(name) or name is None:
        return ""
    name_str = str(name).lower().strip()
    name_str = re.sub(r'[^\w\s]', '', name_str) 
    name_str = re.sub(r'\s+', ' ', name_str).strip() 
    return name_str

def extract_osm_name_raw(osm_row):
    return osm_row.get('name', '') 

def extract_overture_primary_name_raw(names_field):
    if pd.isna(names_field): return ""
    try:
        if isinstance(names_field, list):
            names_list = names_field
        elif isinstance(names_field, str) and names_field.startswith('['):
             names_list = ast.literal_eval(names_field)
        elif isinstance(names_field, str) and names_field.startswith('{'):
            names_dict_eval = ast.literal_eval(names_field)
            if isinstance(names_dict_eval, dict) and 'primary' in names_dict_eval:
                return str(names_dict_eval['primary'])
            if isinstance(names_dict_eval, dict) and 'common' in names_dict_eval and isinstance(names_dict_eval['common'], list) and names_dict_eval['common']:
                 first_common = names_dict_eval['common'][0]
                 if isinstance(first_common, dict) and 'value' in first_common:
                     return str(first_common['value'])
                 else:
                     return str(first_common)
            return str(names_field)
        else:
            return str(names_field)

        if isinstance(names_list, list) and names_list:
            for name_entry in names_list:
                if isinstance(name_entry, dict) and name_entry.get('is_primary') and 'value' in name_entry:
                    return str(name_entry['value'])
            for name_entry in names_list:
                if isinstance(name_entry, dict) and 'value' in name_entry:
                    return str(name_entry['value'])
        return str(names_field)
    except (ValueError, SyntaxError, TypeError):
        return str(names_field)


def load_all_models():
    global sentence_model, xgboost_feature_embedding_model, xgboost_model, llm_model, llm_tokenizer, device
    print(f"--- Loading All Models (Device: {device}) ---")
    # 1. Fine-tuned Sentence Transformer (for direct ST similarity score)
    try:
        if os.path.exists(FINE_TUNED_SENTENCE_TRANSFORMER_PATH):
            print(f"Loading FINE-TUNED Sentence Transformer from: {FINE_TUNED_SENTENCE_TRANSFORMER_PATH}")
            sentence_model = SentenceTransformer(FINE_TUNED_SENTENCE_TRANSFORMER_PATH, device=device)
            print("Fine-tuned Sentence Transformer loaded successfully.")
        else: print(f"Warning: Fine-tuned Sentence Transformer model not found at {FINE_TUNED_SENTENCE_TRANSFORMER_PATH}. Direct ST similarity will be affected.")
    except Exception as e: print(f"ERROR loading Fine-tuned Sentence Transformer: {e}.")

    # 2. Base Sentence Transformer (for XGBoost embedding_cosine_similarity feature)
    try:
        print(f"Loading BASE Sentence Transformer for XGBoost features from: {XGBOOST_EMBEDDING_MODEL_NAME}")
        xgboost_feature_embedding_model = SentenceTransformer(XGBOOST_EMBEDDING_MODEL_NAME, device=device)
        print("Base Sentence Transformer for XGBoost features loaded successfully.")
    except Exception as e: print(f"ERROR loading Base Sentence Transformer for XGBoost: {e}. XGBoost features will be affected.")


    # 3. XGBoost Model
    try:
        if os.path.exists(XGBOOST_MODEL_PATH):
            print(f"Loading XGBoost model from: {XGBOOST_MODEL_PATH}")
            xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
            print("XGBoost model loaded successfully.")
        else: print(f"Warning: XGBoost model not found at {XGBOOST_MODEL_PATH}. XGBoost matching will be skipped.")
    except Exception as e: print(f"ERROR loading XGBoost model: {e}. XGBoost matching will be skipped.")

    # 4. Qwen LLM and Tokenizer
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(QWEN_BASE_MODEL_FOR_TOKENIZER, trust_remote_code=True)
        QWEN_EOS_TOKEN = "<|im_end|>"
        if llm_tokenizer.eos_token_id is None:
            if hasattr(llm_tokenizer, 'im_end_id') and llm_tokenizer.im_end_id is not None:
                llm_tokenizer.eos_token_id, llm_tokenizer.eos_token = llm_tokenizer.im_end_id, QWEN_EOS_TOKEN
            elif QWEN_EOS_TOKEN in llm_tokenizer.vocab:
                llm_tokenizer.eos_token_id, llm_tokenizer.eos_token = llm_tokenizer.vocab[QWEN_EOS_TOKEN], QWEN_EOS_TOKEN
            else:
                llm_tokenizer.add_tokens([QWEN_EOS_TOKEN], special_tokens=True)
                llm_tokenizer.eos_token = QWEN_EOS_TOKEN
                if QWEN_EOS_TOKEN in llm_tokenizer.vocab:
                     llm_tokenizer.eos_token_id = llm_tokenizer.vocab[QWEN_EOS_TOKEN]
                else: 
                    raise ValueError("Failed to set EOS token for Qwen after attempting to add it.")
        
        if llm_tokenizer.pad_token_id is None:
            if llm_tokenizer.eos_token_id is not None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id if llm_tokenizer.pad_token_id is None else llm_tokenizer.pad_token_id
            else: 
                unk_id = llm_tokenizer.unk_token_id
                if unk_id is not None: llm_tokenizer.pad_token, llm_tokenizer.pad_token_id = llm_tokenizer.unk_token, unk_id
                else: 
                    llm_tokenizer.add_tokens(["<|pad|>"], special_tokens=True); llm_tokenizer.pad_token, llm_tokenizer.pad_token_id = "<|pad|>", llm_tokenizer.vocab["<|pad|>"] # type: ignore
                    print("Warning: Had to add a new PAD token as EOS and UNK were unavailable for padding.")

        llm_tokenizer.padding_side = 'left'
        print(f"Qwen Tokenizer configured. EOS ID: {llm_tokenizer.eos_token_id}, PAD ID: {llm_tokenizer.pad_token_id}")
        if llm_tokenizer.pad_token_id is None: raise ValueError("CRITICAL: Qwen Tokenizer pad_token_id is still None.")

        if os.path.exists(QWEN_MERGED_MODEL_PATH):
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
            llm_model = AutoModelForCausalLM.from_pretrained(QWEN_MERGED_MODEL_PATH, quantization_config=quant_config, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            if llm_model is not None: 
                 llm_model.eval()
                 print("Qwen LLM loaded successfully.")
            else: 
                 print(f"Warning: Qwen LLM model from {QWEN_MERGED_MODEL_PATH} resulted in None. LLM verification will be skipped.")
        else: print(f"Warning: Qwen LLM model not found at {QWEN_MERGED_MODEL_PATH}.")
    except Exception as e: print(f"ERROR loading Qwen LLM/Tokenizer: {e}."); traceback.print_exc()
    print("--- Model Loading Complete ---")


def get_xgboost_prediction(name1_globally_cleaned, name2_globally_cleaned, osm_feature=None, overture_feature=None):
    if xgboost_model is None: return False, 0.0
    # Use xgboost_feature_embedding_model for embeddings for XGBoost
    if xgboost_feature_embedding_model is None: 
        print("Warning: Base Sentence Transformer for XGBoost features not loaded. XGBoost prediction will be inaccurate or fail."); 
        return False, 0.0
    try:
        s1, s2 = (str(n) if pd.notna(n) else "" for n in [name1_globally_cleaned, name2_globally_cleaned])
        token_overlap = len(set(s1.lower().split()) & set(s2.lower().split()))
        name_length_diff = abs(len(s1) - len(s2))
        fw_s1, fw_s2 = (s.lower().split()[0] if s else "" for s in [s1, s2])
        first_word_match = 1 if fw_s1 == fw_s2 and fw_s1 != "" else 0
        fuzzy_score_xgb = token_set_ratio(s1, s2) # As per trainname3.py
        
        # Generate embeddings using the BASE model (all-MiniLM-L6-v2) for XGBoost feature
        if not s1 and not s2: cos_sim = 1.0
        elif not s1 or not s2: cos_sim = 0.0
        else:
            emb1 = xgboost_feature_embedding_model.encode(s1, convert_to_tensor=True, device=device)
            emb2 = xgboost_feature_embedding_model.encode(s2, convert_to_tensor=True, device=device)
            cos_sim = util.pytorch_cos_sim(emb1, emb2).item()
        if isinstance(cos_sim, torch.Tensor): cos_sim = cos_sim.cpu().item()
        
        ft_df = pd.DataFrame([[token_overlap, name_length_diff, first_word_match, fuzzy_score_xgb, cos_sim]], columns=XGBOOST_FEATURE_NAMES)
        prob = xgboost_model.predict_proba(ft_df)[:, 1][0]
        return prob >= XGBOOST_PREDICTION_THRESHOLD, float(prob)
    except Exception as e: print(f"Error in XGBoost prediction for ('{s1}', '{s2}'): {e}"); traceback.print_exc(); return False, 0.0


def get_sentence_similarity(name1_globally_cleaned, name2_globally_cleaned):
    # This uses the FINE-TUNED sentence_model for direct ST similarity score
    if sentence_model is None: return 0.0
    try:
        s1, s2 = (str(n) if pd.notna(n) else "" for n in [name1_globally_cleaned, name2_globally_cleaned])
        if not s1 and not s2: return 1.0
        if not s1 or not s2: return 0.0
        emb1 = sentence_model.encode(s1, convert_to_tensor=True, device=device)
        emb2 = sentence_model.encode(s2, convert_to_tensor=True, device=device)
        return util.pytorch_cos_sim(emb1, emb2).item()
    except Exception as e: print(f"Error in sentence similarity for ('{s1}', '{s2}'): {e}"); return 0.0

def get_address_components_from_cols(components_df, target_cols):
    unique_components = set()
    if components_df is not None:
        for col_name in target_cols:
            if col_name in components_df.columns:
                for component in components_df[col_name].dropna().astype(str):
                    cleaned_component = component.strip().lower()
                    if cleaned_component: 
                        unique_components.add(cleaned_component)
            else:
                print(f"Info: Column '{col_name}' not found in address components CSV. Skipping for this component type.")
    return sorted(list(unique_components), key=len, reverse=True) 


def remove_components_from_name(name_to_clean, components_list):
    cleaned_name = name_to_clean 
    for component_item in components_list:
        if not component_item: continue
        try:
            pattern_for_removal = re.compile(r'\b' + re.escape(component_item) + r'\b', re.IGNORECASE)
            cleaned_name = pattern_for_removal.sub("", cleaned_name)
        except re.error:
            continue 
    return " ".join(cleaned_name.split()).strip() 

def preprocess_name_globally(raw_name, general_components_list):
    if pd.isna(raw_name): return ""
    cleaned_name_basic = clean_name(str(raw_name)) 
    return remove_components_from_name(cleaned_name_basic, general_components_list)

def preprocess_name_for_llm(name_already_globally_cleaned, freeform_components_list):
    return remove_components_from_name(name_already_globally_cleaned, freeform_components_list)


def get_qwen_llm_verification(osm_name_for_llm_input, gers_name_for_llm_input):
    if llm_model is None or llm_tokenizer is None: return False
    prompt = QWEN_USER_INSTRUCTION_PROMPT_TEMPLATE.format(osm_name=osm_name_for_llm_input, gers_name=gers_name_for_llm_input)

    if QWEN_DEFAULT_SYSTEM_PROMPT:
        full_prompt = f"<|im_start|>system\n{QWEN_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else: 
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    try:
        target_device = next(llm_model.parameters()).device
        inputs = llm_tokenizer(full_prompt, return_tensors="pt", padding=True).to(target_device)
        
        generate_kwargs = {
            "max_new_tokens": 10,
            "pad_token_id": llm_tokenizer.pad_token_id,
            "eos_token_id": llm_tokenizer.eos_token_id,
            "temperature": 0.0, 
            "do_sample": False 
        }
            
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, **generate_kwargs)
            
        reply = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip().lower()
        if "yes" in reply and "no" not in reply: return True
        if "no" in reply and "yes" not in reply: return False
        print(f"[Warning] Ambiguous Qwen reply for ('{osm_name_for_llm_input}', '{gers_name_for_llm_input}'): '{reply}'. Defaulting to False.")
        return False
    except Exception as e: print(f"[Error] Qwen LLM verification for ('{osm_name_for_llm_input}', '{gers_name_for_llm_input}'): {e}"); traceback.print_exc(); return False


def run_matching_pipeline(run_output_path):
    print(f"--- Starting Matching Pipeline for data in: {run_output_path} ---")
    load_all_models()

    osm_geojson_path = os.path.join(run_output_path, "osm_data", "osm_features.geojson")
    overture_geojson_path = os.path.join(run_output_path, "overture_places_data", "overture_places.geojson")
    address_components_csv_path = os.path.join(run_output_path, "overture_address_components", "overture_extracted_address_components.csv")

    print("Loading datasets...")
    try:
        if not os.path.exists(osm_geojson_path): print(f"ERROR: OSM GeoJSON not found: {osm_geojson_path}"); return
        osm_gdf = gpd.read_file(osm_geojson_path)
        if not os.path.exists(overture_geojson_path): print(f"ERROR: Overture GeoJSON not found: {overture_geojson_path}"); return
        overture_gdf = gpd.read_file(overture_geojson_path)
        
        address_components_df = None
        general_cleaning_components, freeform_components_for_llm = [], []
        if os.path.exists(address_components_csv_path):
            address_components_df = pd.read_csv(address_components_csv_path)
            general_cleaning_cols = ['locality', 'street', 'district', 'country']
            llm_freeform_cols = ['address_freeform_cleaned']
            general_cleaning_components = get_address_components_from_cols(address_components_df, general_cleaning_cols)
            freeform_components_for_llm = get_address_components_from_cols(address_components_df, llm_freeform_cols)
            print(f"Loaded {len(general_cleaning_components)} general components and {len(freeform_components_for_llm)} freeform components for cleaning.")
        else:
            print(f"Warning: Address components CSV not found at {address_components_csv_path}. Name cleaning will be basic.")
    except Exception as e: print(f"Error loading data: {e}"); return

    if osm_gdf.empty or overture_gdf.empty: print("OSM or Overture GDF empty."); return

    osm_gdf['orig_geom_4326'], overture_gdf['orig_geom_4326'] = osm_gdf.geometry, overture_gdf.geometry
    osm_gdf, overture_gdf = (gdf.set_crs("EPSG:4326", allow_override=True).to_crs("EPSG:3857") for gdf in [osm_gdf, overture_gdf])
    overture_sindex = overture_gdf.sindex
    print("Data reprojected & Overture spatial index created.")

    all_match_results = []
    processed_osm_count = 0

    print(f"Starting matching for {len(osm_gdf)} OSM places...")
    for osm_idx, osm_row in osm_gdf.iterrows():
        processed_osm_count += 1
        if processed_osm_count % 20 == 0: print(f"Processed {processed_osm_count}/{len(osm_gdf)} OSM places...")

        osm_name_original = extract_osm_name_raw(osm_row)
        osm_name_globally_cleaned = preprocess_name_globally(osm_name_original, general_cleaning_components)
        
        osm_geom_proj, osm_geom_wgs = osm_row.geometry, osm_row.orig_geom_4326
        osm_id = str(osm_row.get('osmid', osm_row.get('id', osm_row.get('element_type', '') + str(osm_row.get('osmid', osm_idx)))))
        if isinstance(osm_id, list): osm_id = "_".join(map(str, osm_id))
        
        if not osm_name_globally_cleaned: continue

        buffer = osm_geom_proj.buffer(MAX_DISTANCE_METERS)
        cand_idx = list(overture_sindex.intersection(buffer.bounds))
        
        if not cand_idx:
            actual_cands = gpd.GeoDataFrame(columns=overture_gdf.columns, crs=overture_gdf.crs) 
        else:
            actual_cands = overture_gdf.iloc[cand_idx][overture_gdf.iloc[cand_idx].distance(osm_geom_proj) <= MAX_DISTANCE_METERS].copy()
        
        osm_lat, osm_lon = (osm_geom_wgs.centroid.y, osm_geom_wgs.centroid.x) if osm_geom_wgs and not osm_geom_wgs.is_empty else (None, None)

        if actual_cands.empty:
            all_match_results.append({
                'osm_id': osm_id, 'osm_name_original': osm_name_original, 
                'osm_name_globally_cleaned': osm_name_globally_cleaned, 
                'osm_name_llm_input': None, 
                'osm_lat': osm_lat, 'osm_lon': osm_lon, 
                'overture_id': None, 'overture_name_original': None, 
                'overture_name_globally_cleaned': None,
                'overture_name_llm_input': None,
                'overture_lat': None, 'overture_lon': None,
                'distance_m': None, 'fuzzy_score': 0,
                'xg_match': False, 'xg_confidence': 0.0,
                'st_similarity': 0.0, 'st_match': False,
                'llm_called': False, 'llm_match': None,
                'match_confidence_level': 0, 'match_confidence_reason': "No Overture candidates in radius"
            })
            continue
            
        best_cand_for_osm = {'match_confidence_level': -1, 'distance_m': MAX_DISTANCE_METERS + 1, 'fuzzy_score': -1, 'st_similarity': -1}

        for _, ov_row in actual_cands.iterrows():
            res = {'osm_id': osm_id, 'osm_name_original': osm_name_original, 'osm_name_globally_cleaned': osm_name_globally_cleaned, 'osm_lat': osm_lat, 'osm_lon': osm_lon}
            
            ov_name_orig_field = ov_row.get('names')
            ov_primary_raw = extract_overture_primary_name_raw(ov_name_orig_field)
            ov_name_globally_cleaned = preprocess_name_globally(ov_primary_raw, general_cleaning_components)
            
            ov_id = ov_row.get('id', _)
            dist = osm_geom_proj.distance(ov_row.geometry)
            ov_geom_wgs = ov_row.orig_geom_4326
            ov_lat, ov_lon = (ov_geom_wgs.centroid.y, ov_geom_wgs.centroid.x) if ov_geom_wgs and not ov_geom_wgs.is_empty else (None, None)

            res.update({'overture_id': ov_id, 'overture_name_original': ov_name_orig_field, 'overture_name_globally_cleaned': ov_name_globally_cleaned, 'overture_lat': ov_lat, 'overture_lon': ov_lon, 'distance_m': round(dist, 2)})

            if not ov_name_globally_cleaned: continue 

            res.update({'fuzzy_score': 0, 'xg_match': False, 'xg_confidence': 0.0, 'st_similarity': 0.0, 'st_match': False, 
                        'osm_name_llm_input': None, 'overture_name_llm_input': None, 
                        'llm_called': False, 'llm_match': None, 'match_confidence_level': 0, 'match_confidence_reason': "Init"})

            if osm_name_globally_cleaned == ov_name_globally_cleaned:
                res.update({'fuzzy_score': 100, 'xg_match': True, 'xg_confidence': 1.0, 'st_similarity': 1.0, 'st_match': True, 'match_confidence_level': 3, 'match_confidence_reason': "Direct string match (globally cleaned)"})
                if res['distance_m'] < best_cand_for_osm.get('distance_m', MAX_DISTANCE_METERS +1): best_cand_for_osm = res.copy()
                if best_cand_for_osm['match_confidence_level'] == 3: break 
            
            general_fuzzy = WRatio(osm_name_globally_cleaned, ov_name_globally_cleaned)
            res['fuzzy_score'] = round(general_fuzzy, 2)
            if general_fuzzy < MIN_FUZZY_THRESHOLD: 
                res['match_confidence_reason'] = "Fuzzy score below threshold (globally cleaned)"
                if best_cand_for_osm['match_confidence_level'] < 0 and best_cand_for_osm.get('fuzzy_score', -1) < general_fuzzy: best_cand_for_osm = res.copy()
                continue

            xg_match, xg_conf = get_xgboost_prediction(osm_name_globally_cleaned, ov_name_globally_cleaned, osm_row, ov_row)
            res.update({'xg_match': xg_match, 'xg_confidence': round(float(xg_conf), 4)})
            
            st_sim = get_sentence_similarity(osm_name_globally_cleaned, ov_name_globally_cleaned)
            st_match = st_sim >= SENTENCE_MODEL_CONFIDENCE_THRESHOLD
            res.update({'st_similarity': round(st_sim, 4), 'st_match': st_match})

            llm_call, llm_val = False, None
            osm_llm_in_val, ov_llm_in_val = osm_name_globally_cleaned, ov_name_globally_cleaned 

            if xg_match != st_match:
                llm_call = True
                osm_llm_in_val = preprocess_name_for_llm(osm_name_globally_cleaned, freeform_components_for_llm)
                ov_llm_in_val = preprocess_name_for_llm(ov_name_globally_cleaned, freeform_components_for_llm)
                llm_val = get_qwen_llm_verification(osm_llm_in_val, ov_llm_in_val) if osm_llm_in_val.strip() and ov_llm_in_val.strip() else False
            
            res.update({'osm_name_llm_input': osm_llm_in_val if llm_call else None, 
                        'overture_name_llm_input': ov_llm_in_val if llm_call else None,
                        'llm_called': llm_call, 'llm_match': llm_val})

            lvl, reason = 0, "Default no match"
            if res.get('match_confidence_level') == 3: 
                lvl, reason = 3, res['match_confidence_reason']
            elif res['xg_match'] and res['xg_confidence'] >= XGBOOST_CONFIDENT_THRESHOLD and res['st_match']: lvl, reason = 3, "XGBoost (Confident) & ST Agree"
            elif llm_call: lvl, reason = (2, "XG/ST Disagree, LLM Confirmed") if llm_val else (1, "XG/ST Disagree, LLM Denied")
            elif res['xg_match'] and res['st_match']: lvl, reason = 1, "XGBoost (Low Conf) & ST Agree"
            elif not llm_call: 
                if not res['xg_match'] and not res['st_match']: lvl, reason = 0, "Neither XG nor ST matched (models agreed)"
                else: lvl, reason = 0, "Single model weak agreement, no LLM"


            res.update({'match_confidence_level': lvl, 'match_confidence_reason': reason})
            
            if lvl > 0:
                print(f"  MATCH FOUND (OSM ID: {osm_id}, Overture ID: {ov_id}): Level {lvl} - {reason}")
                print(f"    OSM: '{osm_name_original}' -> '{osm_name_globally_cleaned}'")
                print(f"    OVT: '{extract_overture_primary_name_raw(ov_name_orig_field)}' -> '{ov_name_globally_cleaned}'")
                if llm_call:
                    print(f"    LLM In: OSM='{osm_llm_in_val}', OVT='{ov_llm_in_val}', LLM Out: {'Yes' if llm_val else 'No'}")


            if lvl > best_cand_for_osm['match_confidence_level'] or \
               (lvl == best_cand_for_osm['match_confidence_level'] and lvl > 0 and res['distance_m'] < best_cand_for_osm['distance_m']) or \
               (lvl == best_cand_for_osm['match_confidence_level'] and lvl > 0 and res['distance_m'] == best_cand_for_osm['distance_m'] and res['fuzzy_score'] > best_cand_for_osm.get('fuzzy_score', -1)):
                best_cand_for_osm = res.copy()
        
        if best_cand_for_osm['match_confidence_level'] != -1: all_match_results.append(best_cand_for_osm)
        elif not actual_cands.empty and best_cand_for_osm['match_confidence_level'] == -1: 
            fc_row = actual_cands.iloc[0]
            fc_name_orig, fc_primary_raw = fc_row.get('names'), extract_overture_primary_name_raw(fc_row.get('names'))
            fc_globally_cleaned = preprocess_name_globally(fc_primary_raw, general_cleaning_components)
            fc_geom_wgs, fc_lat, fc_lon = fc_row.orig_geom_4326, None, None
            if fc_geom_wgs and not fc_geom_wgs.is_empty: fc_lon, fc_lat = fc_geom_wgs.centroid.x, fc_geom_wgs.centroid.y
            all_match_results.append({'osm_id': osm_id, 'osm_name_original': osm_name_original, 'osm_name_globally_cleaned': osm_name_globally_cleaned, 'osm_name_llm_input': None, 'osm_lat': osm_lat, 'osm_lon': osm_lon, 'overture_id': fc_row.get('id',0), 'overture_name_original': fc_name_orig, 'overture_name_globally_cleaned': fc_globally_cleaned, 'overture_name_llm_input': None, 'overture_lat': fc_lat, 'overture_lon': fc_lon, 'distance_m': round(osm_geom_proj.distance(fc_row.geometry),2), 'fuzzy_score': WRatio(osm_name_globally_cleaned, fc_globally_cleaned), 'match_confidence_level': 0, 'match_confidence_reason': "All cands below fuzzy or empty names (globally cleaned)"})

    out_csv = os.path.join(run_output_path, "matches_with_confidence.csv")
    results_df = pd.DataFrame(all_match_results)
    
    results_df.to_csv(out_csv, index=False)
    print(f"Matching done. Results: {out_csv}")
    
    if not results_df.empty:
        print("\n--- Match Statistics ---")
        total_osm_considered_for_pairing = len(results_df['osm_id'].unique()) 
        print(f"Total OSM places considered for pairing (had candidates or logged as no candidate): {total_osm_considered_for_pairing}")

        actual_matches_df = results_df[results_df['match_confidence_level'] > 0]
        total_actual_matches = len(actual_matches_df)
        print(f"Total actual matches found (Confidence > 0): {total_actual_matches}")

        if total_actual_matches > 0:
            match_reason_counts = actual_matches_df['match_confidence_reason'].value_counts()
            print("\nBreakdown of Match Reasons (for Confidence > 0):")
            for reason, count in match_reason_counts.items():
                percentage = (count / total_actual_matches) * 100
                print(f"- {reason}: {count} ({percentage:.2f}%)")
        else:
            print("No actual matches (Confidence > 0) to provide a breakdown for.")

        print("\nOverall Confidence Level Distribution (all processed pairs):")
        print(results_df['match_confidence_level'].value_counts(dropna=False).sort_index())
        print("\nSample of results:")
        print(results_df[['osm_name_original','osm_name_globally_cleaned', 'overture_name_original', 'overture_name_globally_cleaned', 'match_confidence_level', 'match_confidence_reason']].head())

    else:
        print("Warning: No matches or processed pairs were recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OSM-Overture matching pipeline.")
    parser.add_argument("run_output_path", type=str, help="Path to the directory containing downloaded data.")
    args = parser.parse_args()
    run_matching_pipeline(args.run_output_path)

