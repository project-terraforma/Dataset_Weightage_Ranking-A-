import pandas as pd
from math import radians # Not strictly needed anymore but kept for haversine_distances import
from sklearn.metrics.pairwise import haversine_distances, cosine_similarity # Added cosine_similarity
from sklearn.preprocessing import LabelEncoder # Not used for features, but part of original imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
from rapidfuzz.fuzz import token_set_ratio
import os
import ast # For parsing stringified lists/arrays from CSV
import re # For robust parsing of embedding strings

# --- Configuration ---
# Column names from your cleaned_with_embeddings.csv
NAME1_COL = 'osm_name'
NAME2_COL = 'gers_name'
LABEL_COL = 'verified_label' # Assuming this is still the label column
EMBEDDING_COL1 = 'osm_name_embedding' # Column name for the first embedding
EMBEDDING_COL2 = 'gers_name_embedding' # Column name for the second embedding


# --- 1. Data Loading ---
print("--- 1. Data Loading ---")
# The input file should now be the one containing pre-computed embeddings
input_csv_file = input("Enter the path to your CSV file with embeddings (e.g., cleaned_with_embeddings.csv): ")
# Example: "cleaned_with_embeddings.csv"

try:
    df_cleaned = pd.read_csv(input_csv_file)
    print(f"Successfully loaded '{input_csv_file}'. Shape: {df_cleaned.shape}")
except FileNotFoundError:
    print(f"ERROR: CSV file not found at '{input_csv_file}'. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR: Could not read CSV file. Reason: {e}")
    exit()

# Validate required columns
required_cols = [NAME1_COL, NAME2_COL, LABEL_COL, EMBEDDING_COL1, EMBEDDING_COL2]
missing_cols = [col for col in required_cols if col not in df_cleaned.columns]
if missing_cols:
    print(f"ERROR: Input CSV is missing required columns for names, labels, or embeddings: {', '.join(missing_cols)}.")
    print(f"Expected columns like: {NAME1_COL}, {NAME2_COL}, {LABEL_COL}, {EMBEDDING_COL1}, {EMBEDDING_COL2}")
    print(f"Found columns: {list(df_cleaned.columns)}")
    exit()

# Ensure names are strings and label is integer
df_cleaned[NAME1_COL] = df_cleaned[NAME1_COL].astype(str)
df_cleaned[NAME2_COL] = df_cleaned[NAME2_COL].astype(str)
df_cleaned[LABEL_COL] = df_cleaned[LABEL_COL].astype(int)

print(f"Dataset has {len(df_cleaned)} pairs.")
print("Value counts for the 'match_label':")
print(df_cleaned[LABEL_COL].value_counts())

# Plot match distribution from the loaded CSV
plt.figure(figsize=(8,6))
unique_labels, counts_labels = np.unique(df_cleaned[LABEL_COL], return_counts=True)
sns.barplot(x=unique_labels, y=counts_labels)
plt.xlabel("Match Label (0: Non-Match, 1: Match)")
plt.ylabel("Number of Samples")
plt.title(f"Distribution of Labels in '{os.path.basename(input_csv_file)}'")
plt.savefig("label_distribution_from_csv_with_embeddings.png")
print("Saved label distribution plot to label_distribution_from_csv_with_embeddings.png")


# --- 2. Feature Engineering ---
print("\n--- 2. Feature Engineering ---")
print("Engineering features from names and embeddings...")

# --- 2a. Parse Embedding Strings ---
# Embeddings read from CSV are strings; convert them back to lists/arrays of numbers.
def parse_embedding_string(embedding_str):
    if pd.isna(embedding_str) or not isinstance(embedding_str, str):
        return None
    
    s = embedding_str.strip()
    
    # Attempt to parse common numpy array string representations like '[ 0.1  0.2 -0.3 ]'
    if s.startswith('[') and s.endswith(']'):
        s_content = s[1:-1].strip() # Get content within brackets
        if not s_content: # Empty list representation like '[]'
            return np.array([])
        try:
            # Split by one or more whitespace characters, filter out empty strings that result from multiple spaces
            num_list = [float(x) for x in re.split(r'\s+', s_content) if x.strip()]
            if num_list: # Ensure we got some numbers
                return np.array(num_list)
        except ValueError:
            # If space-split parsing fails, it might be comma-separated or other format
            # Fall through to ast.literal_eval
            pass 

    # Try ast.literal_eval for standard Python list/tuple strings like '[1.0, 2.0, 3.0]' or '1.0,2.0,3.0'
    try:
        # ast.literal_eval is safer than eval
        evaluated = ast.literal_eval(s)
        if isinstance(evaluated, (list, tuple)):
             return np.array(evaluated)
        # If it evaluates to a single number (e.g. string was just "0.5"), wrap it in an array
        elif isinstance(evaluated, (int, float)):
             return np.array([evaluated])

    except (ValueError, SyntaxError, TypeError):
        # print(f"Warning: Could not parse embedding string with ast.literal_eval: {s[:70]}...")
        pass # Fall through if ast.literal_eval fails

    # Final fallback if all parsing attempts fail
    # print(f"Warning: All parsing attempts failed for embedding string: {s[:70]}...")
    return None


print(f"Parsing embeddings from column '{EMBEDDING_COL1}'...")
df_cleaned[EMBEDDING_COL1] = df_cleaned[EMBEDDING_COL1].apply(parse_embedding_string)
print(f"Parsing embeddings from column '{EMBEDDING_COL2}'...")
df_cleaned[EMBEDDING_COL2] = df_cleaned[EMBEDDING_COL2].apply(parse_embedding_string)

# Drop rows where embeddings could not be parsed (if any)
original_len = len(df_cleaned)
df_cleaned.dropna(subset=[EMBEDDING_COL1, EMBEDDING_COL2], inplace=True)
if len(df_cleaned) < original_len:
    print(f"Dropped {original_len - len(df_cleaned)} rows due to unparseable embeddings.")

if df_cleaned.empty:
    print("ERROR: No data left after attempting to parse embeddings. Please check your embedding columns.")
    exit()

# --- 2b. Lexical Features (from previous script) ---
df_cleaned['token_overlap'] = df_cleaned.apply(
    lambda row: len(set(str(row[NAME1_COL]).lower().split()) & set(str(row[NAME2_COL]).lower().split())),
    axis=1
)
df_cleaned['name_length_diff'] = abs(df_cleaned[NAME1_COL].str.len() - df_cleaned[NAME2_COL].str.len())

def get_first_word(name_series):
    return name_series.str.split().str[0].str.lower().fillna('')

df_cleaned['first_word_match'] = (
    get_first_word(df_cleaned[NAME1_COL]) == get_first_word(df_cleaned[NAME2_COL])
).astype(int)
df_cleaned['fuzzy_score'] = df_cleaned.apply(
    lambda row: token_set_ratio(str(row[NAME1_COL]), str(row[NAME2_COL])),
    axis=1
)

# --- 2c. New Feature: Cosine Similarity of Embeddings ---
def calculate_cosine_similarity(row):
    emb1 = row[EMBEDDING_COL1]
    emb2 = row[EMBEDDING_COL2]
    # Check if embeddings are valid numpy arrays and not empty
    if isinstance(emb1, np.ndarray) and emb1.size > 0 and \
       isinstance(emb2, np.ndarray) and emb2.size > 0:
        if emb1.shape == emb2.shape: # Ensure shapes are compatible for cosine similarity
            return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
        else:
            # print(f"Warning: Embeddings have mismatched shapes: {emb1.shape} vs {emb2.shape}. Returning 0 similarity.")
            return 0.0 # Mismatched shapes
    return 0.0 # One or both embeddings are None, empty, or not arrays

print("Calculating cosine similarity from embeddings...")
df_cleaned['embedding_cosine_similarity'] = df_cleaned.apply(calculate_cosine_similarity, axis=1)


# Define the list of feature columns to be used for training the model
feature_cols = [
    'token_overlap',
    'name_length_diff',
    'first_word_match',
    'fuzzy_score',
    'embedding_cosine_similarity' # Added the new feature
]
print(f"Using feature columns: {feature_cols}")

# Prepare X (features) and y (target labels)
X = df_cleaned[feature_cols]
y = df_cleaned[LABEL_COL]

# --- 3. Train/Test Split ---
# (Identical to previous script)
print("\n--- 3. Train/Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# ... (print shapes and distributions) ...
print(f"Data split into training and testing sets:")
print(f"Shape of X_train (training features): {X_train.shape}")
print(f"Shape of y_train (training labels): {y_train.shape}")
print(f"Shape of X_test (testing features): {X_test.shape}")
print(f"Shape of y_test (testing labels): {y_test.shape}")
print("\nDistribution of labels in training set:")
print(y_train.value_counts(normalize=True))
print("\nDistribution of labels in testing set:")
print(y_test.value_counts(normalize=True))


# --- 4. Model Training (XGBoost) ---
print("\n--- 4. Model Training ---")
neg_train, pos_train = np.bincount(y_train)
if pos_train == 0 or neg_train == 0: 
    print("Warning: One class is absent in y_train. Using scale_pos_weight = 1.")
    scale_pos_weight = 1
else:
    scale_pos_weight = (neg_train / pos_train)
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.4f} (neg_train: {neg_train}, pos_train: {pos_train})")

# --- Hyperparameter Tuning with GridSearchCV ---
print("\n--- 4.1 Hyperparameter Tuning using GridSearchCV ---")
param_grid = {
    'n_estimators': [100, 200], # Reduced for speed, expand as needed
    'max_depth': [3, 5, 7],      # Reduced for speed
    'learning_rate': [0.05, 0.1], # Reduced for speed
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
xgb_model_for_grid = XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False
)
grid_search = GridSearchCV(
    estimator=xgb_model_for_grid, param_grid=param_grid, scoring='f1',
    cv=3, verbose=1, n_jobs=-1
)
print("Starting GridSearchCV... This may take some time.")
grid_search.fit(X_train, y_train)
print("\nGridSearchCV Best Parameters found:")
print(grid_search.best_params_)
print(f"GridSearchCV Best F1-score: {grid_search.best_score_:.4f}")
model = grid_search.best_estimator_
print("\nModel training with best parameters complete (or retrieved from GridSearchCV).")

# --- 5. Prediction and Evaluation ---
# (Identical to previous script, just uses the new 'model')
print("\n--- 5. Prediction and Evaluation ---")
y_proba_test = model.predict_proba(X_test)[:, 1] 
y_pred_test = model.predict(X_test)       

def selective_predict(p, low=0.25, high=0.75):
    if p >= high: return 1
    elif p <= low: return 0
    else: return -1

y_selective = np.array([selective_predict(p) for p in y_proba_test])
confident_mask = y_selective != -1 
num_abstained = (~confident_mask).sum()

if len(y_test[confident_mask]) == 0 and num_abstained == len(y_test): 
    print("Warning: Model abstained on all test samples. Selective report cannot be generated.")
elif len(y_test[confident_mask]) == 0: 
     print("Warning: No confident predictions made. Selective report cannot be generated.")
else:
    print(f"\n🧠 Selective Prediction Thresholds: low=0.25, high=0.75")
    print(f"🕳️  Abstained on {num_abstained} of {len(y_test)} test samples ({(num_abstained / len(y_test)) * 100:.2f}%)")
    print("\n📊 Performance on Confident Predictions (Test Set):")
    print(classification_report(y_test[confident_mask], y_selective[confident_mask]))

print("\n📊 Performance on ALL Predictions (Test Set, no abstention):")
print(classification_report(y_test, y_pred_test))


# --- 6. Save Model ---
print("\n--- 6. Save Model ---")
model_filename = "name_match_xgboost_csv_with_embeddings.pkl" # Updated filename
joblib.dump(model, model_filename)
print(f"Trained XGBoost model saved to {model_filename}")


# --- 7. Error Analysis (False Positives & False Negatives) ---
# Create test_results_df once
test_results_df = pd.DataFrame(index=X_test.index) 
test_results_df['true_match_label'] = y_test
test_results_df['predicted_match_label'] = y_pred_test
test_results_df['predicted_probability_match'] = y_proba_test 
test_results_df['predicted_probability_non_match'] = 1 - y_proba_test 

# --- 7a. False Positive Analysis ---
print("\n--- 7a. Analyzing False Positives (Non-match misses on Test Set) ---")
false_positive_indices = test_results_df[
    (test_results_df['true_match_label'] == 0) & (test_results_df['predicted_match_label'] == 1)
].index
print(f"Number of false positives identified in test set: {len(false_positive_indices)}")

if not false_positive_indices.empty:
    false_positives_original_data = df_cleaned.loc[false_positive_indices, [NAME1_COL, NAME2_COL, LABEL_COL]].copy()
    fp_predictions_info = test_results_df.loc[false_positive_indices, ['predicted_match_label', 'predicted_probability_match']]
    false_positives_details_df = false_positives_original_data.join(fp_predictions_info)
    false_positives_details_df = false_positives_details_df.join(X_test.loc[false_positive_indices]) # Join features
    false_positives_details_df.rename(columns={LABEL_COL: 'true_label_from_input_csv'}, inplace=True)

    fp_csv_filename = "false_positives_csv_with_embeddings.csv" # Updated filename
    try:
        false_positives_details_df.to_csv(fp_csv_filename, index=False)
        print(f"False positives details saved to: {fp_csv_filename}")
        print("\n🔍 Preview of False Positives:")
        preview_cols = [NAME1_COL, NAME2_COL, 'fuzzy_score', 'embedding_cosine_similarity', 'predicted_probability_match', 'true_label_from_input_csv']
        existing_preview_cols = [col for col in preview_cols if col in false_positives_details_df.columns]
        print(false_positives_details_df[existing_preview_cols].head())
    except Exception as e:
        print(f"Error saving false positives CSV: {e}")
else:
    print("No false positives found in the test set.")

# --- 7b. False Negative Analysis ---
print("\n--- 7b. Analyzing False Negatives (Match misses on Test Set) ---")
false_negative_indices = test_results_df[
    (test_results_df['true_match_label'] == 1) & (test_results_df['predicted_match_label'] == 0)
].index
print(f"Number of false negatives identified in test set: {len(false_negative_indices)}")

if not false_negative_indices.empty:
    false_negatives_original_data = df_cleaned.loc[false_negative_indices, [NAME1_COL, NAME2_COL, LABEL_COL]].copy()
    fn_predictions_info = test_results_df.loc[false_negative_indices, ['predicted_match_label', 'predicted_probability_non_match', 'predicted_probability_match']]
    false_negatives_details_df = false_negatives_original_data.join(fn_predictions_info)
    false_negatives_details_df = false_negatives_details_df.join(X_test.loc[false_negative_indices]) # Join features
    false_negatives_details_df.rename(columns={LABEL_COL: 'true_label_from_input_csv'}, inplace=True)

    fn_csv_filename = "false_negatives_csv_with_embeddings.csv" # Updated filename
    try:
        false_negatives_details_df.to_csv(fn_csv_filename, index=False)
        print(f"False negatives details saved to: {fn_csv_filename}")
        print("\n🔍 Preview of False Negatives:")
        preview_cols_fn = [NAME1_COL, NAME2_COL, 'fuzzy_score', 'embedding_cosine_similarity', 'predicted_probability_non_match', 'true_label_from_input_csv']
        existing_preview_cols_fn = [col for col in preview_cols_fn if col in false_negatives_details_df.columns]
        print(false_negatives_details_df[existing_preview_cols_fn].head())
    except Exception as e:
        print(f"Error saving false negatives CSV: {e}")
else:
    print("No false negatives found in the test set.")

# --- 8. Optional: Feature Importance ---
print("\n--- 8. Feature Importance ---")
try:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}) # feature_cols now includes embedding_cosine_similarity
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        print(feature_importance_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('XGBoost Feature Importance (Trained with Embeddings)') # Updated title
        plt.tight_layout()
        plt.savefig("feature_importance_csv_with_embeddings.png") # Updated filename
        print("Saved feature importance plot to feature_importance_csv_with_embeddings.png")
    else:
        print("Could not retrieve feature importances from the tuned model.")
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")

print("\n--- Script Finished ---")
