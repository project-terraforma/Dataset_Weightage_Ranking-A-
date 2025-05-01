import pandas as pd
from math import radians
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
from rapidfuzz.fuzz import token_set_ratio


# Load place data
df = pd.read_parquet("foursquare_data/train.parquet")
pairs = pd.read_parquet("foursquare_data/pairs.parquet")

# Ensure IDs are strings
pairs['id_1'] = pairs['id_1'].astype(str)
pairs['id_2'] = pairs['id_2'].astype(str)
df['id'] = df['id'].astype(str)

# Label encode only relevant categorical features
categorical_features = ['address', 'city', 'state', 'zip', 'country' , 'categories']
df[categorical_features] = df[categorical_features].apply(LabelEncoder().fit_transform)

# Merge pair data
df1 = df.rename(columns=lambda x: x + "_1")
df2 = df.rename(columns=lambda x: x + "_2")
merged = pairs.merge(df1, left_on="id_1", right_on="id_1").merge(df2, left_on="id_2", right_on="id_2")

# Plot match distribution
unique, counts = np.unique(merged['match'], return_counts=True)
sns.barplot(x=unique, y=counts)
plt.xlabel("Match (Target Class)")
plt.ylabel("Number of Samples")
plt.title("Distribution of Match vs Non-Match in Merged Dataset")
# plt.show()

# Convert name columns to string
merged['name_1_x'] = merged['name_1_x'].astype(str)
merged['name_2_y'] = merged['name_2_y'].astype(str)

# Feature: haversine distance
def haversine(row):
    coord1 = (radians(row['latitude_1_x']), radians(row['longitude_1_x']))
    coord2 = (radians(row['latitude_2_y']), radians(row['longitude_2_y']))
    return haversine_distances([coord1, coord2])[0][1] * 6371000  # meters

merged['distance_m'] = merged.apply(haversine, axis=1)

# Feature: token overlap
merged['token_overlap'] = merged.apply(
    lambda row: len(set(row['name_1_x'].lower().split()) & set(row['name_2_y'].lower().split())),
    axis=1
)

# Feature: name length difference
merged['name_length_diff'] = abs(merged['name_1_x'].str.len() - merged['name_2_y'].str.len())

# Feature: first word match
merged['first_word_match'] = (
    merged['name_1_x'].str.split().str[0].str.lower() ==
    merged['name_2_y'].str.split().str[0].str.lower()
).astype(int)

# Create fuzzy match score feature
merged['fuzzy_score'] = merged.apply(
    lambda row: token_set_ratio(str(row['name_1_x']), str(row['name_2_y'])),
    axis=1
)


# ✅ NEW: include raw lat/lon as features
feature_cols = [
    'distance_m',
    'token_overlap',
    'name_length_diff',
    'first_word_match',
    'fuzzy_score',
    'latitude_1_x', 'longitude_1_x',
    'latitude_2_y', 'longitude_2_y'
]

X = merged[feature_cols]
y = merged['match'].astype(int)

# # Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(class_weight='balanced', random_state=42)
# model.fit(X_train, y_train)


# Predict and evaluate
# y_pred = model.predict(X_val)
# print(classification_report(y_val, y_pred))
neg, pos = np.bincount(y_train)
scale_pos_weight = (neg / pos) * 1.1  # slight boost for class 0

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,   # optional, useful if you have class imbalance
    random_state=42
)

model.fit(X_train, y_train)
# Predict probabilities instead of labels
y_proba = model.predict_proba(X_val)[:, 1]

# Define abstention thresholds
def selective_predict(p, low=0.25, high=0.75):
    if p >= high:
        return 1  # match
    elif p <= low:
        return 0  # non-match
    else:
        return -1  # abstain

# Apply abstention logic
y_selective = np.array([selective_predict(p) for p in y_proba])

# Filter confident predictions
confident_mask = y_selective != -1
num_abstained = (~confident_mask).sum()

# Show report
print(f"\n🧠 Selective Prediction Thresholds: low=0.25, high=0.75")
print(f"🕳️  Abstained on {num_abstained} of {len(y_val)} samples ({(num_abstained / len(y_val)) * 100:.2f}%)")
print("\n📊 Performance on Confident Predictions:")
print(classification_report(y_val[confident_mask], y_selective[confident_mask]))

joblib.dump(model, "name_match_xgm.pkl")
# Re-attach labels and predictions
# X_val_copy = X_val.copy()
# X_val_copy['true_match'] = y_val.reset_index(drop=True)
# X_val_copy['predicted_match'] = y_pred

# # Add back original names from merged
# X_val_copy['name_1_x'] = merged.loc[X_val_copy.index, 'name_1_x'].values
# X_val_copy['name_2_y'] = merged.loc[X_val_copy.index, 'name_2_y'].values

# # Filter for false positives
# false_positives = X_val_copy[(X_val_copy['true_match'] == 0) & (X_val_copy['predicted_match'] == 1)]

# # Save to CSV
# false_positives.to_csv("false_positive_matches.csv", index=False)

# # Optional: preview in terminal
# print("\n🔍 False Positives (predicted match, actually not):")
# print(false_positives[['name_1_x', 'name_2_y', 'distance_m', 'token_overlap', 'name_length_diff', 'first_word_match']].head(10))
# print("\n💾 Saved to: false_positive_matches.csv")


