import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# --- Configuration ---
INPUT_CSV_FILE = "cleaned.csv"  # Your input CSV file
OUTPUT_CSV_FILE = "cleaned_with_embeddings.csv" # Output file with embeddings

# Define the columns that contain the names for which to generate embeddings
NAME_COLUMN_1 = 'osm_name'
NAME_COLUMN_2 = 'gers_name'

# Choose a pre-trained Sentence Transformer model
# 'all-MiniLM-L6-v2' is a good general-purpose model: fast and good quality.
# Other options: 'paraphrase-MiniLM-L6-v2', 'distilbert-base-nli-stsb-mean-tokens', etc.
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def generate_embeddings(df, text_column_name, model, new_embedding_column_name):
    """
    Generates embeddings for a specified text column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column_name (str): The name of the column containing text data.
        model (SentenceTransformer): The initialized SentenceTransformer model.
        new_embedding_column_name (str): The name for the new column to store embeddings.
    """
    if text_column_name not in df.columns:
        print(f"Error: Column '{text_column_name}' not found in the DataFrame.")
        return df

    print(f"Generating embeddings for column: '{text_column_name}'...")
    
    # Ensure all entries are strings, handle potential NaNs by converting them to empty strings
    texts = df[text_column_name].fillna('').astype(str).tolist()
    
    # Generate embeddings. The model.encode() method can take a list of sentences.
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Add embeddings as a new column. Each embedding will be a list/numpy array.
    df[new_embedding_column_name] = list(embeddings) # Store as list of numpy arrays
    
    print(f"Embeddings generated and stored in column: '{new_embedding_column_name}'")
    return df

def main():
    print(f"--- Embedding Generation Script ---")
    print(f"Using model: {MODEL_NAME}")

    # 1. Load the dataset
    print(f"\nLoading data from '{INPUT_CSV_FILE}'...")
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found.")
        return

    try:
        df = pd.read_csv(INPUT_CSV_FILE)
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Verify necessary name columns exist
    if NAME_COLUMN_1 not in df.columns or NAME_COLUMN_2 not in df.columns:
        print(f"Error: Required name columns ('{NAME_COLUMN_1}' and/or '{NAME_COLUMN_2}') not found in the CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # 2. Initialize the Sentence Transformer model
    print(f"\nLoading Sentence Transformer model: '{MODEL_NAME}'...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Sentence Transformer model: {e}")
        print("Please ensure the model name is correct and you have an internet connection "
              "if downloading for the first time.")
        return

    # 3. Generate embeddings for the first name column
    df = generate_embeddings(df, NAME_COLUMN_1, model, f"{NAME_COLUMN_1}_embedding")

    # 4. Generate embeddings for the second name column
    df = generate_embeddings(df, NAME_COLUMN_2, model, f"{NAME_COLUMN_2}_embedding")

    # 5. Display sample of the DataFrame with new embedding columns
    if f"{NAME_COLUMN_1}_embedding" in df.columns:
        print(f"\nSample of DataFrame with embeddings (first 3 rows):")
        # To prevent overly long output for embeddings, we'll show their shape/type
        for col_name in [f"{NAME_COLUMN_1}_embedding", f"{NAME_COLUMN_2}_embedding"]:
            if col_name in df.columns and not df.empty:
                first_embedding = df[col_name].iloc[0]
                if isinstance(first_embedding, np.ndarray):
                    print(f"  Column '{col_name}' example embedding shape: {first_embedding.shape}, dtype: {first_embedding.dtype}")
                else:
                     print(f"  Column '{col_name}' example embedding type: {type(first_embedding)}")
        print(df[[NAME_COLUMN_1, f"{NAME_COLUMN_1}_embedding", NAME_COLUMN_2, f"{NAME_COLUMN_2}_embedding"]].head(3))


    # 6. Save the DataFrame with embeddings to a new CSV
    # Note: Saving NumPy arrays in CSV cells converts them to string representations.
    # For more efficient storage and retrieval of embeddings, consider formats like Parquet, HDF5, or saving embeddings separately as .npy files.
    print(f"\nSaving DataFrame with embeddings to '{OUTPUT_CSV_FILE}'...")
    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        print(f"Successfully saved data with embeddings to '{OUTPUT_CSV_FILE}'")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
