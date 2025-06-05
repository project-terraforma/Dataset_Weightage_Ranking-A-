import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator # Added for evaluation
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # Added for splitting data
import math
import os # For output path creation
import numpy as np # For evaluation metrics if needed later

# --- Configuration ---
csv_filepath = "cleaned.csv"  # Path to your CSV file with verified labels
name1_column = 'osm_name'
name2_column = 'gers_name'
label_column = 'verified_label' # Your 0 or 1 integer labels

# Model and Training Parameters
base_model_name = 'all-MiniLM-L6-v2' # A good starting model
output_model_path = './fine_tuned_name_matcher_online_contrastive' # Where the fine-tuned model will be saved
num_epochs = 5  # Start with a small number of epochs (e.g., 1-5)
train_batch_size = 16 # Batch size for training (can be 8, 16, 32 depending on GPU memory)
learning_rate = 2e-5 # Common learning rate for fine-tuning transformers
contrastive_margin = 0.5 # Margin for OnlineContrastiveLoss
test_set_size = 0.2 # Proportion of data to use for the test/evaluation set

def main():
    print("--- Sentence Transformer Fine-tuning and Evaluation for Name Matching ---")

    # --- 1. Load Your Verified Name Matching Data ---
    print(f"\nLoading dataset from: {csv_filepath}")
    try:
        df_full = pd.read_csv(csv_filepath)
        # Ensure labels are integers (0 or 1) and handle potential NaNs
        df_full[label_column] = pd.to_numeric(df_full[label_column], errors='coerce')
        df_full.dropna(subset=[name1_column, name2_column, label_column], inplace=True)
        df_full[label_column] = df_full[label_column].astype(int)
        
        # Filter for only 0 and 1 labels if other values exist
        df_full = df_full[df_full[label_column].isin([0, 1])]

        print(f"Loaded {len(df_full)} rows after N/A removal and filtering for 0/1 labels.")
        if df_full.empty:
            print("ERROR: No data loaded after filtering. Exiting.")
            exit()
        print("Full dataset preview:")
        print(df_full.head())
        print("\nFull dataset label distribution:")
        print(df_full[label_column].value_counts(normalize=True))
    except FileNotFoundError:
        print(f"ERROR: File not found at '{csv_filepath}'. Please ensure the path is correct.")
        exit()
    except Exception as e:
        print(f"ERROR: Could not load or process the CSV file. Details: {e}")
        exit()

    # --- 1.1 Split Data into Training and Evaluation Sets ---
    print(f"\nSplitting data into training and evaluation sets (test_size={test_set_size})...")
    train_df, eval_df = train_test_split(
        df_full,
        test_size=test_set_size,
        random_state=42, # For reproducibility
        stratify=df_full[label_column] # Ensure similar class distribution in both sets
    )
    print(f"Training set size: {len(train_df)}")
    print(f"Evaluation set size: {len(eval_df)}")
    print("Training set label distribution:")
    print(train_df[label_column].value_counts(normalize=True))
    print("Evaluation set label distribution:")
    print(eval_df[label_column].value_counts(normalize=True))


    # --- 2. Choose and Load a Base Model to Fine-Tune ---
    print(f"\nLoading pre-trained Sentence Transformer model: {base_model_name}")
    try:
        model = SentenceTransformer(base_model_name)
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load the model '{base_model_name}'. Details: {e}")
        print("Ensure the model name is correct and you have an internet connection.")
        exit()

    # --- 3. Prepare Training Data into InputExamples for OnlineContrastiveLoss ---
    print("\nPreparing training examples for OnlineContrastiveLoss...")
    train_examples_for_contrastive = []
    for index, row in train_df.iterrows(): # Use train_df here
        texts = [str(row[name1_column]), str(row[name2_column])]
        label = int(row[label_column]) 
        train_examples_for_contrastive.append(InputExample(texts=texts, label=label))

    if not train_examples_for_contrastive:
        print("ERROR: No training examples were created. Check your training data.")
        exit()
        
    print(f"Created {len(train_examples_for_contrastive)} InputExamples for training.")
    if train_examples_for_contrastive: # Check if list is not empty before accessing index 0
        print("Example of a training InputExample:")
        print(f"  Texts: {train_examples_for_contrastive[0].texts}")
        print(f"  Label: {train_examples_for_contrastive[0].label} (type: {type(train_examples_for_contrastive[0].label)})")


    # --- 4. Create DataLoader for Training Data ---
    print(f"\nCreating DataLoader for training with batch size {train_batch_size}...")
    train_dataloader_contrastive = DataLoader(train_examples_for_contrastive, 
                                              shuffle=True, 
                                              batch_size=train_batch_size)

    # --- 5. Define the Loss Function: OnlineContrastiveLoss ---
    print("\nDefining OnlineContrastiveLoss function...")
    distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    train_loss = losses.OnlineContrastiveLoss(model=model, 
                                              distance_metric=distance_metric, 
                                              margin=contrastive_margin)
    print(f"Using OnlineContrastiveLoss with margin={contrastive_margin} and distance_metric={distance_metric}")

    # --- 5.1 Prepare Evaluation Data ---
    print("\nPreparing evaluation data...")
    eval_sentences1 = eval_df[name1_column].astype(str).tolist()
    eval_sentences2 = eval_df[name2_column].astype(str).tolist()
    eval_labels = eval_df[label_column].tolist()

    evaluator = BinaryClassificationEvaluator(
        eval_sentences1, 
        eval_sentences2, 
        eval_labels,
        name='name-matching-eval', # Name for the evaluator
        show_progress_bar=True
    )
    print("BinaryClassificationEvaluator created for the evaluation set.")


    # --- 6. Define Training Parameters ---
    print("\nSetting training parameters...")
    num_training_steps_per_epoch = len(train_dataloader_contrastive)
    warmup_steps = math.ceil(num_training_steps_per_epoch * 0.1) # 10% of steps in one epoch

    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps (10% of steps in 1st epoch): {warmup_steps}")
    print(f"  Output path for fine-tuned model: {output_model_path}")

    os.makedirs(output_model_path, exist_ok=True)
    checkpoint_save_path = os.path.join(output_model_path, 'checkpoints')
    os.makedirs(checkpoint_save_path, exist_ok=True)

    # --- 7. Fine-Tune the Model ---
    print("\nStarting model fine-tuning...")
    try:
        model.fit(train_objectives=[(train_dataloader_contrastive, train_loss)],
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  optimizer_params={'lr': learning_rate},
                  output_path=output_model_path,
                  show_progress_bar=True,
                  evaluator=evaluator, # Pass the evaluator
                  evaluation_steps= num_training_steps_per_epoch // 2 if num_training_steps_per_epoch > 50 else num_training_steps_per_epoch, # Evaluate e.g., twice per epoch
                  # output_path_ignore_not_empty=True, # REMOVED THIS LINE
                  checkpoint_path=checkpoint_save_path,
                  checkpoint_save_steps= num_training_steps_per_epoch // 2 if num_training_steps_per_epoch > 100 else num_training_steps_per_epoch,
                  checkpoint_save_total_limit = 2 
                 )
        print("\nModel fine-tuning complete.")
        print(f"Fine-tuned model saved to: {output_model_path}")
    except Exception as e:
        print(f"ERROR: An error occurred during model fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        exit() # Exit if training fails

    # --- 8. Evaluate the Fine-tuned Model on the Test Set ---
    # The model saved in output_model_path is the best one based on evaluator performance (if evaluator is used)
    # Or the model from the last epoch if no evaluator or if save_best_model=False (default is True with evaluator)
    print(f"\n--- Evaluating the fine-tuned model from {output_model_path} on the evaluation set ---")
    
    # Load the best model saved during training (if output_path was used in fit)
    # The 'model' variable should already be the fine-tuned one, but loading explicitly ensures we use what was saved.
    try:
        eval_model = SentenceTransformer(output_model_path)
        print("Loaded fine-tuned model for evaluation.")
    except Exception as e:
        print(f"Error loading the fine-tuned model from {output_model_path}. Using the model in memory. Error: {e}")
        eval_model = model # Fallback to model in memory

    # The BinaryClassificationEvaluator can also be called directly
    # This will compute metrics like accuracy, F1, precision, recall at the best threshold.
    print("\nRunning evaluator on the evaluation set...")
    # The evaluator, when called during fit, already saves its results.
    # Calling it again here will re-evaluate and print/save again.
    # This is fine for explicit evaluation after training completion.
    eval_results_output_path = os.path.join(output_model_path, "final_evaluation_on_test_set") 
    os.makedirs(eval_results_output_path, exist_ok=True) # Ensure this sub-directory exists

    evaluator(eval_model, output_path=eval_results_output_path) 

    print(f"\nEvaluation results (accuracy, F1, etc.) are typically printed by the evaluator.")
    print(f"Detailed evaluation results might be saved in a CSV inside '{eval_results_output_path}' (e.g., 'binary_classification_evaluation_{evaluator.name}_results.csv').")
    
    # You can also manually calculate similarities and metrics if needed:
    # from sentence_transformers import util # Ensure util is imported if using this block
    # print("\nManual calculation of cosine similarities for evaluation set (first 5 pairs):")
    # for i in range(min(5, len(eval_sentences1))):
    #     emb1 = eval_model.encode(eval_sentences1[i])
    #     emb2 = eval_model.encode(eval_sentences2[i])
    #     cos_sim = util.pytorch_cos_sim(emb1, emb2)
    #     print(f"Pair: ('{eval_sentences1[i]}', '{eval_sentences2[i]}'), True Label: {eval_labels[i]}, Cosine Sim: {cos_sim.item():.4f}")


    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()
