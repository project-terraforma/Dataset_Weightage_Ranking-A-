import pandas as pd
import json
import os

# --- Configuration ---
CSV_INPUT_PATH = "cleaned.csv"  # Your input CSV file
JSONL_OUTPUT_PATH_XTUNER = "xtuner_qwen_finetuning_data_structured.jsonl" # Original output file for XTuner
JSONL_OUTPUT_PATH_OASST1 = "oasst1_text_formatted_data.jsonl" # New output file for oasst1-style text format
DEFAULT_LOCATION_CONTEXT = "San Francisco" # Default location context if not in CSV
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant." # Default system prompt

# User instruction part of the prompt (remains the same)
USER_INSTRUCTION_PROMPT = """You are a highly accurate assistant specialized in matching place names. Carefully determine if the two names below refer to the exact same place located in {location_context}.

Be stricter when comparing names for restaurants, banks, cafes, schools, or other common businesses and institutions.

Answer with only "Yes" or "No". Do not provide any explanation or any other text.

OSM Name: "{osm_name}"
GERS Name: "{gers_name}"."""

def format_for_qwen_xtuner_structured(csv_file_path, jsonl_output_path, location_context, system_prompt):
    """
    Reads data from a CSV file, formats it into a structured chat format
    (system, user, assistant turns) suitable for XTuner Qwen fine-tuning,
    and writes it to a JSONL file.
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read {len(df)} rows from {csv_file_path} for XTuner format.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    required_columns = ['osm_name', 'gers_name', 'verified_label']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain the following columns: {', '.join(required_columns)}")
        return

    num_skipped = 0
    with open(jsonl_output_path, 'w', encoding='utf-8') as outfile:
        for index, row in df.iterrows():
            osm_name = str(row['osm_name']).strip()
            gers_name = str(row['gers_name']).strip()
            verified_label_raw = row['verified_label']

            label_str = ""
            if isinstance(verified_label_raw, bool):
                label_str = "Yes" if verified_label_raw else "No"
            elif isinstance(verified_label_raw, (int, float)):
                label_str = "Yes" if int(verified_label_raw) == 1 else "No"
            elif isinstance(verified_label_raw, str):
                label_lower = verified_label_raw.strip().lower()
                if label_lower in ['yes', 'true', '1']:
                    label_str = "Yes"
                elif label_lower in ['no', 'false', '0']:
                    label_str = "No"
            
            if not label_str:
                print(f"Warning: Row {index + 1} (XTuner): Unrecognized 'verified_label' format: '{verified_label_raw}'. Skipping this row.")
                num_skipped +=1
                continue

            if not osm_name or not gers_name:
                print(f"Warning: Row {index + 1} (XTuner): Empty OSM name or GERS name after stripping. Skipping this row.")
                num_skipped +=1
                continue

            current_location_context = location_context

            user_content = USER_INSTRUCTION_PROMPT.format(
                location_context=current_location_context,
                osm_name=osm_name,
                gers_name=gers_name
            )

            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": label_str}
            ]
            
            json_record = {"messages": conversation}
            
            outfile.write(json.dumps(json_record) + "\n")

    print(f"\nXTuner Format Processing complete.")
    print(f"Successfully wrote {len(df) - num_skipped} records to {jsonl_output_path}")
    if num_skipped > 0:
        print(f"Skipped {num_skipped} records from XTuner formatting due to issues.")

def format_for_oasst1_text_field(csv_file_path, jsonl_output_path, location_context, system_prompt):
    """
    Reads data from a CSV file, formats each entry into a single "text" field
    concatenating system, user, and assistant messages. This format is common
    for SFT and can be used by map_fns that process such text fields (e.g., oasst1_map_fn variants).
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read {len(df)} rows from {csv_file_path} for OASST1-text format.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    required_columns = ['osm_name', 'gers_name', 'verified_label']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain the following columns: {', '.join(required_columns)}")
        return

    num_skipped = 0
    with open(jsonl_output_path, 'w', encoding='utf-8') as outfile:
        for index, row in df.iterrows():
            osm_name = str(row['osm_name']).strip()
            gers_name = str(row['gers_name']).strip()
            verified_label_raw = row['verified_label']

            label_str = ""
            if isinstance(verified_label_raw, bool):
                label_str = "Yes" if verified_label_raw else "No"
            elif isinstance(verified_label_raw, (int, float)):
                label_str = "Yes" if int(verified_label_raw) == 1 else "No"
            elif isinstance(verified_label_raw, str):
                label_lower = verified_label_raw.strip().lower()
                if label_lower in ['yes', 'true', '1']:
                    label_str = "Yes"
                elif label_lower in ['no', 'false', '0']:
                    label_str = "No"

            if not label_str:
                print(f"Warning: Row {index + 1} (OASST1-text): Unrecognized 'verified_label' format: '{verified_label_raw}'. Skipping this row.")
                num_skipped +=1
                continue

            if not osm_name or not gers_name:
                print(f"Warning: Row {index + 1} (OASST1-text): Empty OSM name or GERS name after stripping. Skipping this row.")
                num_skipped +=1
                continue

            current_location_context = location_context

            # This is the detailed instruction set for the model
            human_turn_content = USER_INSTRUCTION_PROMPT.format(
                location_context=current_location_context,
                osm_name=osm_name,
                gers_name=gers_name
            )
            assistant_turn_content = label_str

            # Construct the single text field
            # Format:
            # {system_prompt}
            # ### Human:
            # {human_turn_content}
            # ### Assistant:
            # {assistant_turn_content}
            text_parts = []
            if system_prompt: # This is the DEFAULT_SYSTEM_PROMPT
                text_parts.append(system_prompt)
            
            text_parts.append(f"### Human:\n{human_turn_content}")
            text_parts.append(f"### Assistant:\n{assistant_turn_content}")

            # Join with double newline for clear separation between system, human, and assistant parts
            # Some tokenizers/map_fns might be sensitive to the exact spacing/newlines.
            # This is a common and generally robust format.
            text_blob = "\n\n".join(text_parts)
            
            json_record_oasst1 = {"text": text_blob}
            
            outfile.write(json.dumps(json_record_oasst1) + "\n")

    print(f"\nOASST1-text Format Processing complete.")
    print(f"Successfully wrote {len(df) - num_skipped} records to {jsonl_output_path}")
    if num_skipped > 0:
        print(f"Skipped {num_skipped} records from OASST1-text formatting due to issues.")


if __name__ == "__main__":
    # Ensure output directories exist
    output_dir_xtuner = os.path.dirname(JSONL_OUTPUT_PATH_XTUNER)
    if output_dir_xtuner and not os.path.exists(output_dir_xtuner):
        os.makedirs(output_dir_xtuner)
        print(f"Created output directory: {output_dir_xtuner}")

    output_dir_oasst1 = os.path.dirname(JSONL_OUTPUT_PATH_OASST1)
    if output_dir_oasst1 and not os.path.exists(output_dir_oasst1):
        os.makedirs(output_dir_oasst1)
        print(f"Created output directory: {output_dir_oasst1}")

    # Generate the original XTuner Qwen structured data
    print("--- Generating XTuner Qwen Formatted Data ---")
    format_for_qwen_xtuner_structured(
        CSV_INPUT_PATH,
        JSONL_OUTPUT_PATH_XTUNER,
        DEFAULT_LOCATION_CONTEXT,
        DEFAULT_SYSTEM_PROMPT
    )
    print("\n")

    # Generate the new oasst1-style text field data
    print("--- Generating OASST1-style Text Formatted Data ---")
    format_for_oasst1_text_field(
        CSV_INPUT_PATH,
        JSONL_OUTPUT_PATH_OASST1,
        DEFAULT_LOCATION_CONTEXT,
        DEFAULT_SYSTEM_PROMPT
    )
    
    print("\nAll processing finished.")