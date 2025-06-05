import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_user_input(prompt_message, default_value=None):
    """Gets user input with an optional default value."""
    if default_value:
        prompt_message += f" (default: {default_value})"
    user_val = input(prompt_message + ": ").strip()
    return user_val if user_val else default_value

def plot_distribution(df, column_name, title_suffix="", output_dir="analysis_plots"):
    """Plots histogram and KDE for a given column."""
    if df.empty or column_name not in df.columns or df[column_name].empty:
        print(f"No data to plot for {column_name} {title_suffix}.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True, bins=30)
    plt.title(f'Distribution of {column_name} {title_suffix}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    filename = os.path.join(output_dir, f"{column_name.replace(' ', '_')}_distribution{title_suffix.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_cdf(df, column_name, title_suffix="", output_dir="analysis_plots"):
    """Plots the Cumulative Distribution Function (CDF)."""
    if df.empty or column_name not in df.columns or df[column_name].empty:
        print(f"No data to plot CDF for {column_name} {title_suffix}.")
        return

    plt.figure(figsize=(10, 6))
    # Sort data and calculate CDF
    sorted_data = np.sort(df[column_name])
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1 if len(sorted_data) > 1 else 1)
    plt.plot(sorted_data, yvals)
    plt.title(f'CDF of {column_name} {title_suffix}')
    plt.xlabel(column_name)
    plt.ylabel('Cumulative Probability')
    plt.grid(True, linestyle='--', alpha=0.7)
    filename = os.path.join(output_dir, f"{column_name.replace(' ', '_')}_cdf{title_suffix.replace(' ', '_')}.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def plot_boxplots(df_match, df_non_match, column_name, output_dir="analysis_plots"):
    """Plots side-by-side boxplots for matches and non-matches."""
    data_to_plot = []
    labels = []
    if df_match is not None and not df_match.empty and column_name in df_match.columns:
        data_to_plot.append(df_match[column_name])
        labels.append('Matches (Label 1)')
    if df_non_match is not None and not df_non_match.empty and column_name in df_non_match.columns:
        data_to_plot.append(df_non_match[column_name])
        labels.append('Non-Matches (Label 0)')

    if not data_to_plot:
        print(f"No data for boxplot of {column_name}.")
        return

    plt.figure(figsize=(8, 6))
    plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
    plt.title(f'Box Plot of {column_name} for Matches vs Non-Matches')
    plt.ylabel(column_name)
    plt.grid(True, linestyle='--', alpha=0.7)
    filename = os.path.join(output_dir, f"{column_name.replace(' ', '_')}_boxplot_comparison.png")
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def main():
    print("--- Fuzzy Score Deep Dive Analyzer ---")

    # Hardcoded column names as per user request
    name1_col = "osm_name"
    name2_col = "gers_name"
    fuzzy_score_col = "fuzzy_score"
    label_col = "similarity"  # This column now indicates match (1) or non-match (0)
    match_type_col = "match_type" # This column indicates the type of match

    # Get file path from user
    csv_filepath = get_user_input("Enter the path to your matches CSV file (e.g., matches.csv)")
    
    # Values for match/non-match are now fixed (1 and 0)
    match_value = 1
    non_match_value = 0
    
    # Ask if user wants to analyze non-matches too (still useful for comparison)
    analyze_non_matches_input = get_user_input("Do you want to analyze non-matches for comparison? (yes/no)", "yes").lower()


    # Create output directory for plots
    output_dir = "fuzzy_score_analysis_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved in '{output_dir}/' directory.")
    print(f"Using hardcoded columns: name1='{name1_col}', name2='{name2_col}', fuzzy_score='{fuzzy_score_col}', label='{label_col}', match_type='{match_type_col}'")


    try:
        df = pd.read_csv(csv_filepath)
        print(f"\nSuccessfully loaded '{csv_filepath}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at '{csv_filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Validate columns
    required_cols_for_analysis = [name1_col, name2_col, fuzzy_score_col, label_col, match_type_col]
        
    missing_cols = [col for col in required_cols_for_analysis if col not in df.columns]
    if missing_cols:
        print(f"Error: The CSV is missing one or more required columns: {', '.join(missing_cols)}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Convert fuzzy score and label column to numeric, coercing errors
    df[fuzzy_score_col] = pd.to_numeric(df[fuzzy_score_col], errors='coerce')
    df[label_col] = pd.to_numeric(df[label_col], errors='coerce') # 'similarity' is now the label
    
    # Drop rows where essential numeric columns couldn't be converted or are NaN
    df.dropna(subset=[fuzzy_score_col, label_col], inplace=True)
    
    # Ensure label_col contains only 0s and 1s after conversion (or handle other cases)
    valid_labels = [0, 1]
    df = df[df[label_col].isin(valid_labels)]
    df[label_col] = df[label_col].astype(int)


    # Filter for true matches (where label_col is 1)
    df_matches = df[df[label_col] == match_value].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_matches.empty:
        print(f"No rows found where '{label_col}' is '{match_value}'. Cannot perform analysis for matches.")
        # Still proceed to non-match analysis if requested
    else:
        print(f"\n--- Analysis for TRUE MATCHES ('{label_col}' == {match_value}) ---")
        print(f"Number of true matches found: {len(df_matches)}")

        # 1. Descriptive Statistics for Fuzzy Scores of Matches
        print("\n1. Descriptive Statistics of Fuzzy Scores for Matches:")
        desc_stats_matches = df_matches[fuzzy_score_col].describe()
        print(desc_stats_matches)

        # 2. Lowest Fuzzy Scores that are Matches
        min_fuzzy_for_match = df_matches[fuzzy_score_col].min()
        print(f"\n2. Absolute Lowest Fuzzy Score for a Match: {min_fuzzy_for_match:.2f}")
        
        print("\n   Examples of Matches with Low Fuzzy Scores (up to 10 lowest, or all if fewer):")
        low_score_matches = df_matches.nsmallest(min(10, len(df_matches)), fuzzy_score_col)
        for _, row in low_score_matches.iterrows():
            print(f"   - Fuzzy: {row[fuzzy_score_col]:.2f}, {name1_col}: '{row[name1_col]}', {name2_col}: '{row[name2_col]}', Type: {row.get(match_type_col, 'N/A')}")

        # 3. Distribution Plots for Fuzzy Scores of Matches
        print("\n3. Generating Distribution Plots for Fuzzy Scores of Matches...")
        plot_distribution(df_matches, fuzzy_score_col, title_suffix=" (Matches)", output_dir=output_dir)
        plot_cdf(df_matches, fuzzy_score_col, title_suffix=" (Matches)", output_dir=output_dir)

        # 4. Percentage of Matches below Fuzzy Score Thresholds
        print("\n4. Percentage of Matches Below Certain Fuzzy Score Thresholds:")
        thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
        for thresh in thresholds:
            count_below = len(df_matches[df_matches[fuzzy_score_col] < thresh])
            percentage_below = (count_below / len(df_matches)) * 100 if len(df_matches) > 0 else 0
            print(f"   - Matches with Fuzzy Score < {thresh}: {count_below} ({percentage_below:.2f}%)")

    # --- Optional: Analysis for NON-MATCHES ---
    df_non_matches = None
    if analyze_non_matches_input == 'yes':
        df_non_matches = df[df[label_col] == non_match_value].copy() # Use .copy()

        if not df_non_matches.empty:
            print(f"\n--- Analysis for TRUE NON-MATCHES ('{label_col}' == {non_match_value}) ---")
            print(f"Number of true non-matches found: {len(df_non_matches)}")
            print("\n   Descriptive Statistics of Fuzzy Scores for Non-Matches:")
            print(df_non_matches[fuzzy_score_col].describe())
            plot_distribution(df_non_matches, fuzzy_score_col, title_suffix=" (Non-Matches)", output_dir=output_dir)
        else:
            print(f"\nNo rows found where '{label_col}' is '{non_match_value}'. Cannot perform analysis for non-matches.")
            df_non_matches = None # Ensure it's None if empty

    # 5. Comparative Box Plot
    if df_matches is not None and not df_matches.empty: # Check df_matches for plotting
        if df_non_matches is not None and not df_non_matches.empty:
            print("\n5. Generating Comparative Box Plot for Fuzzy Scores...")
            plot_boxplots(df_matches, df_non_matches, fuzzy_score_col, output_dir=output_dir)
        else:
            print("\n   Skipping comparative box plot as non-match data is not available or not analyzed.")
            # Plot boxplot for matches only
            plt.figure(figsize=(6,6))
            plt.boxplot(df_matches[fuzzy_score_col], labels=['Matches (Label 1)'], patch_artist=True)
            plt.title(f'Box Plot of {fuzzy_score_col} (Matches)')
            plt.ylabel(fuzzy_score_col)
            plt.grid(True, linestyle='--', alpha=0.7)
            filename = os.path.join(output_dir, f"{fuzzy_score_col.replace(' ', '_')}_boxplot_matches_only.png")
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
            plt.close()
    else:
        print("\nSkipping box plots as no match data is available.")


    # 6. Analysis with Similarity Score (This section is removed as 'similarity' is now the label)
    # print(f"\n--- Section 6 (Fuzzy vs Semantic Similarity) is skipped as '{label_col}' is now the match label ---")

            
    # 7. Grouped Analysis by 'match_type' (using the hardcoded match_type_col)
    if match_type_col in df.columns and df_matches is not None and not df_matches.empty:
        print(f"\n--- Fuzzy Score Analysis by '{match_type_col}' Category for Matches ---")
        # Ensure match_type_col is string for grouping, handle NaNs
        df_matches[match_type_col] = df_matches[match_type_col].astype(str).fillna('Unknown')
        
        if df_matches[match_type_col].nunique() > 0:
            grouped_stats = df_matches.groupby(match_type_col)[fuzzy_score_col].agg(['min', 'mean', 'median', 'max', 'count']).sort_values(by='mean', ascending=False)
            print(grouped_stats)

            if df_matches[match_type_col].nunique() <= 10: # Avoid too many categories in boxplot
                plt.figure(figsize=(12, 7))
                sns.boxplot(x=match_type_col, y=fuzzy_score_col, data=df_matches)
                plt.title(f'{fuzzy_score_col} by {match_type_col} (Matches)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                filename = os.path.join(output_dir, f"{fuzzy_score_col}_by_{match_type_col}_boxplot.png")
                plt.savefig(filename)
                print(f"Saved plot: {filename}")
                plt.close()
            else:
                print(f"Skipping boxplot for '{match_type_col}' categories due to high number of unique values (>10).")
        else:
            print(f"No unique values found in '{match_type_col}' for matches to perform grouped analysis.")
    elif df_matches is not None and not df_matches.empty:
         print(f"Column '{match_type_col}' not found in DataFrame. Skipping grouped analysis by match type.")


    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
