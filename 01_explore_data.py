import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_bec_dataset_v3():
    """
    Version 3: Uses the new, enriched HAM dataset.
    All outputs and charts are in English.
    """
    print("--- Starting Data Analysis Pipeline (v3 - Enriched Enron) ---")

    # --- 1. Setup Paths ---
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    BASE_BEC_REPO_PATH = '../bec' # Still needed for BEC data
    
    bec_path_1 = os.path.join(BASE_BEC_REPO_PATH, 'data', 'BEC-1-human.csv')
    bec_path_2 = os.path.join(BASE_BEC_REPO_PATH, 'data', 'BEC-2-human.csv')
    
    # Path to the new HAM dataset
    ham_path = os.path.join(output_dir, 'enron_20k_full_headers.csv') 

    # --- 2. Load Data ---
    try:
        df_bec1 = pd.read_csv(bec_path_1)
        df_bec2 = pd.read_csv(bec_path_2)
        df_bec = pd.concat([df_bec1, df_bec2], ignore_index=True)
        
        df_ham = pd.read_csv(ham_path)
        
        print("Datasets loaded successfully!")
        print(f"Number of BEC emails: {len(df_bec)}")
        print(f"Number of HAM emails: {len(df_ham)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'BEC-*-human.csv' and 'enron_20k_full_headers.csv' files exist.")
        return

    # --- 3. Preprocess and Merge Data ---
    print("Standardizing and merging datasets...")
    
    # Standardize BEC dataframe
    if 'text' not in df_bec.columns and 'body' in df_bec.columns:
        df_bec = df_bec.rename(columns={'body': 'text', 'subject': 'Subject'})
    # Add missing columns for consistent merging
    df_bec['From'] = 'generated@bec.com' # Assign a placeholder sender
    df_bec['To'] = 'victim@company.com'   # Assign a placeholder recipient
    df_bec['label'] = 1  # BEC

    # Standardize HAM dataframe
    df_ham['label'] = 0  # HAM

    # Merge
    common_cols = ['From', 'To', 'Subject', 'Message', 'label']
    df_bec_final = df_bec.rename(columns={'text': 'Message'})[common_cols]
    df_ham_final = df_ham[common_cols]

    df_combined = pd.concat([df_bec_final, df_ham_final], ignore_index=True)
    
    # Rename columns for compatibility with subsequent scripts
    df_combined = df_combined.rename(columns={'Message': 'text', 'From': 'from', 'To': 'to', 'Subject': 'subject'})
    
    # Shuffle the dataset
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the processed data
    processed_data_path = os.path.join(output_dir, 'combined_dataset_v3.csv')
    df_combined.to_csv(processed_data_path, index=False)
    print(f"Merged dataset (v3) has been saved to: '{processed_data_path}'")
    
    # --- 4. Analyze the Merged Dataset ---
    print("Performing analysis on the merged dataset...")
    df_combined['text_length'] = df_combined['text'].str.len()
    
    # Plot text length distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_combined, x='text_length', hue='label', kde=True, palette=['skyblue', 'salmon'])
    plt.title('Email Content Length Distribution (BEC vs. HAM) - v3')
    plt.xlabel('Length (number of characters)')
    plt.ylabel('Number of Emails')
    plt.xlim(0, 10000) # Limit x-axis for better visibility
    plt.legend(title='Label', labels=['BEC (1)', 'HAM (0)'])
    fig_path = os.path.join(output_dir, 'text_length_distribution_v3.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Text length distribution chart (v3) saved to: '{fig_path}'")

    # Plot label distribution
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df_combined, x='label', palette=['skyblue', 'salmon'])
    plt.title('Label Distribution (0: HAM, 1: BEC) - v3')
    plt.xlabel('Label')
    plt.ylabel('Count')
    ax.set_xticklabels(['HAM', 'BEC'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    fig_path = os.path.join(output_dir, 'label_distribution_v3.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Label distribution chart (v3) saved to: '{fig_path}'")
    
    print("\n--- Data analysis pipeline (v3) completed successfully! ---")

if __name__ == '__main__':
    explore_bec_dataset_v3()
