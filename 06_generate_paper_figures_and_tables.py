import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.metrics import confusion_matrix

def generate_visualizations():
    """
    Generate all figures and tables for the research paper.
    """
    print("--- Generating Figures and Tables for the Paper ---")
    
    output_dir = 'analysis_outputs'
    
    # --- Load Data and Reports ---
    try:
        df = pd.read_csv(os.path.join(output_dir, 'combined_dataset_nazario.csv'))
        
        # Function to parse classification reports
        def parse_report(path):
            with open(path, 'r') as f:
                text = f.read()
            
            # Extract main metrics for Phishing class and accuracy
            phishing_line = re.search(r'Phishing \(1\)(.*)', text)
            accuracy_line = re.search(r'accuracy(.*)', text)
            
            if phishing_line and accuracy_line:
                p_metrics = phishing_line.group(1).split()
                a_metrics = accuracy_line.group(1).split()
                
                return {
                    'precision': float(p_metrics[0]),
                    'recall': float(p_metrics[1]),
                    'f1-score': float(p_metrics[2]),
                    'accuracy': float(a_metrics[0])
                }
            return None

        report_nb = parse_report(os.path.join(output_dir, 'naive_bayes_report_nazario.txt'))
        report_distilbert = parse_report(os.path.join(output_dir, 'distilbert_report_nazario.txt'))
        report_kanguard = parse_report(os.path.join(output_dir, 'kanguard_report_nazario.txt'))

        reports = {
            'Naive Bayes': report_nb,
            'DistilBERT': report_distilbert,
            'KANGuard-Sim': report_kanguard
        }

    except Exception as e:
        print(f"Error loading data or reports: {e}")
        print("Please ensure all previous scripts have been run successfully.")
        return

    # --- TABLE 1: Dataset Statistics ---
    print("\n--- Generating Table 1: Dataset Statistics ---")
    num_total = len(df)
    num_phishing = df['label'].sum()
    num_safe = num_total - num_phishing
    
    # Assuming 70% train, 10% val, 20% test split from scripts
    num_train = int(num_total * 0.7)
    num_val = int(num_total * 0.1)
    num_test = num_total - num_train - num_val

    table1_data = {
        'Metric': ['Total Emails', 'Phishing Emails', 'Safe Emails', 'Training Set Size', 'Validation Set Size', 'Test Set Size'],
        'Value': [num_total, num_phishing, num_safe, num_train, num_val, num_test]
    }
    df_table1 = pd.DataFrame(table1_data)
    print("Table 1:\n", df_table1.to_string(index=False))
    df_table1.to_csv(os.path.join(output_dir, 'table1_dataset_stats.csv'), index=False)


    # --- TABLE 2: Comprehensive Comparison Results ---
    print("\n--- Generating Table 2: Comprehensive Comparison Results ---")
    table2_data = []
    for name, report in reports.items():
        if report:
            table2_data.append({
                'Model': name,
                'Accuracy': f"{report['accuracy']:.4f}",
                'Phishing Precision': f"{report['precision']:.4f}",
                'Phishing Recall': f"{report['recall']:.4f}",
                'Phishing F1-Score': f"{report['f1-score']:.4f}"
            })
    df_table2 = pd.DataFrame(table2_data)
    print("Table 2:\n", df_table2.to_string(index=False))
    df_table2.to_csv(os.path.join(output_dir, 'table2_comparison_results.csv'), index=False)


    # --- FIGURE 2: Data Distribution ---
    print("\n--- Generating Figure 2: Data Distribution ---")
    df['text_length'] = df['text'].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Nazario Dataset Distribution Analysis', fontsize=16)

    # Subplot (a): Label Distribution
    sns.countplot(ax=axes[0], x='label', data=df, palette=['skyblue', 'salmon'])
    axes[0].set_title('(a) Label Distribution')
    axes[0].set_xlabel('Email Type')
    axes[0].set_ylabel('Count')
    axes[0].set_xticklabels(['Safe', 'Phishing'])

    # Subplot (b): Text Length Distribution
    sns.kdeplot(ax=axes[1], data=df, x='text_length', hue='label', 
                fill=True, common_norm=False, palette=['skyblue', 'salmon'])
    axes[1].set_title('(b) Email Content Length Distribution')
    axes[1].set_xlabel('Length (number of characters)')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim(0, 20000)
    axes[1].legend(title='Label', labels=['Phishing (1)', 'Safe (0)'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, 'figure2_data_distribution.png')
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(f"Figure 2 saved to {fig_path}")


    # --- FIGURE 3: Confusion Matrices ---
    print("\n--- Generating Figure 3: Confusion Matrices ---")
    # We need to re-generate predictions to create CMs
    # This part is simplified; for a real paper, you'd save predictions from each script
    # For now, we'll create placeholder CMs based on the report stats (this is an approximation)
    # A more robust way is to load saved test_predictions from each script if you saved them.
    # We will load the saved .png files instead for simplicity
    print("Note: Figure 3 should be manually assembled from the individual confusion matrix .png files generated by scripts 02 and 03.")


    # --- FIGURE 4: Performance Comparison ---
    print("\n--- Generating Figure 4: Performance Comparison ---")
    df_plot = df_table2.melt(id_vars='Model', value_vars=['Phishing F1-Score', 'Phishing Precision', 'Phishing Recall'],
                             var_name='Metric', value_name='Score')
    df_plot['Score'] = df_plot['Score'].astype(float)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Model Performance Comparison on Phishing Class', fontsize=16)

    # Subplot (a): F1-Score Comparison
    sns.barplot(ax=axes[0], data=df_plot[df_plot['Metric'] == 'Phishing F1-Score'], x='Model', y='Score', palette='viridis')
    axes[0].set_title('(a) F1-Score Comparison for Phishing Detection')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_ylim(0.9, 1.0) # Zoom in for better visualization
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Subplot (b): Precision vs. Recall
    sns.scatterplot(ax=axes[1], data=df_table2, x='Phishing Precision', y='Phishing Recall', hue='Model', s=200, style='Model', palette='deep')
    axes[1].set_title('(b) Precision-Recall Trade-off')
    axes[1].set_xlabel('Precision')
    axes[1].set_ylabel('Recall')
    axes[1].set_xlim(0.9, 1.0)
    axes[1].set_ylim(0.88, 1.0)
    axes[1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, 'figure4_performance_comparison.png')
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(f"Figure 4 saved to {fig_path}")

    print("\n--- All tasks completed! Check the 'analysis_outputs' directory. ---")
    print("Reminder: Figure 1 (Architecture) and Figure 5 (Graph Visualization) need to be created manually.")


if __name__ == '__main__':
    generate_visualizations()
