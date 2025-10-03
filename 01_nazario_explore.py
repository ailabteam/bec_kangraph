import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_nazario_dataset():
    """
    Tải, khám phá, và chuẩn bị bộ dữ liệu Nazario Phishing.
    """
    print("--- Starting Data Analysis Pipeline (Nazario Dataset) ---")

    # --- 1. Setup Paths ---
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = 'datasets/nazario/Phishing_Email.csv'

    # --- 2. Load Data ---
    try:
        df = pd.read_csv(data_path)
        print("Nazario dataset loaded successfully!")
        print(f"Total number of emails: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 3. Explore and Preprocess ---
    print("\n--- Dataset Info ---")
    df.info()
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    # Kiểm tra các giá trị thiếu
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    # Xử lý các giá trị thiếu. Cột 'Email Text' là quan trọng nhất.
    df.dropna(subset=['Email Text'], inplace=True)
    print(f"\nShape after dropping missing 'Email Text': {df.shape}")

    # Chuẩn hóa tên cột và nhãn
    # 'Email Type' có 2 giá trị: 'Phishing Email' và 'Safe Email'
    df = df.rename(columns={
        'Email Text': 'text',
        'Email Type': 'type'
    })
    
    # Chuyển nhãn dạng chuỗi thành số (1 for Phishing, 0 for Safe)
    df['label'] = df['type'].apply(lambda x: 1 if x == 'Phishing Email' else 0)
    
    # Chọn các cột cần thiết. Bộ dữ liệu này không có thông tin sender/receiver.
    # Chúng ta sẽ tập trung vào đồ thị tương đồng nội dung trước.
    df_final = df[['text', 'label']].copy()
    
    # Lưu bộ dữ liệu đã được xử lý
    processed_path = os.path.join(output_dir, 'combined_dataset_nazario.csv')
    df_final.to_csv(processed_path, index=False)
    print(f"\nProcessed Nazario dataset saved to: {processed_path}")

    # --- 4. Analyze the Processed Dataset ---
    print("\nPerforming analysis on the processed dataset...")
    df_final['text_length'] = df_final['text'].str.len()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_final, x='text_length', hue='label', kde=True, palette=['skyblue', 'salmon'])
    plt.title('Email Content Length Distribution (Phishing vs. Safe) - Nazario')
    plt.xlabel('Length (number of characters)')
    plt.ylabel('Number of Emails')
    plt.xlim(0, 20000)
    plt.legend(title='Label', labels=['Phishing (1)', 'Safe (0)'])
    fig_path = os.path.join(output_dir, 'text_length_distribution_nazario.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Text length distribution chart (Nazario) saved.")

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df_final, x='label', palette=['skyblue', 'salmon'])
    plt.title('Label Distribution (0: Safe, 1: Phishing) - Nazario')
    plt.xlabel('Label')
    plt.ylabel('Count')
    ax.set_xticklabels(['Safe', 'Phishing'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    fig_path = os.path.join(output_dir, 'label_distribution_nazario.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Label distribution chart (Nazario) saved.")
    
    print("\n--- Data analysis pipeline (Nazario) completed successfully! ---")

if __name__ == '__main__':
    explore_nazario_dataset()
