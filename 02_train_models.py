import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

def clean_text(text):
    """Một hàm làm sạch văn bản đơn giản."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'http\S+', '', text)  # Xóa URL
    text = re.sub(r'\S+@\S+', '', text)  # Xóa email
    text = re.sub(r'[^a-z\s]', '', text)  # Chỉ giữ lại chữ cái và khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()  # Xóa các khoảng trắng thừa
    return text

def train_naive_bayes(df):
    """Huấn luyện và đánh giá mô hình Naive Bayes."""
    print("\n--- Bắt đầu huấn luyện mô hình Naive Bayes ---")
    
    # 1. Tiền xử lý
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 2. Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # 3. Vector hóa
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 4. Huấn luyện
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    # 5. Đánh giá
    y_pred = model.predict(X_test_tfidf)
    
    print("Kết quả Naive Bayes:")
    report = classification_report(y_test, y_pred, target_names=['HAM', 'BEC'])
    print(report)
    
    # Lưu kết quả
    output_dir = 'analysis_outputs'
    with open(os.path.join(output_dir, 'naive_bayes_report.txt'), 'w') as f:
        f.write("--- Classification Report for Naive Bayes ---\n")
        f.write(report)
    print(f"Báo cáo kết quả Naive Bayes đã được lưu vào '{output_dir}/naive_bayes_report.txt'")
    
    # Vẽ và lưu ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HAM', 'BEC'], yticklabels=['HAM', 'BEC'])
    plt.title('Ma trận nhầm lẫn - Naive Bayes')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.savefig(os.path.join(output_dir, 'naive_bayes_cm.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Ma trận nhầm lẫn Naive Bayes đã được lưu vào '{output_dir}/naive_bayes_cm.png'")


def train_distilbert(df):
    """Huấn luyện và đánh giá mô hình DistilBERT."""
    print("\n--- Bắt đầu huấn luyện mô hình DistilBERT ---")
    
    # 1. Chuẩn bị dữ liệu
    # Lấy một tập con nhỏ hơn để huấn luyện nhanh hơn cho lần thử đầu
    # Bạn có thể bỏ dòng này để huấn luyện trên toàn bộ dữ liệu
    df_sample = df.sample(n=1000, random_state=42) if len(df) > 1000 else df
    
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42, stratify=df_sample['label'])
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # 2. Tokenization
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # 3. Thiết lập mô hình và training arguments
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    output_dir = 'analysis_outputs'
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'distilbert_results'),
        num_train_epochs=3,
        per_device_train_batch_size=8, # Giảm nếu gặp lỗi CUDA out of memory
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, 'distilbert_logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # 4. Huấn luyện
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    
    # 5. Đánh giá
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_test = test_dataset['label']
    
    print("Kết quả DistilBERT:")
    report = classification_report(y_test, y_pred, target_names=['HAM', 'BEC'])
    print(report)
    
    with open(os.path.join(output_dir, 'distilbert_report.txt'), 'w') as f:
        f.write("--- Classification Report for DistilBERT ---\n")
        f.write(report)
    print(f"Báo cáo kết quả DistilBERT đã được lưu vào '{output_dir}/distilbert_report.txt'")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HAM', 'BEC'], yticklabels=['HAM', 'BEC'])
    plt.title('Ma trận nhầm lẫn - DistilBERT')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.savefig(os.path.join(output_dir, 'distilbert_cm.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Ma trận nhầm lẫn DistilBERT đã được lưu vào '{output_dir}/distilbert_cm.png'")


def main():
    """Hàm chính để chạy các mô hình."""
    data_path = 'analysis_outputs/combined_dataset_v2.csv'
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}. Vui lòng chạy script 01_explore_data.py trước.")
        return
        
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)

    # Chạy các mô hình
    train_naive_bayes(df.copy())
    train_distilbert(df.copy())

if __name__ == '__main__':
    main()
