import pandas as pd
import numpy as np
import os
import re
import warnings

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# PyTorch and Hugging Face imports
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Matplotlib and Seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_naive_bayes(df, output_dir, suffix=""):
    print("\n" + "="*50)
    print(f"--- Starting: Naive Bayes (Dataset{suffix}) ---")
    print("="*50)
    
    df['clean_text'] = df['text'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000) # Tăng max_features cho bộ dữ liệu lớn hơn
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)'], zero_division=0)
    
    print(f"\n--- Results: Naive Bayes (Dataset{suffix}) ---")
    print(report)
    
    with open(os.path.join(output_dir, f'naive_bayes_report{suffix}.txt'), 'w') as f: f.write(report)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Phishing'], yticklabels=['Safe', 'Phishing'])
    plt.title(f'Confusion Matrix - Naive Bayes (Dataset{suffix})')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'naive_bayes_cm{suffix}.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Naive Bayes results (Dataset{suffix}) saved.")

def train_distilbert(df, output_dir, suffix=""):
    print("\n" + "="*50)
    print(f"--- Starting: DistilBERT (Dataset{suffix}) ---")
    print("="*50)
    
    df['text'] = df['text'].astype(str)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])
    
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
        
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Dữ liệu khá cân bằng, không cần class weights
    
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f'distilbert_results{suffix}'),
        num_train_epochs=3, # 3 epochs là đủ cho bộ dữ liệu cỡ này
        per_device_train_batch_size=16, # Tăng batch size
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predictions, pos_label=1, average='binary')
        return {"f1": f1}

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, tokenizer=tokenizer, compute_metrics=compute_metrics)
    
    trainer.train()
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_test = test_dataset['label']
    
    report_str = classification_report(y_test, y_pred, target_names=['Safe (0)', 'Phishing (1)'], zero_division=0)
    print(f"\n--- Results: DistilBERT (Dataset{suffix}) ---")
    print(report_str)

    with open(os.path.join(output_dir, f'distilbert_report{suffix}.txt'), 'w') as f: f.write(report_str)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Safe', 'Phishing'], yticklabels=['Safe', 'Phishing'])
    plt.title(f'Confusion Matrix - DistilBERT (Dataset{suffix})')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'distilbert_cm{suffix}.png'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"DistilBERT results (Dataset{suffix}) saved.")

def main():
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, 'combined_dataset_nazario.csv') 
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run 01_nazario_explore.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    train_naive_bayes(df.copy(), output_dir, suffix="_nazario")
    train_distilbert(df.copy(), output_dir, suffix="_nazario")

if __name__ == '__main__':
    main()
