import pandas as pd
import numpy as np
import os
import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch_geometric.data import HeteroData
from tqdm import tqdm
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split

# --- 1. Thiết lập Cấu hình và Thiết bị ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 2. Các hàm tiện ích ---
@torch.no_grad()
def create_node_embeddings(texts, model, tokenizer, batch_size=32):
    print("Bắt đầu tạo node features cho 'email' từ DistilBERT...")
    model.to(DEVICE); model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tạo Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
        outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return torch.tensor(np.vstack(all_embeddings), dtype=torch.float)

def extract_entities(text):
    if not isinstance(text, str): return [], []
    unique_domains, unique_emails = set(), set()
    try:
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        for url in urls:
            try:
                domain = urlparse(url).netloc.replace('www.', '')
                if domain and '.' in domain:
                    unique_domains.add(domain)
            except ValueError:
                continue
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            unique_emails.add(email.lower())
    except Exception:
        pass
    return list(unique_domains), list(unique_emails)

def main():
    print("--- Bắt đầu xây dựng Đồ thị Không đồng nhất (Nazario) ---")
    
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, 'combined_dataset_nazario.csv')
    
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['email_id'] = df.index

    print("Trích xuất các thực thể: domains và addresses...")
    entities = df['text'].progress_apply(extract_entities)
    df['domains'] = entities.apply(lambda x: x[0])
    df['addresses'] = entities.apply(lambda x: x[1])
    
    all_domains = sorted(list(set(d for domains in df['domains'] for d in domains)))
    all_addresses = sorted(list(set(a for addresses in df['addresses'] for a in addresses)))
    
    domain_map = {domain: i for i, domain in enumerate(all_domains)}
    address_map = {address: i for i, address in enumerate(all_addresses)}
    
    print(f"Tìm thấy {len(df)} emails, {len(all_domains)} domains, {len(all_addresses)} addresses.")

    bert_model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    email_features = create_node_embeddings(df['text'].tolist(), bert_model, tokenizer)

    print("Tạo các danh sách cạnh...")
    contains_domain_src, contains_domain_dst = [], []
    mentions_address_src, mentions_address_dst = [], []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Tạo cạnh"):
        for domain in row['domains']:
            if domain in domain_map:
                contains_domain_src.append(row['email_id'])
                contains_domain_dst.append(domain_map[domain])
        for address in row['addresses']:
            if address in address_map:
                mentions_address_src.append(row['email_id'])
                mentions_address_dst.append(address_map[address])

    print("Tạo đối tượng HeteroData...")
    data = HeteroData()
    
    data['email'].x = email_features
    data['email'].y = torch.tensor(df['label'].values, dtype=torch.long)
    data['domain'].num_nodes = len(all_domains)
    data['address'].num_nodes = len(all_addresses)

    contains_edge_index = torch.tensor([contains_domain_src, contains_domain_dst], dtype=torch.long)
    mentions_edge_index = torch.tensor([mentions_address_src, mentions_address_dst], dtype=torch.long)
    
    data['email', 'contains', 'domain'].edge_index = contains_edge_index
    data['email', 'mentions', 'address'].edge_index = mentions_edge_index
    
    # === SỬA LỖI ===
    data['domain', 'rev_contains', 'email'].edge_index = contains_edge_index.flip([0])
    data['address', 'rev_mentions', 'email'].edge_index = mentions_edge_index.flip([0])
    
    indices = torch.arange(data['email'].num_nodes)
    labels = data['email'].y
    train_indices, test_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    train_indices, val_indices, _, _ = train_test_split(train_indices, labels[train_indices], test_size=0.125, random_state=42, stratify=labels[train_indices])

    data['email'].train_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, train_indices, True)
    data['email'].val_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, val_indices, True)
    data['email'].test_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, test_indices, True)

    print("\n--- Thông tin Đồ thị Không đồng nhất (Nazario) ---")
    print(data)
    
    graph_path = os.path.join(output_dir, 'hetero_graph_nazario.pt')
    torch.save(data, graph_path)
    print(f"\nĐồ thị không đồng nhất đã được lưu vào: {graph_path}")
    print("--- Hoàn thành! ---")

if __name__ == '__main__':
    tqdm.pandas(desc="Trích xuất Entities")
    main()
