import pandas as pd
import numpy as np
import os
import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from torch_geometric.data import HeteroData
from tqdm import tqdm
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split # DÒNG SỬA LỖI ĐÃ ĐƯỢC THÊM VÀO

# --- 1. Thiết lập Cấu hình và Thiết bị ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 2. Các hàm tiện ích ---
@torch.no_grad()
def create_node_embeddings(texts, model, tokenizer, batch_size=32):
    """Tạo node features cho 'email' từ DistilBERT."""
    print("Bắt đầu tạo node features cho 'email' từ DistilBERT...")
    model.to(DEVICE); model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tạo Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
        outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return torch.tensor(np.vstack(all_embeddings), dtype=torch.float)

def extract_domains(text):
    """Trích xuất các domain duy nhất từ URL trong văn bản."""
    if not isinstance(text, str): return []
    try:
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        domains = [urlparse(url).netloc.replace('www.', '') for url in urls]
        return list(set(d for d in domains if d))
    except:
        return []


def main():
    print("--- Bắt đầu xây dựng Đồ thị Không đồng nhất (Heterogeneous Graph) ---")
    
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, 'combined_dataset_v2.csv')
    
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['email_id'] = df.index

    print("Trích xuất các thực thể: senders và domains...")
    df['from_filled'] = df['from'].fillna('unknown@sender.com').str.lower()
    df['domains'] = df['text'].apply(extract_domains)
    
    all_senders = sorted(df['from_filled'].unique().tolist())
    all_domains = sorted(list(set(d for domains in df['domains'] for d in domains)))
    
    sender_map = {sender: i for i, sender in enumerate(all_senders)}
    domain_map = {domain: i for i, domain in enumerate(all_domains)}
    
    print(f"Tìm thấy {len(df)} emails, {len(all_senders)} senders, {len(all_domains)} domains.")

    bert_model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    email_features = create_node_embeddings(df['text'].tolist(), bert_model, tokenizer)

    print("Tạo các danh sách cạnh...")
    sends_edges_src = [sender_map[s] for s in df['from_filled']]
    sends_edges_dst = df['email_id'].tolist()
    
    contains_edges_src = []
    contains_edges_dst = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Tạo cạnh email-domain"):
        for domain in row['domains']:
            if domain in domain_map:
                contains_edges_src.append(row['email_id'])
                contains_edges_dst.append(domain_map[domain])

    print("Tạo đối tượng HeteroData...")
    data = HeteroData()
    
    data['email'].x = email_features
    data['email'].y = torch.tensor(df['label'].values, dtype=torch.long)
    data['sender'].num_nodes = len(all_senders)
    data['domain'].num_nodes = len(all_domains)

    sends_edge_index = torch.tensor([sends_edges_src, sends_edges_dst], dtype=torch.long)
    contains_edge_index = torch.tensor([contains_edges_src, contains_edges_dst], dtype=torch.long)
    
    data['sender', 'sends', 'email'].edge_index = sends_edge_index
    data['email', 'contains', 'domain'].edge_index = contains_edge_index
    
    data['email', 'rev_sends', 'sender'].edge_index = sends_edge_index.flip([0])
    data['domain', 'rev_contains', 'email'].edge_index = contains_edge_index.flip([0])
    
    indices = torch.arange(data['email'].num_nodes)
    labels = data['email'].y
    train_indices, test_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    train_indices, val_indices, _, _ = train_test_split(train_indices, labels[train_indices], test_size=0.125, random_state=42, stratify=labels[train_indices])

    data['email'].train_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, train_indices, True)
    data['email'].val_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, val_indices, True)
    data['email'].test_mask = torch.zeros(data['email'].num_nodes, dtype=torch.bool).index_fill_(0, test_indices, True)

    print("\n--- Thông tin Đồ thị Không đồng nhất ---")
    print(data)
    
    graph_path = os.path.join(output_dir, 'hetero_graph.pt')
    torch.save(data, graph_path)
    print(f"\nĐồ thị không đồng nhất đã được lưu vào: {graph_path}")
    print("--- Hoàn thành! ---")

if __name__ == '__main__':
    main()
