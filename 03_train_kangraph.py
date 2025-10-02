import pandas as pd
import numpy as np
import os
import warnings

# PyTorch and PyG imports
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import to_undirected
from sklearn.neighbors import kneighbors_graph

# Transformers for embeddings
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# Scikit-learn for metrics and splitting
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# Import our KAN model
from kan import KAN

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

@torch.no_grad()
def create_node_embeddings(texts, model, tokenizer, batch_size=32):
    print("Bắt đầu tạo node features từ DistilBERT...")
    model.to(DEVICE); model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tạo Embeddings"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
        outputs = model(**inputs)
        all_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
    return torch.tensor(np.vstack(all_embeddings), dtype=torch.float)

def build_similarity_graph(node_features, k=5):
    """Xây dựng đồ thị k-NN dựa trên sự tương đồng cosine."""
    print(f"Bắt đầu xây dựng đồ thị k-NN với k={k}...")
    
    # Chuẩn hóa features để tính cosine similarity bằng dot product
    node_features_norm = F.normalize(node_features, p=2, dim=1)
    
    # Sử dụng scikit-learn để xây dựng đồ thị k-NN hiệu quả
    adj_matrix = kneighbors_graph(node_features_norm.cpu().numpy(), k, mode='connectivity', include_self=False, metric='cosine')
    edge_index = torch.from_numpy(np.stack(adj_matrix.nonzero())).long()
    
    # Đảm bảo đồ thị là vô hướng
    edge_index = to_undirected(edge_index)
    
    return edge_index

class KANGuard(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.kan = KAN([hidden_channels, 64, out_channels])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.kan(x)
        return x

def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(DEVICE), weight=class_weights)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    full_out = model(data.x.to(DEVICE), data.edge_index.to(DEVICE))
    pred = full_out.argmax(dim=-1)
    results = {}
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[mask_name]
        f1 = f1_score(data.y[mask].cpu(), pred[mask].cpu(), pos_label=1, average='binary', zero_division=0)
        results[mask_name] = f1
    return results, pred[data.test_mask]

def main():
    data_path = 'analysis_outputs/combined_dataset_v2.csv'
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['node_id'] = df.index

    bert_model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    
    node_features = create_node_embeddings(df['text'].tolist(), bert_model, tokenizer)
    edge_index = build_similarity_graph(node_features, k=5)
    
    graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(df['label'].values, dtype=torch.long))
    print(f"Đồ thị đã được tạo: {graph_data.num_nodes} nút, {graph_data.num_edges // 2} cạnh vô hướng.")

    indices = torch.arange(graph_data.num_nodes)
    labels = graph_data.y
    train_indices, test_indices, _, _ = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    train_indices, val_indices, _, _ = train_test_split(train_indices, labels[train_indices], test_size=0.125, random_state=42, stratify=labels[train_indices])

    graph_data.train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, train_indices, True)
    graph_data.val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, val_indices, True)
    graph_data.test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, test_indices, True)
    
    model = KANGuard(in_channels=graph_data.num_node_features, hidden_channels=128, out_channels=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)

    class_weights = torch.tensor([1.0, 10.0], device=DEVICE)

    best_val_f1 = 0; patience_counter = 0; patience = 50
    print("\nBắt đầu huấn luyện KANGuard...")
    for epoch in range(1, 201):
        loss = train(model, graph_data, optimizer, class_weights)
        results, _ = test(model, graph_data)
        val_f1 = results['val_mask']
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_kanguard_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {results["test_mask"]:.4f}')

        if patience_counter >= patience:
            print("Early stopping!"); break

    print("\nĐánh giá cuối cùng với model tốt nhất...")
    model.load_state_dict(torch.load('best_kanguard_model.pth'))
    _, test_pred = test(model, graph_data)
    
    report_str = classification_report(graph_data.y[graph_data.test_mask].cpu(), test_pred.cpu(), target_names=['HAM (0)', 'BEC (1)'], zero_division=0)
    
    print("\n--- Báo cáo chi tiết KANGuard ---")
    print(report_str)
    
    output_dir = 'analysis_outputs'
    with open(os.path.join(output_dir, 'kanguard_report.txt'), 'w') as f: f.write(report_str)
    print(f"Báo cáo kết quả KANGuard đã được lưu vào '{output_dir}/kanguard_report.txt'")

if __name__ == '__main__':
    main()
