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
from sklearn.utils.class_weight import compute_class_weight

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
    print(f"Bắt đầu xây dựng đồ thị k-NN với k={k}...")
    node_features_norm = F.normalize(node_features, p=2, dim=1)
    adj_matrix = kneighbors_graph(node_features_norm.cpu().numpy(), k, mode='connectivity', include_self=False, metric='cosine')
    edge_index = torch.from_numpy(np.stack(adj_matrix.nonzero())).long()
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
    return results

def main():
    output_dir = 'analysis_outputs'
    
    # --- Tải dữ liệu ---
    data_path = os.path.join(output_dir, 'combined_dataset_nazario.csv')
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Tạo Features và Đồ thị ---
    bert_model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    bert_model = DistilBertModel.from_pretrained(bert_model_name)
    
    node_features = create_node_embeddings(df['text'].tolist(), bert_model, tokenizer)
    edge_index = build_similarity_graph(node_features, k=5)
    
    graph_data = Data(x=node_features, edge_index=edge_index, y=torch.tensor(df['label'].values, dtype=torch.long))
    print(f"Graph created: {graph_data.num_nodes} nodes, {graph_data.num_edges // 2} undirected edges.")

    # --- Chia Train/Val/Test Masks ---
    indices = torch.arange(graph_data.num_nodes)
    labels = graph_data.y
    train_indices, test_indices, _, y_test_labels = train_test_split(indices, labels, test_size=0.2, random_state=42, stratify=labels)
    train_indices, val_indices, y_train_labels, y_val_labels = train_test_split(train_indices, labels[train_indices], test_size=0.125, random_state=42, stratify=labels[train_indices])

    graph_data.train_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, train_indices, True)
    graph_data.val_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, val_indices, True)
    graph_data.test_mask = torch.zeros(graph_data.num_nodes, dtype=torch.bool).index_fill_(0, test_indices, True)
    
    # --- Khởi tạo Model và Optimizer ---
    model = KANGuard(in_channels=graph_data.num_node_features, hidden_channels=128, out_channels=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)

    # === SỬA LỖI: Tính toán Class Weights một cách khoa học ===
    train_labels = graph_data.y[graph_data.train_mask].cpu().numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Computed class weights: {class_weights}")

    # --- Vòng lặp Huấn luyện ---
    best_val_f1 = 0; patience_counter = 0; patience = 50
    best_model_path = os.path.join(output_dir, 'best_kanguard_model_nazario.pth')
    
    print("\nStarting KANGuard training...")
    for epoch in range(1, 201):
        loss = train(model, graph_data, optimizer, class_weights)
        results = test(model, graph_data)
        val_f1 = results['val_mask']
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {results["test_mask"]:.4f}')

        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # === PHẦN ĐÁNH GIÁ CUỐI CÙNG ĐÃ ĐƯỢC SỬA LẠI ===
    print("\n--- Final Evaluation on Best Model ---")
    
    # 1. Tải lại model tốt nhất đã lưu
    if not os.path.exists(best_model_path):
        print("Error: Best model was not saved. Something went wrong during training.")
        return
        
    model.load_state_dict(torch.load(best_model_path))
    
    # 2. Chạy lại đánh giá một cách tường minh
    model.eval()
    with torch.no_grad():
        final_out = model(graph_data.x.to(DEVICE), graph_data.edge_index.to(DEVICE))
        
        # Lấy dự đoán và nhãn thật chỉ cho tập test
        test_pred_logits = final_out[graph_data.test_mask]
        test_pred_labels = test_pred_logits.argmax(dim=-1)
        test_true_labels = graph_data.y[graph_data.test_mask]

    # 3. In thông tin Debug
    print("\n--- DEBUG INFO ---")
    print(f"Number of test predictions: {len(test_pred_labels)}")
    print(f"Number of true test labels: {len(test_true_labels)}")
    print(f"Unique predicted values: {torch.unique(test_pred_labels.cpu())}")
    print(f"True label distribution in test set:\n{pd.Series(test_true_labels.cpu().numpy()).value_counts(normalize=True)}")
    print("--- END DEBUG INFO ---\n")

    # 4. Tạo và in báo cáo
    report_str = classification_report(
        test_true_labels.cpu().numpy(), 
        test_pred_labels.cpu().numpy(), 
        target_names=['Safe (0)', 'Phishing (1)'],
        zero_division=0
    )
    
    print("\n--- KANGuard Final Report ---")
    print(report_str)
    
    report_path = os.path.join(output_dir, 'kanguard_report_nazario.txt')
    with open(report_path, 'w') as f:
        f.write(report_str)
    print(f"KANGuard report saved to '{report_path}'")

if __name__ == '__main__':
    main()
