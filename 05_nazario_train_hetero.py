import pandas as pd
import numpy as np
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from kan import KAN

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {DEVICE}")

# --- 1. Định nghĩa Kiến trúc GNN và KANGuard ---

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x

class HeteroKANGuard(torch.nn.Module):
    def __init__(self, data_structure, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Lưu lại num_nodes cho các loại nút không có feature sẵn
        self.num_nodes_dict = {
            node_type: data_structure[node_type].num_nodes 
            for node_type in data_structure.node_types if node_type != 'email'
        }

        # Tạo các lớp embedding/linear ban đầu
        self.email_lin = nn.Linear(data_structure['email'].num_features, hidden_channels)
        self.emb_dict = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes, hidden_channels)
            for node_type, num_nodes in self.num_nodes_dict.items()
        })

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data_structure.metadata(), aggr='sum')
        self.kan = KAN([hidden_channels, 64, out_channels])

    def forward(self, x_dict, edge_index_dict):
        x_map = {'email': self.email_lin(x_dict['email'])}
        for node_type, num_nodes in self.num_nodes_dict.items():
            node_ids = torch.arange(num_nodes, device=x_dict['email'].device)
            x_map[node_type] = self.emb_dict[node_type](node_ids)
        
        x_map = self.gnn(x_map, edge_index_dict)
        email_embedding = x_map['email']
        out = self.kan(email_embedding)
        return out

# --- 2. Vòng lặp Huấn luyện và Đánh giá ---
def train(model, data, optimizer, class_weights):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = F.cross_entropy(out[data['email'].train_mask], data['email'].y[data['email'].train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    pred = out.argmax(dim=-1)
    results = {}
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['email'][mask_name]
        f1 = f1_score(data['email'].y[mask].cpu(), pred[mask].cpu(), pos_label=1, average='binary', zero_division=0)
        results[mask_name] = f1
    return results, pred[data['email'].test_mask]

# --- 3. Hàm Chính ---
def main():
    print("--- Bắt đầu huấn luyện HeteroKANGuard (Nazario Dataset) ---")
    
    output_dir = 'analysis_outputs'
    graph_path = os.path.join(output_dir, 'hetero_graph_nazario.pt')
    if not os.path.exists(graph_path):
        print(f"Lỗi: Không tìm thấy file đồ thị. Vui lòng chạy 04_nazario_build_hetero.py trước.")
        return
    data = torch.load(graph_path)

    model = HeteroKANGuard(data, hidden_channels=128, out_channels=2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
    
    train_labels = data['email'].y[data['email'].train_mask].cpu().numpy()
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Computed class weights: {class_weights}")

    best_val_f1 = 0; patience_counter = 0; patience = 50
    best_model_path = os.path.join(output_dir, 'best_hetero_kanguard_model_nazario.pth')
    
    print("\nBắt đầu huấn luyện...")
    data = data.to(DEVICE)
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, class_weights)
        results, _ = test(model, data)
        val_f1 = results['val_mask']
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"*** New best Val F1: {best_val_f1:.4f} at epoch {epoch}. Model saved. ***")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {results["test_mask"]:.4f}')

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print("\n--- Final Evaluation ---")
    model.load_state_dict(torch.load(best_model_path))
    _, test_pred = test(model, data)
    
    report_str = classification_report(data['email'].y[data['email'].test_mask].cpu(), test_pred.cpu(), target_names=['Safe (0)', 'Phishing (1)'], zero_division=0)
    
    print("\n--- HeteroKANGuard Final Report (Nazario) ---")
    print(report_str)
    
    report_path = os.path.join(output_dir, 'hetero_kanguard_report_nazario.txt')
    with open(report_path, 'w') as f: f.write(report_str)
    print(f"Report saved to '{report_path}'")
    print("--- Completed! ---")

if __name__ == '__main__':
    main()
