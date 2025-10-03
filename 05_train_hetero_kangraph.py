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

        # SỬA LỖI: Lưu num_nodes vào thuộc tính của model
        self.sender_num_nodes = data_structure['sender'].num_nodes
        self.domain_num_nodes = data_structure['domain'].num_nodes

        self.email_lin = nn.Linear(data_structure['email'].num_features, hidden_channels)
        self.sender_emb = nn.Embedding(self.sender_num_nodes, hidden_channels)
        self.domain_emb = nn.Embedding(self.domain_num_nodes, hidden_channels)

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, data_structure.metadata(), aggr='sum')

        self.kan = KAN([hidden_channels, 64, out_channels])

    def forward(self, x_dict, edge_index_dict):
        # SỬA LỖI: Sử dụng các thuộc tính đã lưu để tạo tensor node IDs
        x_map = {
          'email': self.email_lin(x_dict['email']),
          'sender': self.sender_emb(torch.arange(self.sender_num_nodes, device=x_dict['email'].device)),
          'domain': self.domain_emb(torch.arange(self.domain_num_nodes, device=x_dict['email'].device)),
        }
        
        x_map = self.gnn(x_map, edge_index_dict)
        
        email_embedding = x_map['email']
        out = self.kan(email_embedding)
        
        return out


# --- 2. Vòng lặp Huấn luyện và Đánh giá (Giữ nguyên) ---
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

# --- 3. Hàm Chính (Giữ nguyên) ---
def main():
    print("--- Bắt đầu huấn luyện KANGuard trên Đồ thị Không đồng nhất (v4) ---")
    
    output_dir = 'analysis_outputs'
    graph_path = os.path.join(output_dir, 'hetero_graph_v2.pt')
    if not os.path.exists(graph_path):
        print(f"Lỗi: Không tìm thấy file đồ thị. Vui lòng chạy 04_build_hetero_graph.py trước.")
        return
    data = torch.load(graph_path)

    model = HeteroKANGuard(data, hidden_channels=128, out_channels=2).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
    
    class_weights = torch.tensor([1.0, 10.0], device=DEVICE)

    best_val_f1 = 0; patience_counter = 0; patience = 50
    print("\nBắt đầu huấn luyện HeteroKANGuard...")
    data = data.to(DEVICE)
    for epoch in range(1, 201):
        loss = train(model, data, optimizer, class_weights)
        results, _ = test(model, data)
        val_f1 = results['val_mask']
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_hetero_kanguard_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val F1: {val_f1:.4f}, Test F1: {results["test_mask"]:.4f}')

        if patience_counter >= patience:
            print("Early stopping!"); break

    print("\nĐánh giá cuối cùng với model tốt nhất...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_hetero_kanguard_model.pth')))
    _, test_pred = test(model, data)
    
    report_str = classification_report(data['email'].y[data['email'].test_mask].cpu(), test_pred.cpu(), target_names=['HAM (0)', 'BEC (1)'], zero_division=0)
    
    print("\n--- Báo cáo chi tiết HeteroKANGuard ---")
    print(report_str)
    
    with open(os.path.join(output_dir, 'hetero_kanguard_report.txt'), 'w') as f: f.write(report_str)
    print(f"Báo cáo kết quả HeteroKANGuard đã được lưu vào '{output_dir}/hetero_kanguard_report.txt'")
    print("--- Hoàn thành! ---")

if __name__ == '__main__':
    main()
