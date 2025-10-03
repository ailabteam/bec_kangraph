import pandas as pd
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def create_hard_negative_dataset(k_neighbors=5):
    """
    Tạo một bộ dữ liệu thách thức hơn bằng cách tìm các mẫu "hard negative".
    Hard negatives là các email HAM (hợp lệ) có nội dung giống với email BEC (lừa đảo) nhất.
    """
    print("--- Bắt đầu tạo bộ dữ liệu Hard Negative (v4) ---")

    # --- 1. Tải dữ liệu ---
    output_dir = 'analysis_outputs'
    data_path = os.path.join(output_dir, 'combined_dataset_v3.csv')
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file {data_path}. Vui lòng chạy 01_explore_data.py trước.")
        return
        
    df = pd.read_csv(data_path)
    df.dropna(subset=['text'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Tách riêng BEC và HAM
    df_bec = df[df['label'] == 1].copy()
    df_ham = df[df['label'] == 0].copy()

    print(f"Đã tải {len(df_bec)} email BEC và {len(df_ham)} email HAM.")

    # --- 2. Tải mô hình Sentence-BERT ---
    # 'all-MiniLM-L6-v2' là một mô hình rất nhanh và hiệu quả
    print("Tải mô hình Sentence-BERT (có thể mất vài phút lần đầu)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Kiểm tra GPU
    if torch.cuda.is_available():
        model = model.to('cuda')
    print("Mô hình đã sẵn sàng.")

    # --- 3. Tạo Embeddings ---
    print("Tạo embeddings cho tất cả các email...")
    
    # Chuyển đổi văn bản thành list để encode
    bec_texts = df_bec['text'].tolist()
    ham_texts = df_ham['text'].tolist()

    # Tạo embedding (có thể mất vài phút)
    bec_embeddings = model.encode(bec_texts, convert_to_tensor=True, show_progress_bar=True)
    ham_embeddings = model.encode(ham_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Tạo embeddings hoàn tất.")

    # --- 4. Tìm kiếm Hard Negatives ---
    print(f"Bắt đầu tìm kiếm {k_neighbors} mẫu HAM khó nhất cho mỗi mẫu BEC...")
    
    # Sử dụng `util.semantic_search` để tìm kiếm hiệu quả
    # Hàm này sẽ tính cosine similarity giữa tất cả các cặp (bec, ham)
    # và trả về top_k kết quả cho mỗi mẫu bec.
    hits = util.semantic_search(bec_embeddings, ham_embeddings, top_k=k_neighbors)
    
    # Tập hợp các index của các mẫu HAM khó
    hard_negative_indices = set()
    for hit_list in hits:
        for hit in hit_list:
            hard_negative_indices.add(hit['corpus_id'])
            
    print(f"Đã tìm thấy {len(hard_negative_indices)} mẫu HAM khó duy nhất.")

    # --- 5. Tạo bộ dữ liệu mới ---
    # Lấy các mẫu HAM khó từ dataframe ban đầu
    df_hard_ham = df_ham.iloc[list(hard_negative_indices)]
    
    # Kết hợp lại với bộ dữ liệu BEC
    df_final = pd.concat([df_bec, df_hard_ham], ignore_index=True)
    
    # Trộn dữ liệu
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # --- 6. Lưu kết quả ---
    final_path = os.path.join(output_dir, 'combined_dataset_v4_hard.csv')
    df_final.to_csv(final_path, index=False)
    
    print("\n--- Hoàn thành! ---")
    print(f"Bộ dữ liệu mới đã được lưu vào: {final_path}")
    print(f"Tổng số mẫu: {len(df_final)}")
    print("Phân phối nhãn trong bộ dữ liệu mới:")
    print(df_final['label'].value_counts())


if __name__ == '__main__':
    # Chúng ta sẽ tìm 5 mẫu HAM khó cho mỗi mẫu BEC
    # Tổng số mẫu HAM sẽ là khoảng 5 * 299 = ~1500
    create_hard_negative_dataset(k_neighbors=5)
