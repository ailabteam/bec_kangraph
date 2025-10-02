import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from email import message_from_string
from tqdm import tqdm # Thêm tqdm để có thanh tiến trình đẹp mắt

def read_enron_emails(root_dir):
    """
    Đọc tất cả các file email trong các thư mục con của Enron.
    Trích xuất Subject, From, To, và nội dung (Message Body).
    """
    email_data = []
    print(f"Bắt đầu quét các email Enron từ thư mục: {root_dir}")
    
    # Sử dụng os.walk để duyệt qua tất cả các thư mục và file
    for dirpath, _, filenames in tqdm(os.walk(root_dir), desc="Scanning Enron directories"):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                    msg = message_from_string(content)
                    
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            ctype = part.get_content_type()
                            cdispo = str(part.get('Content-Disposition'))
                            if ctype == 'text/plain' and 'attachment' not in cdispo:
                                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                    else:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

                    email_data.append({
                        'From': msg['From'],
                        'To': msg['To'],
                        'Subject': msg['Subject'],
                        'Message': body
                    })
            except Exception as e:
                # Bỏ qua các file không thể đọc được
                pass
                
    return pd.DataFrame(email_data)


def explore_bec_dataset_v2():
    """
    Hàm chính phiên bản 2, làm việc với cấu trúc dữ liệu thực tế.
    """
    print("--- Bắt đầu quá trình phân tích dữ liệu (v2) ---")

    # --- 1. Thiết lập đường dẫn và thư mục output ---
    BASE_BEC_REPO_PATH = '../bec' 
    output_dir = 'analysis_outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Các kết quả sẽ được lưu vào thư mục: '{output_dir}'")

    # --- 2. Tải dữ liệu BEC từ các file CSV có sẵn ---
    bec_path_1 = os.path.join(BASE_BEC_REPO_PATH, 'data', 'BEC-1-human.csv')
    bec_path_2 = os.path.join(BASE_BEC_REPO_PATH, 'data', 'BEC-2-human.csv')
    
    try:
        df_bec1 = pd.read_csv(bec_path_1)
        df_bec2 = pd.read_csv(bec_path_2)
        df_bec = pd.concat([df_bec1, df_bec2], ignore_index=True)
        print(f"Tải dữ liệu BEC thành công! Tổng số email BEC: {len(df_bec)}")
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu BEC: {e}")
        return

    # --- 3. Tải dữ liệu HAM bằng cách quét thư mục Enron ---
    enron_root_path = os.path.join(BASE_BEC_REPO_PATH, 'enron')
    df_ham = read_enron_emails(enron_root_path)
    print(f"Tải dữ liệu HAM (Enron) thành công! Tổng số email HAM: {len(df_ham)}")


    # --- 4. Chuẩn bị và hợp nhất dữ liệu ---
    print("Chuẩn hóa và hợp nhất dữ liệu...")
    
    # Chuẩn hóa df_bec, giả sử cột nội dung là 'text'
    # Bạn cần kiểm tra tên cột thực tế trong df_bec
    if 'text' not in df_bec.columns and 'body' in df_bec.columns:
        df_bec = df_bec.rename(columns={'body': 'text'})

    df_ham = df_ham.rename(columns={'Subject': 'subject', 'Message': 'text', 'From': 'from', 'To': 'to'})
    
    df_bec['label'] = 1  # BEC
    df_ham['label'] = 0  # HAM
    
    # Giữ lại các cột chung và cần thiết
    common_columns = ['subject', 'text', 'label']
    graph_columns_ham = ['from', 'to']
    
    # df_bec có thể không có cột from/to, chúng ta sẽ thêm vào với giá trị None
    if 'from' not in df_bec.columns: df_bec['from'] = None
    if 'to' not in df_bec.columns: df_bec['to'] = None

    df_combined = pd.concat([
        df_bec[['from', 'to', 'subject', 'text', 'label']], 
        df_ham[['from', 'to', 'subject', 'text', 'label']]
    ], ignore_index=True)
    
    # Loại bỏ các dòng bị thiếu nội dung
    df_combined.dropna(subset=['text'], inplace=True)
    df_combined = df_combined[df_combined['text'].str.strip() != '']

    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    processed_data_path = os.path.join(output_dir, 'combined_dataset_v2.csv')
    df_combined.to_csv(processed_data_path, index=False)
    print(f"Dữ liệu hợp nhất đã được lưu vào: '{processed_data_path}'")
    
    # --- 5. Phân tích dữ liệu hợp nhất (tương tự như trước) ---
    print("Thực hiện phân tích trên dữ liệu hợp nhất...")
    df_combined['text_length'] = df_combined['text'].str.len()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_combined, x='text_length', hue='label', kde=True, bins=100)
    plt.title('Phân phối Độ dài Nội dung Email (BEC vs. HAM) - v2')
    plt.xlabel('Độ dài (số ký tự)')
    plt.ylabel('Số lượng Email')
    plt.xlim(0, 5000)
    fig_path = os.path.join(output_dir, 'text_length_distribution_v2.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Biểu đồ phân phối độ dài đã được lưu vào: '{fig_path}'")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_combined, x='label')
    plt.title('Phân phối Nhãn (0: HAM, 1: BEC) - v2')
    plt.xlabel('Nhãn')
    plt.ylabel('Số lượng')
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    fig_path = os.path.join(output_dir, 'label_distribution_v2.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Biểu đồ phân phối nhãn đã được lưu vào: '{fig_path}'")
    
    print("\n--- Hoàn thành phân tích dữ liệu (v2) ---")

if __name__ == '__main__':
    explore_bec_dataset_v2()
