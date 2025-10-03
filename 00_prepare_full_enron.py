import os
import pandas as pd
from email import message_from_string
from email.utils import getaddresses
from tqdm import tqdm
import random

def parse_email_addresses(address_string):
    """Trích xuất danh sách email từ một chuỗi header (From, To, Cc)."""
    if not isinstance(address_string, str):
        return []
    # getaddresses trả về list các tuple (real_name, email_address)
    addresses = getaddresses([address_string])
    return [addr.lower() for name, addr in addresses if '@' in addr]

def get_email_body(msg):
    """Lấy nội dung text/plain từ một đối tượng email."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                try:
                    # Thử decode với utf-8 trước
                    body = part.get_payload(decode=True).decode('utf-8')
                except UnicodeDecodeError:
                    # Nếu lỗi, thử với các encoding phổ biến khác
                    body = part.get_payload(decode=True).decode('latin-1', errors='ignore')
                return body
    else:
        # Email không phải multipart
        try:
            body = msg.get_payload(decode=True).decode('utf-8')
        except UnicodeDecodeError:
            body = msg.get_payload(decode=True).decode('latin-1', errors='ignore')
    return body

def process_enron_maildir(root_dir, num_emails_to_sample=20000):
    """
    Quét thư mục maildir của Enron, trích xuất thông tin, và lấy mẫu.
    """
    email_data = []
    all_email_paths = []

    print(f"Bắt đầu quét tất cả các file email trong {root_dir}...")
    # Quét để lấy danh sách tất cả các đường dẫn file trước
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Bỏ qua các file không phải là email (ví dụ, file ẩn)
            if not filename.startswith('.'):
                all_email_paths.append(os.path.join(dirpath, filename))

    print(f"Tìm thấy tổng cộng {len(all_email_paths)} file email.")
    print(f"Bắt đầu lấy mẫu ngẫu nhiên và xử lý {num_emails_to_sample} email...")

    # Lấy mẫu ngẫu nhiên để xử lý
    random.seed(42)
    sampled_paths = random.sample(all_email_paths, min(num_emails_to_sample, len(all_email_paths)))

    for file_path in tqdm(sampled_paths, desc="Đang xử lý email"):
        try:
            with open(file_path, 'r', errors='ignore') as f:
                content = f.read()
                msg = message_from_string(content)
                
                body = get_email_body(msg)
                
                # Chỉ xử lý các email có nội dung
                if not body or not body.strip():
                    continue

                # Trích xuất địa chỉ email
                from_addr = parse_email_addresses(msg['From'])
                to_addrs = parse_email_addresses(msg['To'])

                # Chỉ giữ lại các email có ít nhất 1 người gửi và 1 người nhận hợp lệ
                if not from_addr or not to_addrs:
                    continue

                email_data.append({
                    'From': from_addr[0], # Chỉ lấy địa chỉ email đầu tiên
                    'To': ', '.join(to_addrs), # Nối các người nhận lại
                    'Subject': str(msg['Subject']),
                    'Message': body
                })
        except Exception:
            continue
            
    return pd.DataFrame(email_data)

def main():
    # --- THAY ĐỔI ĐƯỜNG DẪN NÀY NẾU CẦN ---
    # Giả sử thư mục 'maildir' nằm cùng cấp với repo 'bec_kangraph'
    enron_root = '../maildir'
    output_path = 'analysis_outputs/enron_20k_full_headers.csv'
    
    if not os.path.isdir(enron_root):
        print(f"Lỗi: Không tìm thấy thư mục '{enron_root}'.")
        print("Vui lòng tải và giải nén bộ dữ liệu Enron vào đúng vị trí.")
        return

    df_enron = process_enron_maildir(enron_root, num_emails_to_sample=20000)
    
    print(f"\nĐã xử lý và lọc được {len(df_enron)} email hợp lệ.")
    print("5 dòng đầu của bộ dữ liệu mới:")
    print(df_enron.head())

    # Lưu ra file CSV
    df_enron.to_csv(output_path, index=False)
    print(f"\nBộ dữ liệu HAM mới đã được lưu vào: {output_path}")

if __name__ == '__main__':
    main()
