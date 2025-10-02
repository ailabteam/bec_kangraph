import os
import pandas as pd
from email import message_from_string

def analyze_directory(dir_path):
    """Phân tích một thư mục con của Enron."""
    num_files = 0
    total_size = 0
    sample_files_content = []
    
    if not os.path.isdir(dir_path):
        return None

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            num_files += 1
            total_size += os.path.getsize(file_path)
            
            # Đọc nội dung của 2 file đầu tiên làm ví dụ
            if len(sample_files_content) < 2:
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        msg = message_from_string(content)
                        sample_info = {
                            'filename': filename,
                            'From': msg.get('From', 'N/A'),
                            'To': msg.get('To', 'N/A'),
                            'Subject': msg.get('Subject', 'N/A')
                        }
                        sample_files_content.append(sample_info)
                except:
                    pass

    avg_size = (total_size / num_files) / 1024 if num_files > 0 else 0 # in KB
    
    return {
        'num_files': num_files,
        'avg_size_kb': round(avg_size, 2),
        'samples': sample_files_content
    }

def main():
    enron_root = '../bec/enron'
    
    if not os.path.isdir(enron_root):
        print(f"Lỗi: không tìm thấy thư mục {enron_root}")
        return
        
    subdirectories = [d for d in os.listdir(enron_root) if os.path.isdir(os.path.join(enron_root, d))]
    
    all_stats = []
    
    print("--- Bắt đầu phân tích cấu trúc thư mục Enron ---")
    for subdir in subdirectories:
        print(f"Đang phân tích: {subdir}...")
        dir_path = os.path.join(enron_root, subdir)
        stats = analyze_directory(dir_path)
        if stats:
            stats['directory'] = subdir
            all_stats.append(stats)
            
    # In kết quả ra màn hình
    print("\n--- Tổng hợp kết quả ---")
    for stats in all_stats:
        print(f"\n--- Thư mục: {stats['directory']} ---")
        print(f"  Số lượng file: {stats['num_files']}")
        print(f"  Kích thước trung bình: {stats['avg_size_kb']} KB")
        print("  File ví dụ:")
        for sample in stats['samples']:
            print(f"    - {sample['filename']}:")
            print(f"      From: {str(sample['From']).strip()}")
            print(f"      To: {str(sample['To']).strip()}")
            print(f"      Subject: {str(sample['Subject']).strip()}")

if __name__ == '__main__':
    main()
