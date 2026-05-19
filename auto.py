import os
import time
from datetime import datetime

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Khởi tạo Auto Retrain...")

# Import các hàm kiểm tra từ các module
try:
    from a_ML3 import needs_retrain as nr3, train_model_for_symbol as train3, model_exists as me3
except ImportError:
    pass

try:
    from a_ML4_daily import needs_retrain as nr4, train_model_for_symbol as train4, model_exists as me4
except ImportError:
    pass

try:
    from a_ML4_weekly import needs_retrain as nr_w, train_model_for_symbol as train_w, model_exists as me_w
except ImportError:
    pass

try:
    from ML4_4h import needs_retrain as nr_4h, train_model_for_symbol as train_4h, model_exists as me_4h
except ImportError:
    pass

try:
    from a_ML2 import train_model_for_symbol as train2, model_exists as me2, get_model_paths as get_paths2, CONFIG as config2
except ImportError:
    pass


def auto_retrain_all():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BẮT ĐẦU KIỂM TRA VÀ AUTO-RETRAIN...")
    
    # Ở đây chúng ta sẽ lấy danh sách các mã (symbol) đã được train từ các thư mục
    symbols = set()
    for directory in ["saved_model", "saved_model_pure", "saved_model_weekly"]:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                # parse symbol từ tên file, ví dụ meta_HPG.json, meta_pure_VNM.json
                if filename.endswith(".json") and filename.startswith("meta_"):
                    # file meta thường có dạng meta_VNM.json hoặc meta_pure_VNM.json
                    parts = filename.replace(".json", "").split("_")
                    symbol = parts[-1].upper()
                    symbols.add(symbol)
    
    if not symbols:
        print("Không tìm thấy model nào đã lưu.")
        return

    print(f"Đã tìm thấy {len(symbols)} mã chứng khoán có model: {', '.join(symbols)}")
    
    for symbol in symbols:
        print(f"\n--- Kiểm tra mã {symbol} ---")
        
        # Kiểm tra Model 3
        if 'nr3' in globals() and me3(symbol):
            if nr3(symbol):
                print(f"[Model 3] {symbol} đã cũ. Đang retrain...")
                train3(symbol)
            else:
                print(f"[Model 3] {symbol} vẫn còn mới.")
                
        # Kiểm tra Model 4 (Daily)
        if 'nr4' in globals() and me4(symbol):
            if nr4(symbol):
                print(f"[Model 4 Daily] {symbol} đã cũ. Đang retrain...")
                train4(symbol)
            else:
                print(f"[Model 4 Daily] {symbol} vẫn còn mới.")

        # Kiểm tra Model 4 (Weekly)
        if 'nr_w' in globals() and me_w(symbol):
            if nr_w(symbol):
                print(f"[Model 4 Weekly] {symbol} đã cũ. Đang retrain...")
                train_w(symbol)
            else:
                print(f"[Model 4 Weekly] {symbol} vẫn còn mới.")
                
        # Kiểm tra Model 4 (4H)
        if 'nr_4h' in globals() and me_4h(symbol):
            if nr_4h(symbol):
                print(f"[Model 4H] {symbol} đã cũ. Đang retrain...")
                train_4h(symbol)
            else:
                print(f"[Model 4H] {symbol} vẫn còn mới.")

        # Kiểm tra Model 2 (Lasso + Momentum)
        if 'me2' in globals() and me2(symbol):
            # Model 2 không có sẵn hàm needs_retrain, ta tự check meta file
            paths2 = get_paths2(symbol)
            if os.path.exists(paths2["meta"]):
                import json
                with open(paths2["meta"], "r", encoding="utf-8") as f:
                    meta2 = json.load(f)
                saved_at2 = datetime.fromisoformat(meta2.get("saved_at", "2000-01-01"))
                retrain_every2 = config2.get("retrain_every", 63)
                if (datetime.now() - saved_at2).days >= retrain_every2:
                    print(f"[Model 2] {symbol} đã cũ. Đang retrain...")
                    train2(symbol)
                else:
                    print(f"[Model 2] {symbol} vẫn còn mới.")

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HOÀN TẤT AUTO-RETRAIN.")

if __name__ == "__main__":
    auto_retrain_all()
