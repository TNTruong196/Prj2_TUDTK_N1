import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import class DataPipeline từ file data_pipeline.py
from data_pipeline import DataPipeline

def run_tests():
    print("="*40)
    print("🚀 BẮT ĐẦU TEST DATAPIPELINE")
    print("="*40)
    
    # 1. Đọc dữ liệu
    try:
        df = pd.read_csv('part2/data/AirQuality.csv')
        print(f"[+] Đã đọc file AirQuality.csv. Kích thước ban đầu: {df.shape}")
    except FileNotFoundError:
        print("❌ LỖI: Không tìm thấy file 'AirQuality.csv'. Hãy kiểm tra lại thư mục.")
        return

    # 2. Xử lý target (Bắt buộc phải làm trước khi đưa vào Pipeline)
    target_col = 'C6H6(GT)'
    if target_col in df.columns:
        df_clean = df.dropna(subset=[target_col]).copy()
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        print(f"[+] Đã tách Target. Kích thước X: {X.shape}")
    else:
        print(f"❌ LỖI: Không tìm thấy cột target '{target_col}'")
        return

    # 3. Chia Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[+] Đã chia Train/Test: X_train {X_train.shape}, X_test {X_test.shape}")

    # 4. Khởi tạo và chạy Pipeline
    print("\n⏳ Đang chạy Pipeline...")
    pipeline = DataPipeline(missing_threshold=0.5, numeric_strategy='median')
    
    # Fit & Transform trên Train
    X_train_processed = pipeline.fit_transform(X_train)
    # CHỈ Transform trên Test
    X_test_processed = pipeline.transform(X_test)
    
    print("✅ Chạy Pipeline thành công! Bắt đầu kiểm tra tính toàn vẹn dữ liệu:\n")
    
    # ==========================================
    # CÁC BÀI TEST TỰ ĐỘNG
    # ==========================================
    passed_all = True
    
    # TEST 1: Kiểm tra rò rỉ cấu trúc (Shape mismatch)
    cols_train = set(X_train_processed.columns)
    cols_test = set(X_test_processed.columns)
    if cols_train == cols_test and len(X_train_processed.columns) == len(X_test_processed.columns):
        print("✅ TEST 1 PASSED: Tập Train và Test có số lượng và cấu trúc cột giống hệt nhau.")
    else:
        print("❌ TEST 1 FAILED: Số lượng cột giữa Train và Test bị lệch (Lỗi Reindex).")
        passed_all = False

    # TEST 2: Kiểm tra Missing Values
    n_missing_train = X_train_processed.isna().sum().sum()
    n_missing_test = X_test_processed.isna().sum().sum()
    if n_missing_train == 0 and n_missing_test == 0:
        print("✅ TEST 2 PASSED: Đã điền khuyết toàn bộ, không còn giá trị NaN nào.")
    else:
        print(f"❌ TEST 2 FAILED: Vẫn còn NaN (Train: {n_missing_train}, Test: {n_missing_test}).")
        passed_all = False

    # TEST 3: Kiểm tra Chuẩn hóa Z-score (Mean ≈ 0, Std ≈ 1 trên tập Train)
    numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    # Lấy sai số rất nhỏ (1e-7) vì float trong python xử lý số 0 thường ra dạng 0.00000000000001
    is_mean_zero = X_train_processed[numeric_cols].mean().abs().max() < 1e-7 
    # Độ lệch chuẩn mặc định của std() đôi khi ra NaN với cột dummy 0, ta đã fillna(1)
    is_std_one = np.abs(X_train_processed[numeric_cols].std().fillna(1).mean() - 1.0) < 1e-2 
    
    if is_mean_zero and is_std_one:
        print("✅ TEST 3 PASSED: Dữ liệu đã được chuẩn hóa Z-score hoàn hảo (Mean = 0, Std = 1).")
    else:
        print("❌ TEST 3 FAILED: Công thức chuẩn hóa bị lỗi.")
        passed_all = False

    # ==========================================
    # KẾT LUẬN
    # ==========================================
    print("\n" + "="*40)
    # TEST 4: Kiểm tra chi tiết cực đoan (Null, NaN và Infinity)
    # Lấy danh sách các cột đang bị dính Null/NaN
    null_train_cols = X_train_processed.columns[X_train_processed.isna().any()].tolist()
    null_test_cols = X_test_processed.columns[X_test_processed.isna().any()].tolist()
    
    # Lấy danh sách các cột đang bị dính Inf (Vô cực)
    numeric_train = X_train_processed.select_dtypes(include=[np.number])
    inf_train_cols = numeric_train.columns[np.isinf(numeric_train).any()].tolist()
    
    numeric_test = X_test_processed.select_dtypes(include=[np.number])
    inf_test_cols = numeric_test.columns[np.isinf(numeric_test).any()].tolist()
    
    if not null_train_cols and not null_test_cols and not inf_train_cols and not inf_test_cols:
        print("✅ TEST 4 PASSED: Dữ liệu sạch 100%! Không có bất kỳ phần tử Null, NaN hay Vô cực (Inf) nào.")
    else:
        print("❌ TEST 4 FAILED: Phát hiện phần tử rác trong dữ liệu!")
        if null_train_cols:
            print(f"   -> ⚠️ LỖI: Tập Train bị dính Null tại các cột: {null_train_cols}")
        if null_test_cols:
            print(f"   -> ⚠️ LỖI: Tập Test bị dính Null tại các cột: {null_test_cols}")
        if inf_train_cols:
            print(f"   -> ⚠️ LỖI: Tập Train bị dính Vô cực (Inf) tại các cột: {inf_train_cols}")
        if inf_test_cols:
            print(f"   -> ⚠️ LỖI: Tập Test bị dính Vô cực (Inf) tại các cột: {inf_test_cols}")
        passed_all = False
    
    # TEST 5: Kiểm tra Outlier Handling (Winsorization)
    if hasattr(pipeline, 'outlier_bounds') and len(pipeline.outlier_bounds) > 0:
        # Kiểm tra trên dữ liệu trước chuẩn hóa: inverse Z-score rồi check bounds
        outlier_found = False
        for col, (lower, upper) in pipeline.outlier_bounds.items():
            if col in X_train_processed.columns:
                # Inverse Z-score: x_original = x_zscore * std + mean
                col_mean = pipeline.means[col]
                col_std = pipeline.stds[col] if pipeline.stds[col] != 0 else 1.0
                original_vals = X_train_processed[col] * col_std + col_mean
                violations = ((original_vals < lower - 1e-6) | (original_vals > upper + 1e-6)).sum()
                if violations > 0:
                    outlier_found = True
                    break
        if not outlier_found:
            print("TEST 5 PASSED: Outliers da duoc xu ly bang Winsorization (IQR Capping).")
            print(f"   -> So cot duoc Winsorize: {len(pipeline.outlier_bounds)}")
        else:
            print("TEST 5 FAILED: Van con outlier sau khi Winsorization.")
            passed_all = False
    else:
        print("TEST 5 FAILED: Pipeline khong co outlier_bounds (chua xu ly outlier).")
        passed_all = False

    if passed_all:
        print("🎉 XUẤT SẮC! Class DataPipeline hoạt động hoàn hảo và sẵn sàng để train.")
        print("Danh sách các cột đã bị drop (Missing > 50%):", pipeline.cols_to_drop)
        print("\n👀 Xem thử 3 dòng dữ liệu Train sau xử lý:")
        print(X_train_processed.head(3))
    else:
        print("⚠️ CẦN SỬA LỖI: Vui lòng kiểm tra lại code class DataPipeline.")
        
if __name__ == "__main__":
    run_tests()