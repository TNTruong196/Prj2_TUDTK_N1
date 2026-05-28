import pandas as pd
import numpy as np

class DataPipeline:
    """
    Pipeline xử lý dữ liệu cho bộ Air Quality theo mô hình fit/transform.

    Quy trình:
        1. Feature engineering (Date/Time → Month, DayOfWeek, Hour)
        2. Loại bỏ cột có tỉ lệ missing vượt ngưỡng (missing_threshold)
        3. Điền khuyết (Imputation) bằng mean/median
        4. Xử lý outliers bằng IQR Capping (Winsorization)
        5. Mã hóa biến phân loại (One-Hot Encoding)
        6. Chuẩn hóa Z-score

    Biện luận chiến lược Imputation (dựa trên phân tích EDA):
        Dữ liệu Air Quality có cơ chế missing phổ biến nhất là MAR
        (Missing At Random) — giá trị thiếu phụ thuộc vào các biến quan
        sát được khác (ví dụ: nhiệt độ cao khiến cảm biến CO tạm dừng).
        Do đó, ta KHÔNG dùng Listwise Deletion (xóa dòng) vì sẽ mất
        quá nhiều thông tin, mà thay vào đó dùng Imputation (Mean/Median)
        để bảo toàn dữ liệu. Các cột có tỉ lệ missing quá cao (> threshold)
        sẽ bị loại bỏ hoàn toàn vì imputation quá nhiều giá trị trong 1 cột
        sẽ làm dữ liệu không đáng tin cậy.
    """

    def __init__(self, missing_threshold=0.5, numeric_strategy='median'):
        """
        Khởi tạo Pipeline.

        Parameters:
            missing_threshold (float): Tỉ lệ missing tối đa để giữ lại cột.
                Mặc định 0.5 (50%).
            numeric_strategy (str): Chiến lược điền khuyết cho biến số thực.
                'mean' hoặc 'median'. Mặc định 'median' vì median ít bị
                ảnh hưởng bởi outliers hơn mean — phù hợp với dữ liệu
                cảm biến thường có nhiều giá trị ngoại lai.
        """
        self.missing_threshold = missing_threshold
        self.numeric_strategy = numeric_strategy

        # Các tham số sẽ được học từ tập Train (để tránh data leakage)
        self.cols_to_drop = []
        self.impute_values = {}
        self.outlier_bounds = {}  # {col: (lower, upper)} — giới hạn IQR cho Winsorization
        self.categorical_cols = []
        self.encoding_columns = None
        self.means = None
        self.stds = None

    def _extract_temporal_features(self, X):
        """Feature engineering cho Date và Time."""
        df_temp = X.copy()
        if 'Date' in df_temp.columns:
            # Ép kiểu format chuẩn của data Mỹ (Tháng/Ngày/Năm) để tránh parse nhầm
            dt = pd.to_datetime(df_temp['Date'], format='%m/%d/%Y', errors='coerce')
            df_temp['Month'] = dt.dt.month
            df_temp['DayOfWeek'] = dt.dt.dayofweek
            df_temp.drop(columns=['Date'], inplace=True)
            
        if 'Time' in df_temp.columns:
            df_temp['Hour'] = pd.to_datetime(df_temp['Time'], format='%H:%M:%S', errors='coerce').dt.hour
            df_temp.drop(columns=['Time'], inplace=True)
            
        return df_temp

    def fit(self, X):
        # 1. Trích xuất thời gian trước để tính toán trên các cột mới
        X_processed = self._extract_temporal_features(X)
        
        # 2. Xác định cột cần xóa do missing quá cao
        missing_ratios = X_processed.isna().mean()
        self.cols_to_drop = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        X_reduced = X_processed.drop(columns=self.cols_to_drop)
        
        # 3. Tính giá trị điền khuyết (Imputation)
        # Biện luận: Cơ chế missing của Air Quality là MAR, nên dùng Imputation
        # thay vì xóa dòng. Median được ưu tiên vì dữ liệu cảm biến
        # thường chứa outliers, median ít bị ảnh hưởng hơn mean.
        for col in X_reduced.columns:
            if X_reduced[col].dtype == 'object':
                self.impute_values[col] = X_reduced[col].mode()[0]
            elif col in ['Month', 'DayOfWeek', 'Hour']:
                # Đặc thù: Các biến chu kỳ thời gian luôn dùng median để ra số nguyên
                self.impute_values[col] = X_reduced[col].median()
            else:
                self.impute_values[col] = X_reduced[col].mean() if self.numeric_strategy == 'mean' else X_reduced[col].median()

        # 4. Tính giới hạn IQR cho Winsorization (chỉ trên tập Train)
        # Biện luận: Dùng Winsorization (clip về biên) thay vì loại bỏ outlier
        # vì đã chọn chiến lược bảo toàn dữ liệu (imputation cho missing values).
        # Loại bỏ outlier sẽ mâu thuẫn — vừa impute để giữ data, vừa xóa data.
        X_temp = X_reduced.fillna(self.impute_values)
        numeric_cols = X_temp.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Month', 'DayOfWeek', 'Hour']:  # Không clip biến thời gian
                Q1 = X_temp[col].quantile(0.25)
                Q3 = X_temp[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                self.outlier_bounds[col] = (lower, upper)
                X_temp[col] = X_temp[col].clip(lower=lower, upper=upper)

        # 5. Tính toán cho Encoding
        
        # Lưu lại TÊN CÁC CỘT object để ép tập Test phải tuân thủ
        self.categorical_cols = X_temp.select_dtypes(include=['object']).columns.tolist()
        
        if len(self.categorical_cols) > 0:
            X_encoded = pd.get_dummies(X_temp, columns=self.categorical_cols, dtype=float)            
        else:
            X_encoded = X_temp.copy()

        # 6. Lưu lại cấu trúc cột và tham số chuẩn hóa
        self.encoding_columns = X_encoded.columns
        self.means = X_encoded.mean()
        self.stds = X_encoded.std()
        
        return self

    def transform(self, X):
        # Thực hiện đúng thứ tự yêu cầu 
        X_out = self._extract_temporal_features(X)
        X_out = X_out.drop(columns=self.cols_to_drop, errors='ignore')
        
        # 1. Xử lý Missing Values bằng giá trị đã học từ Train
        X_out = X_out.fillna(self.impute_values)

        # 2. Xử lý Outliers bằng IQR Capping (Winsorization)
        # Dùng bounds đã học từ Train để clip cả Train lẫn Test
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower=lower, upper=upper)

        # 3. Encoding (Chỉ encode đúng những cột đã học ở Train)
        if len(self.categorical_cols) > 0:
            X_out = pd.get_dummies(X_out, columns=self.categorical_cols, dtype=float)

        # Reindex để đảm bảo tập Test có đủ (và chỉ có) các cột như tập Train
        X_out = X_out.reindex(columns=self.encoding_columns, fill_value=0)
        
        # 4. Chuẩn hóa Z-score
        # Thay thế 0 bằng 1 và fillna(1) để tránh lỗi chia cho 0 nếu cột std bị NaN
        stds_safe = self.stds.replace(0, 1).fillna(1)
        X_out = (X_out - self.means) / stds_safe
        
        return X_out

    def fit_transform(self, X):
        """Hàm tiện ích kết hợp fit và transform"""
        return self.fit(X).transform(X)

# Test

from sklearn.model_selection import train_test_split

def run_tests():
    print("="*40)
    print("BẮT ĐẦU TEST DATAPIPELINE")
    print("="*40)
    
    # 1. Đọc dữ liệu
    try:
        # Giả định đường dẫn tương đối so với gốc dự án
        df = pd.read_csv('part2/data/AirQuality.csv')
        print(f"[+] Đã đọc file AirQuality.csv. Kích thước ban đầu: {df.shape}")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'part2/data/AirQuality.csv'. Hãy kiểm tra lại thư mục.")
        return

    # 2. Xử lý target (Bắt buộc phải làm trước khi đưa vào Pipeline)
    target_col = 'C6H6(GT)'
    if target_col in df.columns:
        df_clean = df.dropna(subset=[target_col]).copy()
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        print(f"[+] Đã tách Target. Kích thước X: {X.shape}")
    else:
        print(f"LỖI: Không tìm thấy cột target '{target_col}'")
        return

    # 3. Chia Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[+] Đã chia Train/Test: X_train {X_train.shape}, X_test {X_test.shape}")

    # 4. Khởi tạo và chạy Pipeline
    print("\nĐang chạy Pipeline...")
    pipeline = DataPipeline(missing_threshold=0.5, numeric_strategy='median')
    
    # Fit & Transform trên Train
    X_train_processed = pipeline.fit_transform(X_train)
    # CHỈ Transform trên Test
    X_test_processed = pipeline.transform(X_test)
    
    print("Chạy Pipeline thành công! Bắt đầu kiểm tra tính toàn vẹn dữ liệu:\n")
    
    # ==========================================
    # CÁC BÀI TEST TỰ ĐỘNG
    # ==========================================
    passed_all = True
    
    # TEST 1: Kiểm tra rò rỉ cấu trúc (Shape mismatch)
    cols_train = set(X_train_processed.columns)
    cols_test = set(X_test_processed.columns)
    if cols_train == cols_test and len(X_train_processed.columns) == len(X_test_processed.columns):
        print("TEST 1 PASSED: Tập Train và Test có số lượng và cấu trúc cột giống hệt nhau.")
    else:
        print("TEST 1 FAILED: Số lượng cột giữa Train và Test bị lệch (Lỗi Reindex).")
        passed_all = False

    # TEST 2: Kiểm tra Missing Values
    n_missing_train = X_train_processed.isna().sum().sum()
    n_missing_test = X_test_processed.isna().sum().sum()
    if n_missing_train == 0 and n_missing_test == 0:
        print("TEST 2 PASSED: Đã điền khuyết toàn bộ, không còn giá trị NaN nào.")
    else:
        print(f"TEST 2 FAILED: Vẫn còn NaN (Train: {n_missing_train}, Test: {n_missing_test}).")
        passed_all = False

    # TEST 3: Kiểm tra Chuẩn hóa Z-score (Mean ≈ 0, Std ≈ 1 trên tập Train)
    numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
    # Lấy sai số rất nhỏ (1e-7) vì float trong python xử lý số 0 thường ra dạng 0.00000000000001
    is_mean_zero = X_train_processed[numeric_cols].mean().abs().max() < 1e-7 
    # Độ lệch chuẩn mặc định của std() đôi khi ra NaN với cột dummy 0, ta đã fillna(1)
    is_std_one = np.abs(X_train_processed[numeric_cols].std().fillna(1).mean() - 1.0) < 1e-2 
    
    if is_mean_zero and is_std_one:
        print("TEST 3 PASSED: Dữ liệu đã được chuẩn hóa Z-score hoàn hảo (Mean = 0, Std = 1).")
    else:
        print("TEST 3 FAILED: Công thức chuẩn hóa bị lỗi.")
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
        print("TEST 4 PASSED: Dữ liệu sạch 100%! Không có bất kỳ phần tử Null, NaN hay Vô cực (Inf) nào.")
    else:
        print("TEST 4 FAILED: Phát hiện phần tử rác trong dữ liệu!")
        if null_train_cols:
            print(f"   -> LỖI: Tập Train bị dính Null tại các cột: {null_train_cols}")
        if null_test_cols:
            print(f"   -> LỖI: Tập Test bị dính Null tại các cột: {null_test_cols}")
        if inf_train_cols:
            print(f"   -> LỖI: Tập Train bị dính Vô cực (Inf) tại các cột: {inf_train_cols}")
        if inf_test_cols:
            print(f"   -> LỖI: Tập Test bị dính Vô cực (Inf) tại các cột: {inf_test_cols}")
        passed_all = False
    
    # TEST 5: Kiểm tra Outlier Handling (Winsorization)
    if hasattr(pipeline, 'outlier_bounds') and len(pipeline.outlier_bounds) > 0:
        # Kiểm tra trên dữ liệu trước chuẩn hóa: inverse Z-score rồi check bounds
        outlier_found = False
        for col, (lower, upper) in pipeline.outlier_bounds.items():
            if col in X_train_processed.columns:
                # Inverse Z-score: x_original = x_zscore * std + mean
                col_mean = pipeline.means[col]
                col_std = pipeline.stds[col] if pipeline.stds[col] != 1 else 1.0 # fixed logic
                original_vals = X_train_processed[col] * col_std + col_mean
                violations = ((original_vals < lower - 1e-6) | (original_vals > upper + 1e-6)).sum()
                if violations > 0:
                    outlier_found = True
                    break
        if not outlier_found:
            print("TEST 5 PASSED: Outliers đã được xử lý bằng Winsorization (IQR Capping).")
            print(f"   -> Số cột được Winsorize: {len(pipeline.outlier_bounds)}")
        else:
            print("TEST 5 FAILED: Vẫn còn outlier sau khi Winsorization.")
            passed_all = False
    else:
        print("TEST 5 FAILED: Pipeline không có outlier_bounds (chưa xử lý outlier).")
        passed_all = False

    if passed_all:
        print("XUẤT SẮC! Class DataPipeline hoạt động hoàn hảo và sẵn sàng để train.")
        print("Danh sách các cột đã bị drop (Missing > 50%):", pipeline.cols_to_drop)
        print("\nXem thử 3 dòng dữ liệu Train sau xử lý:")
        print(X_train_processed.head(3))
    else:
        print("CẦN SỬA LỖI: Vui lòng kiểm tra lại code class DataPipeline.")
        
if __name__ == "__main__":
    run_tests()
