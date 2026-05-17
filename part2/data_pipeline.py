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