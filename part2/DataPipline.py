import pandas as pd
import numpy as np

class DataPipeline:
    def __init__(self, missing_threshold=0.5, numeric_strategy='median'):
        """
        missing_threshold: Tỉ lệ missing tối đa để giữ lại cột.
        numeric_strategy: Chiến lược điền khuyết cho biến số thực ('mean' hoặc 'median').
        """
        self.missing_threshold = missing_threshold
        self.numeric_strategy = numeric_strategy
        
        # Các tham số sẽ được học từ tập Train
        self.cols_to_drop = []
        self.impute_values = {}
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
        for col in X_reduced.columns:
            if X_reduced[col].dtype == 'object':
                self.impute_values[col] = X_reduced[col].mode()[0]
            elif col in ['Month', 'DayOfWeek', 'Hour']: 
                # Đặc thù: Các biến chu kỳ thời gian luôn dùng median để ra số nguyên
                self.impute_values[col] = X_reduced[col].median()
            else:
                self.impute_values[col] = X_reduced[col].mean() if self.numeric_strategy == 'mean' else X_reduced[col].median()
        
        # 4. Tính toán cho Encoding
        X_temp = X_reduced.fillna(self.impute_values)
        
        # Lưu lại TÊN CÁC CỘT object để ép tập Test phải tuân thủ
        self.categorical_cols = X_temp.select_dtypes(include=['object']).columns.tolist()
        
        if len(self.categorical_cols) > 0:
            X_encoded = pd.get_dummies(X_temp, columns=self.categorical_cols, dtype=float)
        else:
            X_encoded = X_temp.copy()
            
        # 5. Lưu lại cấu trúc cột và tham số chuẩn hóa
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
        
        # 2. Encoding (Chỉ encode đúng những cột đã học ở Train)
        if len(self.categorical_cols) > 0:
            X_out = pd.get_dummies(X_out, columns=self.categorical_cols, dtype=float)
        
        # Reindex để đảm bảo tập Test có đủ (và chỉ có) các cột như tập Train
        X_out = X_out.reindex(columns=self.encoding_columns, fill_value=0)
        
        # 3. Chuẩn hóa Z-score
        # Thay thế 0 bằng 1 và fillna(1) để tránh lỗi chia cho 0 nếu cột std bị NaN
        stds_safe = self.stds.replace(0, 1).fillna(1)
        X_out = (X_out - self.means) / stds_safe
        
        return X_out

    def fit_transform(self, X):
        """Hàm tiện ích kết hợp fit và transform"""
        return self.fit(X).transform(X)