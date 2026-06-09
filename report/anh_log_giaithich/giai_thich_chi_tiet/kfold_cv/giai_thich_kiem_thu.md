# Giải thích kiểm thử hàm: `kfold_cv`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `kfold_cv` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ridge_lasso_cv.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ridge_lasso_cv.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $50 \times 6$.
*   **Vector biến mục tiêu `y`**: Kích thước $50 \times 1$.
*   **Số lượng phân hoạch `k`**: $k = 5$ (đánh giá chéo 5-Fold).
*   **Hàm khớp mô hình `fit_func`**: Nhận hàm lambda tương ứng gọi `ridge_fit` hoặc `lasso_fit` cho từng bước kiểm thử lambda.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Lỗi đánh giá chéo trung bình `cv_mse`**: Lỗi bình phương trung bình kiểm định trung bình của $5$ lượt phân hoạch.
*   **Danh sách lỗi từng phân hoạch `fold_mse`**: Mảng chứa lỗi MSE thực tế của từng phân hoạch trong 5 lượt.

## 3. Ý nghĩa thống kê & toán học
Phương pháp đánh giá chéo K-Fold Cross-Validation chia tập dữ liệu ngẫu nhiên thành $K$ phần có kích thước xấp xỉ nhau (folds). Thuật toán thực hiện lặp $K$ lần:
*   Tại lượt $i$, sử dụng phần thứ $i$ làm tập kiểm định (validation set) và $K-1$ phần còn lại làm tập huấn luyện (training set).
*   Khớp mô hình trên tập huấn luyện và dự báo sai số bình phương trung bình (MSE) trên tập kiểm định:
    $$\text{MSE}_i = \frac{1}{n_i} \sum_{j \in \text{Fold}_i} (y_j - \hat{y}_j)^2$$
*   Tính giá trị lỗi đánh giá chéo trung bình:
    $$\text{CV-MSE} = \frac{1}{K} \sum_{i=1}^K \text{MSE}_i$$
Phương pháp này giúp đánh giá năng lực tổng quát hóa của mô hình trên dữ liệu mới một cách công tâm, loại bỏ thiên lệch từ việc chia tập dữ liệu tĩnh, đồng thời là cơ chế chuẩn để tinh chỉnh hyperparameters ($\lambda$).

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi thực hiện đánh giá chéo 5-Fold cho một dãy 9 giá trị lambda khác nhau của cả Ridge và Lasso. Kết quả so sánh chỉ ra giá trị lambda tối ưu giúp giảm lỗi dự báo trên tập kiểm định hiệu quả nhất.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `kfold_cv` chỉ thực hiện chia tập và trả về số liệu tính toán. Trong kịch bản kiểm thử này, chúng tôi đã thu thập lỗi của từng lambda và trực quan hóa thành **Đồ thị sai số Cross-Validation** với trục hoành $\log_{10}(\lambda)$ để làm nổi bật vị trí điểm tối ưu của Ridge và Lasso.
