# Giải thích kiểm thử hàm: `model_metrics`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `model_metrics` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ols_stats.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ols_stats.py), các tham số đầu vào được cung cấp như sau:
*   **Vector thực tế `y`**: Kích thước $15 \times 1$.
*   **Vector dự báo `y_hat`**: Kích thước $15 \times 1$.
*   **Số lượng biến độc lập `p`**: Số nguyên ($p = 3$, không bao gồm hệ số chặn).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về một từ điển (`dict`) chứa các chỉ số chất lượng mô hình:
*   **RSS (Residual Sum of Squares)**: Tổng bình phương phần dư.
*   **TSS (Total Sum of Squares)**: Tổng bình phương sai lệch toàn bộ.
*   **R_squared ($R^2$)**: Hệ số xác định.
*   **Adj_R_squared ($R^2_{\text{adjusted}}$)**: Hệ số xác định hiệu chỉnh bậc tự do.
*   **F_statistic (Trị thống kê F)**: Giá trị thống kê dùng cho kiểm định ý nghĩa đồng thời của các hệ số.
*   **F_p_value (Giá trị p của kiểm định F)**: Mức ý nghĩa thống kê của mô hình.

## 3. Ý nghĩa thống kê & toán học
Các chỉ số đo lường mức độ phù hợp của mô hình hồi quy tuyến tính:
*   **Hệ số xác định**:
    $$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}$$
*   **Hệ số xác định hiệu chỉnh**: Hiệu chỉnh lại số lượng biến độc lập để tránh hiện tượng tăng ảo $R^2$ khi thêm biến nhiễu:
    $$R^2_{\text{adj}} = 1 - \left[ \frac{\text{RSS}/(n - p - 1)}{\text{TSS}/(n - 1)} \right]$$
*   **Kiểm định F**: Đánh giá giả thuyết $H_0: \beta_1 = \beta_2 = \dots = \beta_p = 0$:
    $$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n - p - 1)}$$

## 4. Ghi chú về kiểm thử riêng biệt
Mô hình kiểm thử khớp dữ liệu giả lập cho kết quả ý nghĩa thống kê cao ($p$-value của kiểm định F cực kỳ nhỏ, tiến sát về 0) và hệ số xác định hiệu chỉnh cao, chứng tỏ biến độc lập giải thích tốt biến mục tiêu.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `model_metrics` chỉ trả về cấu trúc từ điển dữ liệu thô. Trong phần kiểm thử này, chúng tôi đã trực quan hóa kết quả đầu ra thành một **Bảng tổng hợp chỉ số mô hình (Model Summary Metrics Table)** có khung viền hộp ASCII đôi sang trọng, hiển thị chi tiết các thông số làm tài liệu minh họa trực tiếp cho báo cáo.
