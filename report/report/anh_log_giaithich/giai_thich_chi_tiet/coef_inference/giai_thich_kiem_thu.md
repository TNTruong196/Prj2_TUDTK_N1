# Giải thích kiểm thử hàm: `coef_inference`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `coef_inference` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ols_stats.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ols_stats.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $15 \times 4$.
*   **Vector biến mục tiêu `y`**: Kích thước $15 \times 1$.
*   **Vector hệ số ước lượng `beta_hat`**: Kích thước $4 \times 1$.
*   **Ước lượng phương sai nhiễu `sigma2`**: Số thực được tính từ mô hình OLS.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về một từ điển (`dict`) chứa các danh sách thông số suy diễn cho từng hệ số hồi quy $\beta_j$:
*   **Standard_Errors (SE)**: Sai số chuẩn của các ước lượng hệ số.
*   **t_statistics (t-stat)**: Giá trị thống kê t cho kiểm định ý nghĩa từng phần.
*   **p_values**: Giá trị p-value tương ứng với kiểm định t hai phía.
*   **CI_95**: Danh sách các tuple chứa khoảng tin cậy 95% của từng hệ số.

## 3. Ý nghĩa thống kê & toán học
Hàm thực hiện suy diễn thống kê dựa trên phân phối mẫu của các ước lượng OLS dưới giả thuyết nhiễu có phân phối chuẩn:
*   **Ma trận hiệp phương sai của $\hat{\beta}$**:
    $$\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$$
*   **Sai số chuẩn của $\hat{\beta}_j$**:
    $$\text{SE}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 C_{jj}} \quad \text{với} \quad C = (X^T X)^{-1}$$
*   **Kiểm định t**: Để kiểm tra ý nghĩa của từng hệ số hồi quy riêng lẻ $H_0: \beta_j = 0$:
    $$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t(n - p - 1)$$
*   **Khoảng tin cậy 95%**:
    $$\hat{\beta}_j \pm t_{\alpha/2, n-p-1} \times \text{SE}(\hat{\beta}_j)$$

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi chạy thử nghiệm suy diễn hệ số trên mẫu giả lập kích thước nhỏ $n=15$ nhằm chỉ ra sự khác biệt rõ nét về bậc tự do và phân phối t của Student so với phân phối chuẩn tiệm cận. Kết quả kiểm tra cũng chỉ ra biến nào có ý nghĩa thống kê thật sự bằng dấu hoa thị `*` nếu $p$-value $<0.05$.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `coef_inference` chỉ trả về từ điển chứa các mảng số thực. Trong phần kiểm thử này, kết quả đầu ra đã được trực quan hóa thêm bằng một **Bảng suy diễn hệ số hồi quy chi tiết (Coefficient Inference Table)** giúp dễ dàng quan sát, chụp ảnh và chèn vào báo cáo thực hành.
