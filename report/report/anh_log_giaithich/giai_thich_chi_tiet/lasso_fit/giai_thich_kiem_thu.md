# Giải thích kiểm thử hàm: `lasso_fit`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `lasso_fit` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ridge_lasso_cv.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ridge_lasso_cv.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $50 \times 6$.
*   **Vector biến mục tiêu `y`**: Kích thước $50 \times 1$.
*   **Hệ số phạt L1 `lam` ($\lambda$)**: Số thực không âm đại diện cho mức độ phạt (ví dụ: $\lambda = 0.1$).
*   **Tham số thuật toán**: `max_iter=1000` (số vòng lặp tối đa của Coordinate Descent) và `tol=1e-6` (ngưỡng hội tụ).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Vector hệ số Lasso `beta`**: Kích thước $6 \times 1$ dạng danh sách 2D. Các biến nhiễu hoặc không quan trọng được kỳ vọng triệt tiêu về mức đúng $0.0000$.

## 3. Ý nghĩa thống kê & toán học
Hồi quy Lasso (Least Absolute Shrinkage and Selection Operator) bổ sung thành phần phạt L1 (tổng trị tuyệt đối của các hệ số) vào hàm mất mát:
$$\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$
Do hình học của khối cầu L1 có các đỉnh nhọn trùng với các trục tọa độ, Lasso ép một số hệ số hồi quy về đúng 0 khi giá trị $\lambda$ đủ lớn. Đây là tính năng tự động chọn biến (Feature Selection) vô cùng quý giá trong học máy và thống kê thực nghiệm, giúp tạo ra các mô hình tinh gọn, dễ giải thích.
Thuật toán được cài đặt bằng phương pháp giảm tọa độ tuần tự (Coordinate Descent).

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi tạo ra đặc trưng $x_3$ là đặc trưng hoàn toàn độc lập với biến mục tiêu (hệ số thực tế bằng $0.0$). Trong kết quả kiểm thử, hệ số của biến độc lập $x_3$ dưới ước lượng Lasso bị ép hoàn toàn về $0.000$ (trong khi OLS hoặc Ridge chỉ co ngắn nhưng vẫn khác 0).

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `lasso_fit` chỉ thực hiện tính toán số học và không vẽ đồ thị. Trong kịch bản kiểm thử này, chúng tôi đã vẽ đồ thị vết hệ số Lasso **Lasso Coefficient Path** tương ứng với lưới $100$ điểm lambda để so sánh trực quan hiệu ứng chọn đặc trưng của Lasso so với Ridge.
