# Giải thích kiểm thử hàm: `ridge_fit`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `ridge_fit` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ridge_lasso_cv.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ridge_lasso_cv.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $50 \times 6$ ($n=50$ dòng quan sát, $p+1=6$ cột đặc trưng kể cả hệ số chặn).
*   **Vector biến mục tiêu `y`**: Kích thước $50 \times 1$.
*   **Hệ số regularization `lam` ($\lambda$)**: Số thực không âm đại diện cho mức độ phạt (ví dụ: $\lambda = 1.0$).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Vector hệ số hồi quy Ridge `beta`**: Kích thước $6 \times 1$ chứa các giá trị hệ số co rút.

## 3. Ý nghĩa thống kê & toán học
Hồi quy Ridge giải quyết bài toán đa cộng tuyến hoặc quá khớp (overfitting) của OLS bằng cách thêm thành phần phạt L2 (tổng bình phương các hệ số) vào hàm mất mát:
$$\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I')^{-1} X^T y$$
Trong đó, ma trận $I'$ là ma trận đơn vị có phần tử đầu tiên $I'[0][0] = 0$ để không thực hiện phạt hệ số chặn (intercept), đảm bảo mô hình không bị dịch chuyển không gian gốc tọa độ vô căn cứ.
Khi $\lambda$ tăng:
- Sai số bình phương trung bình trên tập huấn luyện có thể tăng nhẹ.
- Các hệ số hồi quy bị co rút lại (shrinkage) dần về sát 0 nhưng không bao giờ bằng 0 hoàn toàn.
- Mô hình trở nên ổn định hơn trước nhiễu và đa cộng tuyến.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm tra việc co rút của hệ số hồi quy Ridge so với OLS chuẩn trên mẫu dữ liệu giả lập có $50$ mẫu. Trong trường hợp đặc biệt $\lambda = 0$, kết quả của Ridge được đối chứng là trùng khớp hoàn toàn với nghiệm OLS của hàm `ols_fit`.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `ridge_fit` chỉ thực hiện tính toán số và trả về danh sách 2D số thực. Trong phần kiểm thử này, kết quả đầu ra đã được trực quan hóa thêm dưới dạng một **Bảng so sánh hệ số (Coefficient Comparison Table)** giữa Hệ số gốc, Hệ số Ridge và Hệ số Lasso giúp người đọc dễ dàng so sánh mức độ co rút hệ số.
