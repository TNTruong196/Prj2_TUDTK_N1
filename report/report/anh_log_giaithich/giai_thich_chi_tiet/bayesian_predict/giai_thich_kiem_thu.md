# Giải thích kiểm thử hàm: `bayesian_predict`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `bayesian_predict` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_bayesian_lr.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_bayesian_lr.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $25 \times 3$ chứa dữ liệu các điểm cần dự báo.
*   **Từ điển phân phối hậu nghiệm `posterior`**: Được trả về từ bước khớp mô hình `bayesian_linear_fit`, chứa kỳ vọng hậu nghiệm `posterior_mean`.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Vector giá trị dự đoán `y_pred`**: Kích thước $25 \times 1$ dạng danh sách 2D.

## 3. Ý nghĩa thống kê & toán học
Trong hồi quy tuyến tính Bayes, thay vì chọn một nghiệm điểm $\hat{\beta}$ duy nhất để dự báo, kỳ vọng toán học của phân phối dự báo (Predictive Distribution) của một điểm dữ liệu mới $x_*$ được xác định bằng cách lấy tích phân trên toàn bộ không gian tham số:
$$E[y_* | x_*, y] = \int (x_*^T \beta) p(\beta | y) d\beta = x_*^T E[\beta | y] = x_*^T m_n$$
Do đó, hàm thực hiện phép nhân ma trận đơn giản giữa ma trận thiết kế của các điểm mới và kỳ vọng hậu nghiệm $m_n$ của các hệ số:
$$\hat{y} = X m_n$$
Đây là điểm dự báo tối ưu trung bình giảm thiểu hàm tổn thất bình phương.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm tra xem giá trị dự đoán của hàm có trùng khớp với kết quả nhân ma trận trực tiếp $X \times m_n$ hay không. Kịch bản chạy thực tế cho thấy kết quả trùng khớp hoàn toàn.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `bayesian_predict` chỉ thực hiện phép nhân ma trận và trả về danh sách dự đoán. Trong kịch bản kiểm thử tích hợp (xem nhóm so sánh mô hình), các giá trị dự báo này đã được trực quan hóa trên **Biểu đồ phân tán Thực tế vs Dự báo (Actual vs Predicted Scatter Plot)** để chẩn đoán độ chính xác dự báo của mô hình Bayes so với OLS.
