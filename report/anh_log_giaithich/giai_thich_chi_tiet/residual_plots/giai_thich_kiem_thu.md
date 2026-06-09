# Giải thích kiểm thử hàm: `residual_plots`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `residual_plots` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ridge_lasso_cv.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ridge_lasso_cv.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $50 \times 6$.
*   **Vector biến mục tiêu `y`**: Kích thước $50 \times 1$.
*   **Vector hệ số OLS `beta_ols`**: Kích thước $6 \times 1$.
*   **Tham số điều khiển hiển thị `show`**: Thiết lập là `False` để tự động lưu ảnh và viết đè nhãn tiếng Việt.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Hình vẽ matplotlib `fig`** và các trục tọa độ **`axes`** của lưới 4 đồ thị con chẩn đoán phần dư.
*   Một tệp hình ảnh đồ thị chẩn đoán hoàn chỉnh được lưu trên đĩa.

## 3. Ý nghĩa thống kê & toán học
Các biểu đồ chẩn đoán phần dư kiểm tra tính hợp lệ của các giả định trong mô hình hồi quy tuyến tính cổ điển:
1.  **Residuals vs Fitted (Phần dư vs Giá trị khớp)**: Kiểm tra giả định về tính tuyến tính của mối liên hệ. Các điểm phần dư nên phân bổ ngẫu nhiên xung quanh trục hoành $y=0$ và không tạo ra bất kỳ mẫu hình học (như hình parabol) nào.
2.  **Normal Q-Q (Biểu đồ phân vị - phân vị chuẩn)**: Kiểm tra giả định phần dư có phân phối chuẩn. Nếu giả định đúng, các điểm phần dư thực nghiệm sẽ bám sát đường thẳng chéo lý thuyết màu đỏ.
3.  **Scale-Location (Phần dư chuẩn hóa vs Giá trị khớp)**: Kiểm tra giả định về phương sai đồng đều của phần dư. Các điểm phân bổ đều theo dải nằm ngang cho biết phương sai đồng đều (homoscedasticity).
4.  **Residuals vs Leverage / Cook's Distance**: Phát hiện các điểm dữ liệu dị biệt có sức ảnh hưởng quá lớn tới đường hồi quy. Các điểm vượt qua ngưỡng cảnh báo $4/n$ cần được xem xét loại bỏ.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi áp dụng hàm này để đánh giá phần dư thu được từ ước lượng mô hình OLS cơ bản trên 50 mẫu dữ liệu giả lập. Đồ thị chẩn đoán cho thấy các giả thuyết cơ bản của mô hình hồi quy tuyến tính đều được thỏa mãn tốt.

## 5. Lưu ý trực quan hóa
*Hàm này vốn dĩ đã có tính năng vẽ đồ thị. Trong kiểm thử này, chúng tôi đã can thiệp vào đối tượng trục `axes` sau khi hàm trả về để dịch toàn bộ nhãn đồ thị, chú giải và tiêu đề của cả 4 đồ thị con sang tiếng Việt, nâng cao chất lượng trực quan cho báo cáo.*
