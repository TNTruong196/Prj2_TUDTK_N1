# Giải thích kiểm thử hàm: `plot_ridge_trace`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `plot_ridge_trace` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ridge_lasso_cv.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ridge_lasso_cv.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $50 \times 6$.
*   **Vector biến mục tiêu `y`**: Kích thước $50 \times 1$.
*   **Mảng tham số phạt `lambdas`**: Một lưới gồm $100$ giá trị tăng dần theo thang log từ $10^{-3}$ đến $10^3$.
*   **Tham số điều khiển hiển thị `show`**: Thiết lập là `False` để cho phép ghi đè nhãn tiếng Việt và lưu tệp hình ảnh tự động.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Danh sách vết hệ số `beta_traces`**: Mảng kích thước $100 \times 6$ chứa thông số của 6 hệ số ứng với 100 bước giá trị lambda.
*   **Đồ thị biểu diễn**: Một đồ thị dòng thể hiện sự thay đổi của các hệ số theo thang $\log_{10}(\lambda)$.

## 3. Ý nghĩa thống kê & toán học
Biểu đồ vết hồi quy Ridge (Ridge Trace) là một công cụ chẩn đoán quan trọng trong hồi quy co rút:
*   Trục hoành biểu thị giá trị $\log_{10}(\lambda)$.
*   Trục tung biểu thị giá trị của các hệ số $\hat{\beta}_j$ ($j=1, \dots, p$).
Biểu đồ cho thấy tốc độ co rút của các hệ số khi tăng cường lực lượng phạt $\lambda$. Điều này giúp các nhà nghiên cứu xác định biến nào có độ ổn định cao, biến nào nhạy cảm với sự thay đổi của hệ số phạt L2.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi tạo lưới $100$ điểm lambda phân bổ đều trên thang logarit để đảm bảo các đường cong vết hệ số được vẽ mềm mại, trơn tru. Tên các đặc trưng được truyền đầy đủ để chú giải biểu đồ (legend) rõ ràng.

## 5. Lưu ý trực quan hóa
*Hàm này vốn dĩ đã có tính năng vẽ đồ thị. Trong kiểm thử này, chúng tôi đã cấu hình bổ sung hệ nhãn nhan đề tiếng Việt hoàn toàn mới và xuất ảnh dưới định dạng 300 DPI sắc nét phục vụ báo cáo.*
