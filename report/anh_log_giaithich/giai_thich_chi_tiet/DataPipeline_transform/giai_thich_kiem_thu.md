# Giải thích kiểm thử hàm: `DataPipeline.transform`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `DataPipeline.transform` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_pipeline.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_pipeline.py), các tham số đầu vào được cung cấp như sau:
*   **DataFrame thô cần biến đổi `X`**: Tập dữ liệu thô (có thể là tập huấn luyện hoặc tập kiểm định).
*   Các tham số tiền xử lý đã được học từ bước huấn luyện trước đó thông qua phương thức `fit`.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **DataFrame đã xử lý hoàn chỉnh**: Dữ liệu đầu ra sạch sẽ, không còn giá trị rỗng, các ngoại lai đã được giới hạn biên, các đặc trưng phân loại được mã hóa One-Hot, các cột chu kỳ thời gian được tạo lập và toàn bộ dữ liệu đã được chuẩn hóa Z-score với trung bình bằng 0 và độ lệch chuẩn bằng 1.

## 3. Ý nghĩa thống kê & toán học
Quy trình biến đổi dữ liệu thực thi tuần tự các bước:
1.  **Trích xuất chu kỳ thời gian**: Date chuyển đổi thành tháng (`Month`) và thứ trong tuần (`DayOfWeek`). Time chuyển đổi thành giờ trong ngày (`Hour`). Loại bỏ các cột văn bản gốc.
2.  **Loại bỏ đặc trưng rỗng**: Xóa các cột đã được xác định loại bỏ trong danh sách `cols_to_drop`.
3.  **Điền khuyết (Imputation)**: Thay thế toàn bộ giá trị khuyết (NaN) bằng các giá trị đại diện tương ứng đã học:
    $$x_{ij} \leftarrow \text{impute\_value}_j \quad \text{nếu} \quad x_{ij} \text{ bị khuyết}$$
4.  **Giới hạn ngoại lai (Winsorization)**: Cắt các giá trị ngoại lai về khoảng biên an toàn IQR:
    $$x_{ij} \leftarrow \text{clip}(x_{ij}, \text{lower}_j, \text{upper}_j)$$
5.  **Mã hóa One-Hot**: Chuyển đổi các cột phân loại dạng chuỗi ký tự thành các cột nhị phân $0/1$.
6.  **Chuẩn hóa Z-score**: Đưa dữ liệu về cùng một thang đo có trung bình bằng 0 và độ lệch chuẩn bằng 1 để cải thiện tính ổn định của các thuật toán tối ưu số học:
    $$z_{ij} = \frac{x_{ij} - \mu_j}{s_j}$$

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm thử quá trình biến đổi trên cùng tập dữ liệu thô. Kết quả kiểm tra thống kê xác nhận số lượng giá trị khuyết (NaN) giảm về chính xác bằng 0, giá trị trung bình sau chuẩn hóa của các cột số thực đều xấp xỉ bằng $0.0$, và độ lệch chuẩn bằng đúng $1.0$.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `DataPipeline.transform` chỉ trả về DataFrame số liệu và không có tính năng vẽ đồ thị. Trong kiểm thử này, chúng tôi đã trực quan hóa hiệu quả biến đổi bằng **Biểu đồ cột so sánh tỷ lệ khuyết thiếu** và **Biểu đồ hộp so sánh phân phối ngoại lai Winsorization** trước và sau khi đi qua pipeline.
