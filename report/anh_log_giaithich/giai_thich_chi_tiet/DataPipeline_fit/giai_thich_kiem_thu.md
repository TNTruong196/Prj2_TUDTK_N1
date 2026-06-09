# Giải thích kiểm thử hàm: `DataPipeline.fit`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `DataPipeline.fit` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_pipeline.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_pipeline.py), các tham số đầu vào được cung cấp như sau:
*   **Tham số khởi tạo**: `missing_threshold=0.5` (tỷ lệ khuyết tối đa cho phép để giữ lại một cột đặc trưng) và `numeric_strategy='median'` (sử dụng trung vị để điền khuyết cho các biến dạng số thực).
*   **DataFrame thô huấn luyện `df_raw`**: Kích thước $100 \times 6$ có chứa các cột `Date` và `Time` cần bóc tách chu kỳ thời gian, một cột phân loại `Location_Type`, cột `Feature_B_Excessive_Missing` chứa $60\%$ dữ liệu rỗng và cột `Feature_A` chứa các giá trị ngoại lai lớn.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Đối tượng pipeline tự thân (`self`)**: Đã lưu trữ toàn bộ các thuộc tính tham số học được từ tập dữ liệu huấn luyện để sẵn sàng cho bước biến đổi dữ liệu tiếp theo.

## 3. Ý nghĩa thống kê & toán học
Hàm `fit` chịu trách nhiệm học các tham số tiền xử lý từ dữ liệu huấn luyện một cách độc lập nhằm ngăn chặn hiện tượng rò rỉ dữ liệu (data leakage) sang tập kiểm thử. Quy trình học bao gồm:
1.  **Xác định cột cần loại bỏ**:
    $$\text{Tỷ lệ khuyết} = \frac{\text{Số dòng khuyết}}{n}$$
    Nếu tỷ lệ khuyết $> \text{missing\_threshold}$, tên cột được thêm vào danh sách loại bỏ (`cols_to_drop`).
2.  **Học giá trị điền khuyết (`impute_values`)**:
    - Đối với biến dạng số thực chu kỳ hoặc khi có ngoại lai, sử dụng Trung vị (Median) vì tính ổn định cao.
    - Đối với biến phân loại, sử dụng Yếu vị (Mode).
3.  **Học ngưỡng ngoại lai (`outlier_bounds`)**: Tính toán biên IQR cho mỗi biến số thực:
    $$IQR = Q_3 - Q_1$$
    $$\text{Ngưỡng dưới} = Q_1 - 1.5 \times IQR, \quad \text{Ngưỡng trên} = Q_3 + 1.5 \times IQR$$
4.  **Học thông số Z-score**: Tính giá trị trung bình mẫu $\mu_j$ và độ lệch chuẩn mẫu $s_j$ của từng biến đã được mã hóa để chuẩn bị cho bước scaling.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm tra tính đúng đắn của việc học tham số trên một tập dữ liệu giả lập có chủ đích có lỗi khuyết thiếu cực đại ($60\%$) và ngoại lai giá trị cực lớn để xác thực rằng Pipeline học đúng các giá trị điền khuyết và biên capping.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `DataPipeline.fit` chỉ thực hiện tính toán và cập nhật các thuộc tính lưu trữ nội bộ của đối tượng và không có bất kỳ biểu đồ nào. Trong kiểm thử này, chúng tôi đã trực quan hóa kết quả học được dưới dạng **Bảng tham số học được (Learned Pipeline Parameters)** in ra màn hình console để chẩn đoán.
