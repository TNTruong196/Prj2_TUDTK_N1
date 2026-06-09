# Giải thích kiểm thử hàm: `train_and_compare`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `train_and_compare` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_model_comparison.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_model_comparison.py), các tham số đầu vào được cung cấp như sau:
*   **Đường dẫn dữ liệu `data_path`**: Đường dẫn tới tệp dữ liệu thực tế `AirQuality.csv`.
*   **Số lượng phân hoạch `k`**: $k = 5$ (sử dụng 5-Fold CV cho bước chọn lambda tốt nhất).
*   **Mảng tham số phạt `lambdas`**: Một dãy giá trị lambda điều chuẩn để tìm kiếm điểm tối ưu nhanh.
*   **Tham số điều khiển vẽ đồ thị `plot`**: Đặt là `False` để ngăn chặn việc vẽ đồ thị thô có hàm `plt.show()` chặn luồng, thay vào đó ta vẽ đồ thị tùy biến chất lượng cao bằng đoạn mã của kịch bản test.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về một từ điển (`dict`) kết quả phong phú bao gồm:
*   **best_lambda_ridge / best_lambda_lasso**: Các tham số phạt tốt nhất được chọn ra từ đánh giá chéo.
*   **metrics_table**: DataFrame dạng bảng so sánh chỉ số đánh giá (MAE, RMSE, $R^2$) của cả 5 mô hình trên tập kiểm thử (Test Set).
*   **selection_result**: Nhật ký các bước loại bỏ biến (Backward Elimination) của OLS.
*   **models**: Từ điển lưu các mô hình hồi quy đã khớp.
*   **predictions**: Các giá trị dự báo của từng mô hình trên tập Test Set.

## 3. Ý nghĩa thống kê & toán học
Hàm `train_and_compare` đóng vai trò là khối điều phối kiểm nghiệm mô hình tổng thể:
1.  **Chuẩn bị dữ liệu**: Đọc dữ liệu thực tế, chia Train/Test ($80/20$), chạy `DataPipeline` để làm sạch, xử lý ngoại lai và chuẩn hóa.
2.  **Lựa chọn đặc trưng**: Chạy thuật toán Backward Elimination trên tập Train loại bỏ các biến có đa cộng tuyến ($\text{VIF} > 10$) hoặc không có ý nghĩa thống kê ($p\text{-value} > 0.05$).
3.  **Tối ưu hóa tham số**: Chạy 5-Fold CV trên tập Train để chọn lambda tối ưu nhất cho Ridge và Lasso.
4.  **Đánh giá chéo mô hình**: Huấn luyện cả 5 mô hình (OLS thường, OLS chọn biến, Ridge tối ưu, Lasso tối ưu, Bayesian LR) và kiểm nghiệm khả năng tổng quát hóa trên tập dữ liệu Test Set độc lập chưa từng thấy bằng các chỉ số MAE, RMSE, $R^2$.

## 4. Ghi chú về kiểm thử riêng biệt
Mô hình chạy trên toàn bộ tập dữ liệu thực tế Air Quality của Ý với mục tiêu dự báo nồng độ Benzene ($C_6H_6(GT)$). Kết quả cho thấy thuật toán Backward Elimination đã loại bỏ thành công 2 biến `PT08.S2(NMHC)` và `T` do đa cộng tuyến nghiêm trọng. Mô hình Lasso tối ưu đạt hiệu năng cao nhất trên tập kiểm thử.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `train_and_compare` mặc định chỉ hiển thị các đồ thị thô của Matplotlib thông qua việc gọi `plt.show()` trực tiếp gây nghẽn tiến trình chạy tự động. Trong kịch bản kiểm thử này, chúng tôi đã đặt tham số `plot=False`, sau đó sử dụng dữ liệu trả về để vẽ **Biểu đồ so sánh hiệu năng của 5 mô hình** và **Biểu đồ phân tán so sánh Thực tế vs Dự báo** dưới dạng lưới 2x3, căn chỉnh bố cục hợp lý để không bị cắt chữ tiêu đề và lưu thành các tệp ảnh 300 DPI chất lượng cao.
