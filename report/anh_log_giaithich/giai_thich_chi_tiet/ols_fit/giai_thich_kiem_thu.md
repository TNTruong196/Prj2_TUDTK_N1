# Giải thích kiểm thử hàm: `ols_fit`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `ols_fit` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ols_stats.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ols_stats.py), các tham số đầu vào được giả lập như sau:
*   **Ma trận thiết kế `X`**: Kích thước $15 \times 4$ ($n=15$ dòng quan sát, $p+1=4$ cột đặc trưng kể cả cột hệ số chặn chứa toàn số 1 ở cột đầu tiên).
*   **Vector biến mục tiêu `y`**: Kích thước $15 \times 1$ chứa các giá trị thực nghiệm mô phỏng nồng độ khí thải.

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về bộ giá trị bao gồm:
*   **Nghiệm tối ưu `beta` ($\hat{\beta}$)**: Kích thước $4 \times 1$ chứa các hệ số ước lượng tương ứng với từng biến độc lập.
*   **Ước lượng phương sai nhiễu `sigma2` ($\hat{\sigma}^2$)**: Số thực biểu thị mức độ biến động sai số nhiễu ngẫu nhiên của mô hình.

## 3. Ý nghĩa thống kê & toán học
Hàm `ols_fit` thực hiện tìm ước lượng bình phương tối thiểu (Ordinary Least Squares - OLS) cho các hệ số hồi quy theo công thức:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$
Phương sai sai số được ước lượng thông qua tổng bình phương phần dư (RSS) chia cho số bậc tự do hiệu dụng:
$$\hat{\sigma}^2 = \frac{\text{RSS}}{n - (p+1)}$$
Đây là mô hình hồi quy tuyến tính chuẩn làm nền tảng đối chứng cho tất cả các mô hình nâng cao sau này.

## 4. Ghi chú về kiểm thử riêng biệt
Mẫu thử nghiệm sử dụng dữ liệu gồm $15$ quan sát để đảm bảo ma trận đầu ra gọn gàng, dễ chụp ảnh đưa vào báo cáo mà không bị cuộn màn hình. Dữ liệu được cố ý tạo độ tương quan cao giữa các biến để kiểm tra đồng thời khả năng hoạt động của ước lượng OLS khi có đa cộng tuyến.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `ols_fit` chỉ thực hiện tính toán số học và không vẽ đồ thị. Trong phần kiểm thử này, kết quả đầu ra đã được trực quan hóa thêm bằng **Bảng suy diễn hệ số hồi quy** và **Bản đồ nhiệt ma trận Hat** đi kèm để dễ dàng quan sát cấu trúc và nhận diện kết quả một cách trực quan trong báo cáo.
