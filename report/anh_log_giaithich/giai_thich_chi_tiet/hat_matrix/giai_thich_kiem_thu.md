# Giải thích kiểm thử hàm: `hat_matrix`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `hat_matrix` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ols_stats.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ols_stats.py), tham số đầu vào được giả lập như sau:
*   **Ma trận thiết kế `X`**: Kích thước $15 \times 4$ ($n=15$ dòng quan sát, $p+1=4$ cột đặc trưng kể cả cột hệ số chặn chứa toàn số 1 ở cột đầu tiên).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Ma trận chiếu chéo `H` (Hat Matrix)**: Kích thước $15 \times 15$ ($n \times n$). Các phần tử trên đường chéo chính $h_{ii}$ biểu thị giá trị tựa (leverage) của từng quan sát thực tế.

## 3. Ý nghĩa thống kê & toán học
Ma trận hình mũ $H$ ánh xạ trực tiếp từ vector giá trị quan sát thực tế $y$ sang vector giá trị dự báo $\hat{y}$ thông qua biểu thức:
$$\hat{y} = H y \quad \text{với} \quad H = X(X^T X)^{-1} X^T$$
Đặc tính toán học quan trọng của ma trận Hat bao gồm:
1.  **Tính đối xứng**: $H = H^T$.
2.  **Tính lũy đẳng (Idempotent)**: $H \times H = H$.
3.  **Vết (Trace)**: $\text{tr}(H) = \sum h_{ii} = p + 1$ (số lượng hệ số trong mô hình).
Giá trị $h_{ii}$ nằm trong khoảng $[1/n, 1]$ và đo lường khoảng cách từ điểm quan sát thứ $i$ tới trung tâm của các điểm quan sát khác trong không gian biến độc lập.

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi sử dụng một ma trận thiết kế $15 \times 4$ để tạo ra ma trận Hat $15 \times 15$. Kích thước này đủ nhỏ để biểu diễn trọn vẹn toàn bộ ma trận dưới dạng bản đồ nhiệt và kiểm tra trực quan các tính chất toán học của ma trận hình mũ.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `hat_matrix` chỉ trả về một danh sách 2D số thực và không vẽ đồ thị. Trong kịch bản kiểm thử này, chúng tôi đã trực quan hóa ma trận chéo $15 \times 15$ thành một **Bản đồ nhiệt Hat Matrix Heatmap** với hệ màu xanh dương để người đọc dễ dàng nhận diện các điểm quan sát có giá trị leverage cao nguy cơ ảnh hưởng lớn tới mô hình hồi quy.
