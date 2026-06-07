# Giải thích kiểm thử hàm: `vif`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `vif` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_ols_stats.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_ols_stats.py), tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $15 \times 4$ ($n=15$ dòng quan sát, $p+1=4$ cột đặc trưng kể cả cột hệ số chặn chứa toàn số 1 ở cột đầu tiên).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Danh sách VIF**: Mảng chứa $p=3$ giá trị số thực biểu thị hệ số VIF của từng biến độc lập (không bao gồm hệ số chặn).

## 3. Ý nghĩa thống kê & toán học
Hệ số phóng đại phương sai (Variance Inflation Factor - VIF) đo lường mức độ ảnh hưởng của đa cộng tuyến tới phương sai của các ước lượng hệ số OLS.
Đối với mỗi biến độc lập $x_j$, ta xây dựng một hồi quy phụ hồi quy $x_j$ theo tất cả các biến độc lập còn lại và tính hệ số xác định $R^2_j$. Hệ số VIF của biến $x_j$ được xác định bởi:
$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$
*   $\text{VIF} = 1$: Các biến độc lập hoàn toàn không tương quan tuyến tính.
*   $\text{VIF} > 5$: Dấu hiệu nghi ngờ có đa cộng tuyến vừa phải.
*   $\text{VIF} > 10$: Đa cộng tuyến nghiêm trọng. Sai số chuẩn của các hệ số bị thổi phồng dẫn tới kết luận suy diễn không chính xác (các biến mất đi ý nghĩa thống kê thực tế).

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi tạo lập đặc trưng $x_3$ phụ thuộc tuyến tính trực tiếp vào $x_1$ và $x_2$ cộng thêm nhiễu siêu nhỏ:
$$x_3 = 2x_1 + x_2 + e \quad (e \sim N(0, 0.01^2))$$
Kịch bản này tạo ra hệ số đa cộng tuyến rất lớn ($\text{VIF} > 300$). Kết quả kiểm thử phản ánh chính xác tình trạng này trong phần chẩn đoán.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `vif` chỉ trả về danh sách các giá trị số thực thô. Trong kịch bản kiểm thử này, chúng tôi đã trực quan hóa dữ liệu VIF thành một **Biểu đồ cột VIF Bar Chart** có phân màu cảnh báo mức độ đa cộng tuyến (Đỏ: Nghiêm trọng, Cam: Vừa phải, Xanh: An toàn) giúp người đọc dễ quan sát sự phóng đại sai số trong báo cáo thực hành.
