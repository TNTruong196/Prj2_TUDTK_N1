# Giải thích kiểm thử hàm: `credible_intervals`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `credible_intervals` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_bayesian_lr.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_bayesian_lr.py), các tham số đầu vào được cung cấp như sau:
*   **Từ điển phân phối hậu nghiệm `posterior`**: Chứa kỳ vọng hậu nghiệm `posterior_mean` và ma trận hiệp phương sai hậu nghiệm `posterior_covariance`.
*   **Mức tin cậy `level`**: Mặc định là $0.95$ (khoảng tin cậy Bayesian 95%).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về:
*   **Danh sách các khoảng tin cậy `intervals`**: Mảng chứa các cặp giá trị biên dưới và biên trên `(lower, upper)` tương ứng với từng hệ số hồi quy $\beta_j$.

## 3. Ý nghĩa thống kê & toán học
Khoảng tin cậy Bayesian (Credible Interval) biểu thị khoảng giá trị mà tham số có xác suất rơi vào đó bằng đúng mức tin cậy $1-\alpha$ (ví dụ: xác suất thực sự là 95%), khác biệt với khoảng tin cậy tần suất luận (Confidence Interval) của OLS vốn chỉ đại diện cho tần suất lặp mẫu.
Do phân phối hậu nghiệm của các hệ số có phân phối chuẩn, khoảng tin cậy Bayesian 95% cho hệ số $\beta_j$ được tính trực tiếp từ kỳ vọng và phương sai hậu nghiệm:
$$\text{CI}_j = m_{n,j} \pm z_{1-\alpha/2} \times \sqrt{S_{n,jj}}$$
Trong đó:
*   $m_{n,j}$ là kỳ vọng hậu nghiệm của hệ số thứ $j$.
*   $S_{n,jj}$ là phần tử đường chéo của ma trận hiệp phương sai posterior đại diện cho phương sai hậu nghiệm của hệ số thứ $j$.
*   $z_{1-\alpha/2}$ là phân vị chuẩn (ví dụ: $1.96$ cho mức $95\%$).

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm thử khoảng tin cậy trên mô hình Bayes prior chuẩn. Kết quả cho thấy khoảng tin cậy bao phủ chính xác giá trị hệ số thực tế (True Coefficients), phản ánh độ tin cậy toán học cao của mô hình.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `credible_intervals` chỉ trả về mảng số thực biểu thị khoảng tin cậy. Trong kịch bản kiểm thử này, chúng tôi đã trực quan hóa dữ liệu này thành **Biểu đồ khoảng tin cậy Bayesian (95% Credible Intervals Plot)** với thanh sai số (error bars) màu tím và đánh dấu hệ số gốc bằng ngôi sao đỏ để minh họa tính chính xác của ước lượng trong báo cáo.
