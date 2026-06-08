# Giải thích kiểm thử hàm: `bayesian_linear_fit`

Tài liệu này cung cấp chi tiết về việc chạy thử nghiệm và chẩn đoán cho hàm `bayesian_linear_fit` phục vụ báo cáo.

## 1. Thông số đầu vào (Input)
Trong kịch bản kiểm thử tại [test_bayesian_lr.py](file:///c:/TranNhatTruong_2026/HK1/Toan%20Ung%20Dung/Prj2/main/TEST_TO_REPORT/test_bayesian_lr.py), các tham số đầu vào được cung cấp như sau:
*   **Ma trận thiết kế `X`**: Kích thước $25 \times 3$.
*   **Vector biến mục tiêu `y`**: Kích thước $25 \times 1$.
*   **Phương sai sai số nhiễu `sigma2`**: Ước lượng phương sai từ OLS ($\sigma^2_{\text{OLS}}$).
*   **Tham số Prior**: Kỳ vọng tiên nghiệm `prior_mean` (mặc định là vector 0), phương sai tiên nghiệm `prior_variance` ($\sigma^2_0$, ví dụ: $1e10$ cho prior yếu, $1.0$ cho prior chuẩn, $0.01$ cho prior mạnh), và phương sai cho hệ số chặn `intercept_prior_variance` (thiết lập là $1e12$ để tránh điều chuẩn hệ số chặn).

## 2. Đầu ra thực tế thu được (Output)
Hàm trả về một từ điển (`dict`) biểu thị phân phối hậu nghiệm (Posterior Distribution) của các hệ số hồi quy:
*   **posterior_mean ($m_n$)**: Vector kỳ vọng hậu nghiệm (kích thước $3 \times 1$).
*   **posterior_covariance ($S_n$)**: Ma trận hiệp phương sai hậu nghiệm (kích thước $3 \times 3$).

## 3. Ý nghĩa thống kê & toán học
Hồi quy tuyến tính Bayes (Bayesian Linear Regression) xem các hệ số hồi quy $\beta$ là các biến ngẫu nhiên có phân phối xác suất. Ta thiết lập một phân phối tiên nghiệm Gaussian độc lập cho các hệ số:
$$p(\beta) = \mathcal{N}(\beta; m_0, S_0) \quad \text{với} \quad m_0 = 0, \quad S_0 = \text{diag}(\sigma^2_{\text{intercept}}, \sigma^2_0, \dots, \sigma^2_0)$$
Sau khi quan sát dữ liệu thực nghiệm thông qua hàm hợp lý (likelihood), ta cập nhật phân phối hậu nghiệm theo định lý Bayes. Do tính liên hợp Gaussian, phân phối hậu nghiệm cũng là phân phối chuẩn $p(\beta | y) = \mathcal{N}(\beta; m_n, S_n)$ với các tham số cập nhật:
$$S_n = \left( S_0^{-1} + \frac{1}{\sigma^2} X^T X \right)^{-1}$$
$$m_n = S_n \left( S_0^{-1} m_0 + \frac{1}{\sigma^2} X^T y \right)$$
*   **Prior yếu** ($\sigma^2_0 \to \infty$): Kỳ vọng hậu nghiệm tiến sát về nghiệm OLS.
*   **Prior mạnh** ($\sigma^2_0 \to 0$): Kỳ vọng hậu nghiệm bị kéo mạnh về phía kỳ vọng tiên nghiệm (vector 0).

## 4. Ghi chú về kiểm thử riêng biệt
Chúng tôi kiểm thử ước lượng Bayes dưới 3 chế độ tiên nghiệm khác nhau trên mẫu $25$ quan sát. Kết quả phản ánh rõ nét hiệu ứng co rút của mô hình Bayes: phương sai prior càng nhỏ, ước lượng điểm của hệ số càng co rút mạnh về 0.

## 5. Lưu ý trực quan hóa
> **LƯU Ý:** Hàm gốc `bayesian_linear_fit` chỉ thực hiện các phép nhân ma trận và tính toán số học để trả về phân phối hậu nghiệm thô. Trong kịch bản kiểm thử này, chúng tôi đã trực quan hóa ma trận hiệp phương sai hậu nghiệm $S_n$ bằng **Bản đồ nhiệt Posterior Covariance Matrix Heatmap** để chẩn đoán độ biến động của các ước lượng tham số.
