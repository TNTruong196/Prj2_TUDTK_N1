# Báo cáo công việc Giai đoạn 3 - Người 4: Tân

---

## 1. Cơ sở lý thuyết

### 1.1. Ridge Regression

OLS tìm vector hệ số bằng cách tối thiểu hóa tổng bình phương phần dư:

$$
\hat{\beta}_{OLS} = (X^T X)^{-1}X^T y
$$

Khi dữ liệu có đa cộng tuyến hoặc các cột của ma trận thiết kế có tương quan cao, ma trận \(X^T X\) có thể gần suy biến. Điều này làm hệ số OLS dao động mạnh và kém ổn định. Ridge Regression xử lý vấn đề này bằng cách thêm thành phần phạt L2:

$$
\hat{\beta}_{ridge}
= \arg\min_{\beta}\left\{ \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2 \right\}
$$

Nghiệm đóng của Ridge:

$$
\hat{\beta}_{ridge} = (X^T X + \lambda I')^{-1}X^T y
$$

Trong phần cài đặt, \(I'\) là ma trận đơn vị nhưng phần tử đầu tiên \(I'_{00}=0\). Cách làm này giúp không regularize hệ số chặn \(\beta_0\), vì intercept chỉ đóng vai trò dịch mức dự đoán trung bình, không phải độ dốc của một đặc trưng.

Các tính chất quan trọng:

- Khi \(\lambda = 0\), Ridge trở về OLS.
- Khi \(\lambda\) tăng, các hệ số của feature bị co dần về 0.
- Ridge giúp ổn định hệ số khi các biến đầu vào có tương quan mạnh.
- Ridge thường đánh đổi một lượng bias nhỏ để giảm variance của mô hình.

### 1.2. Ridge Trace

Ridge Trace là biểu đồ thể hiện sự thay đổi của từng hệ số \(\beta_j\) theo \(\log_{10}(\lambda)\). Biểu đồ này giúp quan sát trực quan mức độ co hệ số khi lambda tăng.

Ý nghĩa:

- Feature có hệ số lớn và ổn định qua nhiều giá trị lambda thường có ảnh hưởng mạnh hơn.
- Khi lambda quá lớn, các hệ số bị co mạnh, mô hình có nguy cơ underfit.
- Ridge Trace hỗ trợ giải thích tác động của regularization trong báo cáo.

### 1.3. k-Fold Cross-Validation

k-fold CV chia dữ liệu train thành \(k\) phần gần bằng nhau. Mỗi lần dùng \(k-1\) fold để huấn luyện và 1 fold để validation. Điểm CV-MSE được tính:

$$
CV_{(k)} = \frac{1}{k}\sum_{i=1}^{k} MSE_i
$$

Trong phần cài đặt, `kfold_cv(X, y, k, fit_func=None)` cho phép truyền hàm fit bất kỳ. Nhờ đó cùng một hàm CV có thể dùng cho OLS hoặc Ridge. Khi chọn lambda cho Ridge, mỗi giá trị lambda được bọc vào một `fit_func`, sau đó tính CV-MSE và chọn lambda có CV-MSE nhỏ nhất.

### 1.4. Các metric đánh giá mô hình

Mô hình được so sánh trên test set bằng ba chỉ số:

$$
MAE = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
$$

$$
RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}
$$

$$
R^2 = 1 - \frac{RSS}{TSS}
$$

MAE dễ diễn giải theo đơn vị gốc của target. RMSE phạt lỗi lớn mạnh hơn MAE. R2 cho biết tỉ lệ biến thiên của target được mô hình giải thích.

### 1.5. Bayesian Linear Regression

Bayesian Linear Regression được chọn làm kỹ thuật nâng cao cho phần bonus. Mô hình vẫn giữ dạng tuyến tính:

$$
y = X\beta + \epsilon,\quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

Khác với OLS frequentist, Bayesian Linear Regression xem \(\beta\) là biến ngẫu nhiên và đặt prior Gaussian:

$$
\beta \sim \mathcal{N}(m_0, S_0)
$$

Sau khi quan sát dữ liệu, posterior của \(\beta\) là:

$$
\beta|X,y \sim \mathcal{N}(m_n, S_n)
$$

với:

$$
S_n = \left(S_0^{-1} + \frac{1}{\sigma^2}X^T X\right)^{-1}
$$

$$
m_n = S_n\left(S_0^{-1}m_0 + \frac{1}{\sigma^2}X^T y\right)
$$

Trong triển khai hiện tại:

- `prior_mean` mặc định là vector 0.
- `prior_variance=100.0` cho các hệ số feature, tương ứng prior tương đối yếu.
- `intercept_prior_variance=1e12` để gần như không regularize intercept.
- `sigma2` dùng ước lượng từ OLS trên tập train.
- Dự đoán dùng posterior mean: \(\hat{y}=X m_n\).

Bayesian Linear Regression cũng cung cấp posterior covariance, từ đó có thể tính credible interval cho từng hệ số. Đây là điểm bổ sung so với OLS/Ridge: ngoài dự đoán, mô hình còn biểu diễn độ bất định của hệ số.

---

## 2. Lý do chọn Bayesian Linear Regression thay vì Kernel Regression

Trong `phase3_plan.md`, phần bonus ban đầu gợi ý Kernel Ridge Regression vì đây là mở rộng từ Ridge. Tuy nhiên sau khi kiểm tra dataset và baseline, Bayesian Linear Regression được chọn vì các lý do sau:

### 2.1. Phù hợp hơn với trọng tâm OLS của đề bài

Đồ án tập trung vào Data Fitting, OLS, Ridge/Lasso, residual analysis và so sánh mô hình hồi quy. Bayesian Linear Regression mở rộng trực tiếp từ mô hình tuyến tính:

```text
OLS -> Ridge -> Bayesian Linear Regression
```

Mạch này dễ trình bày trong báo cáo hơn Kernel Regression, vì vẫn giữ hệ số tuyến tính và vẫn liên hệ chặt với công thức \(X^T X\), \(X^T y\).

### 2.2. Phù hợp với tính chất dataset AirQuality

Dataset AirQuality có target `C6H6(GT)` và các biến cảm biến môi trường. Khi khảo sát nhanh, target có tương quan rất mạnh với nhiều cảm biến:

| Feature | Tương quan với `C6H6(GT)` |
|---|---:|
| `PT08.S2(NMHC)` | khoảng 0.982 |
| `CO(GT)` | khoảng 0.931 |
| `NMHC(GT)` | khoảng 0.903 |
| `PT08.S1(CO)` | khoảng 0.884 |
| `PT08.S5(O3)` | khoảng 0.866 |

Các kết quả này cho thấy quan hệ tuyến tính hoặc gần tuyến tính đã rất mạnh. Trên baseline hiện tại, OLS và Ridge đều đạt R2 test khoảng 0.964, nên lợi ích kỳ vọng từ Kernel Regression không lớn trong khi chi phí triển khai và tính toán cao hơn.

### 2.3. Tránh chi phí ma trận Gram quá lớn

Kernel Regression cần xây dựng và nghịch đảo ma trận Gram kích thước \(n \times n\), với \(n\) là số mẫu train. AirQuality có hơn 9000 dòng, sau khi chia train/test vẫn còn vài nghìn dòng train. Ma trận Gram vì vậy rất lớn.

Trong khi đó, Bayesian Linear Regression chỉ cần nghịch đảo ma trận kích thước \((p+1)\times(p+1)\), với \(p\) là số feature sau tiền xử lý. Điều này phù hợp hơn với helper `mat_inverse` tự cài bằng Python 2D list.

### 2.4. Dễ giải thích feature importance

Đề phần 2 yêu cầu feature importance bằng hệ số hồi quy. Bayesian Linear Regression vẫn cho hệ số theo từng feature thông qua posterior mean, nên có thể dùng cùng hướng giải thích với OLS/Ridge. Kernel Regression không có hệ số tuyến tính trực tiếp, nếu muốn giải thích feature importance phải dùng phương pháp phụ như permutation importance.

### 2.5. Ít rủi ro hơn cho phần bonus

Phần kỹ thuật nâng cao chỉ là bonus +0.5 điểm. Vì vậy lựa chọn hợp lý là phương pháp:

- Hoàn thành được chắc chắn.
- Dễ kiểm thử.
- Dễ tích hợp vào bảng so sánh.
- Dễ giải thích khi vấn đáp.
- Không làm ảnh hưởng các phần bắt buộc.

Bayesian Linear Regression đáp ứng tốt các tiêu chí này hơn Kernel Regression trong phạm vi codebase hiện tại.

---

## 3. Các phần đã triển khai

### 3.1. `part1/ridge_lasso.py`

Các hàm chính đã triển khai:

- `ridge_fit(X, y, lam)`: fit Ridge Regression bằng công thức đóng.
- `plot_ridge_trace(X, y, lambdas=None, show=True, feature_names=None, max_features=None)`: vẽ Ridge Trace.
- `ridge_trace(X, y, lam=None)`: wrapper gọi `plot_ridge_trace`.

Chi tiết kỹ thuật:

- Input `X`, `y` là 2D list.
- `X` đã có cột intercept.
- Tạo ma trận \(I'\), trong đó `I[0][0] = 0` để không regularize intercept.
- Tính:

```text
XT = X^T
XTX = X^T X
ridge_matrix = XTX + lambda * I'
beta = ridge_matrix^-1 X^T y
```

Unit tests đã có:

- `test_ridge_fit_lamda0`: kiểm tra lambda = 0 cho kết quả gần OLS.
- `test_ridge_fit_against_sklearn`: kiểm chứng với `sklearn.linear_model.Ridge`.
- `test_ridge_shrinkage`: lambda lớn làm hệ số feature bị shrink.
- `test_plot_ridge_trace_runs`: Ridge Trace chạy được và trả về đúng số lượng beta trace.

### 3.2. `part1/cross_validation.py`

Các hàm chính đã triển khai:

- `ols_beta_fit(X_train, y_train)`: wrapper lấy beta từ `ols_fit`.
- `kfold_cv(X, y, k, fit_func=None)`: k-fold CV trả về `cv_mse` và `fold_mses`.

Chi tiết kỹ thuật:

- Dùng `SEED = 42`.
- Shuffle index bằng `random.seed(SEED)` và `random.shuffle(indices)`.
- Chia fold gần bằng nhau, phân phần dư vào các fold đầu.
- Mỗi vòng:
  - Tách train/validation theo chỉ số.
  - Fit mô hình bằng `fit_func`.
  - Dự đoán bằng `mat_mul(X_val, beta)`.
  - Tính MSE trên validation fold.
- Trả về:

```python
(cv_mse, fold_mses)
```

Unit tests đã có:

- `test_kfold_returns_correct_structure`: kiểm tra kiểu dữ liệu và số fold.
- `test_kfold_reproducible`: chạy hai lần cho cùng kết quả với seed 42.
- `test_kfold_with_custom_fit_func`: kiểm tra dùng được với custom fit function cho Ridge.

### 3.3. `part2/model_comparison.py`

File này dùng để áp dụng mô hình lên dữ liệu AirQuality đã qua pipeline.

Các hàm chính:

- `default_lambdas()`: tạo grid lambda từ \(10^{-4}\) đến \(10^4\).
- `df_to_2d_list(data)`: chuyển DataFrame/Series/list sang 2D list.
- `add_intercept(X)`: thêm cột intercept.
- `predict(X, beta)`: dự đoán \(X\beta\).
- `compute_metrics(y_true, y_pred)`: tính MAE, RMSE, R2.
- `select_best_lambda(X, y, k=5, lambdas=None)`: chọn lambda tốt nhất cho Ridge bằng k-fold CV.
- `plot_cv_results(cv_results, show=True)`: vẽ CV-MSE theo log10(lambda).
- `ridge_coefficients_table(beta, feature_names)`: tạo bảng hệ số.
- `ridge_trace_table(...)`: tạo bảng ridge trace dạng đọc được.
- `prepare_air_quality_data(...)`: load data, chia train/test, fit-transform pipeline, chuyển sang 2D list.
- `train_and_compare(...)`: train OLS, Ridge và Bayesian Linear Regression, sau đó đánh giá trên test set.

Luồng xử lý dữ liệu:

1. Đọc `part2/data/AirQuality.csv`.
2. Bỏ các dòng thiếu target `C6H6(GT)`.
3. Tách `X`, `y`.
4. Chia train/test bằng `train_test_split(..., random_state=42)`.
5. Fit `DataPipeline` trên train.
6. Transform train và test.
7. Chuyển DataFrame/Series sang 2D list.
8. Thêm cột intercept.
9. Train và đánh giá mô hình.

Các mô hình hiện được so sánh:

- OLS cơ bản.
- Ridge với lambda tối ưu từ k-fold CV.
- Bayesian Linear Regression.

### 3.4. `part2/advanced_methods.py`

File này triển khai phần nâng cao bằng Bayesian Linear Regression.

Các hàm chính:

- `bayesian_linear_fit(...)`: fit Bayesian Linear Regression bằng posterior mean/covariance.
- `bayesian_predict(X, posterior)`: dự đoán bằng posterior mean.
- `credible_intervals(posterior, level=0.95)`: tính credible interval xấp xỉ cho từng hệ số.

Chi tiết kỹ thuật:

- Thuật toán chính dùng 2D list và helper ma trận từ `part1/matrix_helper.py`.
- Không dùng NumPy/sklearn trong phần fit/predict chính.
- Nếu `sigma2=None`, hàm tự ước lượng \(\sigma^2\) bằng `ols_fit`.
- Intercept được đặt prior variance rất lớn để gần như không bị regularize.

Unit tests đã có:

- `test_bayesian_weak_prior_close_to_ols`: prior yếu cho posterior mean gần OLS.
- `test_bayesian_strong_prior_shrinks_feature_coef`: prior mạnh làm hệ số feature bị shrink.
- `test_bayesian_predict_shape`: dự đoán trả về vector cột đúng kích thước.

---

## 4. Kết quả chạy thực nghiệm hiện tại

Khi chạy:

```bash
python part2\advanced_methods.py
```

Kết quả:

```text
TEST 1 PASSED: weak prior cho posterior mean gan OLS
TEST 2 PASSED: prior manh lam he so feature bi shrink
TEST 3 PASSED: bayesian_predict tra ve vector cot dung kich thuoc
```

Khi chạy:

```bash
python part2\model_comparison.py
```

Kết quả chọn lambda:

```text
Best lambda: 0.1
```

Top lambda theo CV-MSE:

| lambda | CV-MSE |
|---:|---:|
| 0.100000 | 1.993500 |
| 0.046416 | 1.993500 |
| 0.021544 | 1.993500 |
| 0.010000 | 1.993500 |
| 0.004642 | 1.993501 |

Bảng đánh giá trên test set:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| OLS | 0.889981 | 1.446089 | 0.963608 |
| Ridge (lambda=0.1) | 0.889957 | 1.446081 | 0.963608 |
| Bayesian Linear Regression | 0.889976 | 1.446087 | 0.963608 |

Nhận xét:

- Ridge cải thiện rất nhẹ so với OLS.
- Bayesian Linear Regression cho kết quả gần OLS/Ridge vì prior đang tương đối yếu và dataset có quan hệ tuyến tính mạnh.
- R2 khoảng 0.964 cho thấy các biến cảm biến sau tiền xử lý giải thích phần lớn biến thiên của target `C6H6(GT)`.
- Kết quả gần nhau là hợp lý, không phải lỗi, vì cả ba mô hình đều thuộc họ tuyến tính và dataset đang phù hợp tốt với mô hình tuyến tính.

---

## 5. Nhận xét về chất lượng và tính phù hợp

### 5.1. Điểm phù hợp với yêu cầu đề bài

- Có cài đặt thuật toán chính từ công thức toán học.
- Có sử dụng k-fold CV để chọn lambda cho Ridge.
- Có đánh giá mô hình trên test set bằng MAE, RMSE, R2.
- Có kỹ thuật nâng cao để lấy bonus.
- Có unit test cho các hàm chính.
- Có dùng `random_state=42` để kết quả tái lập được.
- Có tách train/test trước khi fit pipeline, tránh data leakage.

### 5.2. Điểm mạnh của phần triển khai

- Code bám sát các helper ma trận có sẵn, không tạo thêm dependency không cần thiết.
- `kfold_cv` thiết kế linh hoạt với `fit_func`, dùng được cho nhiều mô hình.
- Ridge không regularize intercept, đúng quy ước hồi quy.
- Bayesian Linear Regression được tích hợp trực tiếp vào cùng bảng so sánh với OLS/Ridge.
- Kết quả thực nghiệm phù hợp với phân tích dữ liệu: AirQuality có quan hệ tuyến tính mạnh giữa target và các biến cảm biến.

### 5.3. Giới hạn còn lại

- Hiện tại bảng chính có 3 mô hình: OLS, Ridge, Bayesian Linear Regression. Theo Giai đoạn 4, cần bổ sung thêm OLS chọn biến để đạt yêu cầu so sánh \(\ge 4\) mô hình.
- Phần notebook `part2_notebook.ipynb` cần được cập nhật thêm markdown giải thích quy trình chọn lambda, kết quả so sánh model và nhận xét.
- Cần vẽ residual plots cho mô hình tốt nhất ở phần 2 theo yêu cầu Giai đoạn 4/Final.
- Cần bổ sung phần feature importance vào notebook/báo cáo chính.

---

## 6. Nội dung đề xuất đưa vào báo cáo chính

### 6.1. Mô tả ngắn về Ridge

Ridge Regression được sử dụng để cải thiện độ ổn định của OLS khi dữ liệu có đa cộng tuyến. Mô hình thêm thành phần phạt L2 vào hàm mất mát, từ đó co các hệ số về gần 0 và giảm variance. Trong cài đặt, nhóm không regularize intercept để giữ ý nghĩa của hệ số chặn. Lambda được chọn bằng k-fold cross-validation thay vì chọn thủ công.

### 6.2. Mô tả ngắn về chọn lambda

Nhóm sử dụng grid lambda từ \(10^{-4}\) đến \(10^4\). Với mỗi lambda, Ridge được fit trên các fold train và đánh giá bằng MSE trên fold validation. Lambda có CV-MSE trung bình nhỏ nhất được chọn làm lambda tối ưu. Cách làm này giúp quyết định regularization dựa trên dữ liệu train, không dùng test set để tuning.

### 6.3. Mô tả ngắn về Bayesian Linear Regression

Bayesian Linear Regression được chọn làm kỹ thuật nâng cao vì đây là mở rộng tự nhiên của mô hình tuyến tính. Thay vì chỉ ước lượng một vector hệ số như OLS, mô hình đặt prior Gaussian lên hệ số và cập nhật thành posterior sau khi quan sát dữ liệu. Posterior mean được dùng để dự đoán, còn posterior covariance cho biết độ bất định của hệ số.

### 6.4. Giải thích vì sao không chọn Kernel Regression

Kernel Regression có khả năng mô hình hóa quan hệ phi tuyến, nhưng không phù hợp bằng trong bối cảnh này. Dataset AirQuality có số dòng lớn, trong khi Kernel Regression cần nghịch đảo ma trận Gram kích thước theo số mẫu. Ngoài ra, OLS/Ridge đã đạt R2 test khoảng 0.964, cho thấy quan hệ tuyến tính đã giải thích rất tốt target. Bayesian Linear Regression vì vậy là lựa chọn gọn hơn, dễ diễn giải hơn và phù hợp hơn với trọng tâm OLS của đồ án.

### 6.5. Nhận xét kết quả

OLS, Ridge và Bayesian Linear Regression cho kết quả gần như tương đương trên test set. Điều này hợp lý vì các biến cảm biến có tương quan rất mạnh với `C6H6(GT)` và pipeline đã chuẩn hóa dữ liệu. Ridge và Bayesian Linear Regression không tạo cải thiện lớn về R2, nhưng cung cấp lợi ích về ổn định hệ số và diễn giải regularization/prior.

---

## 7. Danh sách file liên quan

| File | Vai trò |
|---|---|
| `part1/ridge_lasso.py` | Cài đặt Ridge Regression, Ridge Trace và unit tests |
| `part1/cross_validation.py` | Cài đặt k-fold CV và unit tests |
| `part2/model_comparison.py` | Chuẩn bị dữ liệu, chọn lambda, train và so sánh mô hình |
| `part2/advanced_methods.py` | Cài đặt Bayesian Linear Regression cho phần nâng cao |
| `part2/data_pipeline.py` | Pipeline tiền xử lý dữ liệu từ Người 3, được tái sử dụng |
| `part2/data/AirQuality.csv` | Dataset thực tế dùng cho phần 2 |

---

