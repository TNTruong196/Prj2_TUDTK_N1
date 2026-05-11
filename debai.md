# ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH
## TRƯỜNG ĐẠI HỌC KHOA HỌC TỰ NHIÊN
### KHOA CÔNG NGHỆ THÔNG TIN

# ĐỒ ÁN 2
## Data Fitting và Phương Pháp OLS

| Thông tin | Chi tiết |
| :--- | :--- |
| **Môn học:** | Toán Ứng Dụng và Thống Kê |
| **Mã môn:** | MTH00051 |
| **Học kỳ:** | HỌC KỲ 2, 2025-2026 |

**Thông tin giảng viên**
* **GV Thực hành:** ThS. Võ Nam Thục Đoan, ThS. Lê Nhựt Nam
* **E-mail:** {vntdoan, lnnam}@fit.hcmus.edu.vn

> *Tài liệu này dành riêng cho mục đích học thuật.*

---

## Mục lục

* **Giới Thiệu Đồ Án**
* **1. Phần 1: Lý Thuyết Data Fitting và Minh Họa**
    * 1.1 Bài Toán Data Fitting
    * 1.2 Phương Pháp Ordinary Least Squares (OLS)
    * 1.3 Đánh Giá Mô Hình
    * 1.4 Các Vấn Đề Nâng Cao trong Data Fitting
    * 1.5 Yêu Cầu Cài Đặt Python - Phần 1
    * 1.6 Tiêu Chí Đánh Giá - Phần 1
* **2. Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế**
    * 2.1 Tiêu Chí Chọn Bộ Dữ Liệu
    * 2.2 Tiền Xử Lý Dữ Liệu
    * 2.3 Xây Dựng và Đánh Giá Mô Hình
    * 2.4 Kỹ Thuật Nâng Cao (Tùy Chọn)
    * 2.5 Yêu Cầu Cài Đặt Python - Phần 2
    * 2.6 Tiêu Chí Đánh Giá - Phần 2
* **3. Yêu Cầu Chung và Hướng Dẫn Nộp Bài**
    * 3.1 Cấu Trúc Báo Cáo
    * 3.2 Cấu Trúc Thư Mục Nộp Bài
    * 3.3 Yêu Cầu Kỹ Thuật
    * 3.4 Phân Công Nhóm và Đạo Đức Học Thuật
    * 3.5 Thang Điểm Tổng Hợp
* **Tài Liệu Tham Khảo**

---

## Giới Thiệu Đồ Án

### Mục tiêu tổng quát
Đồ án này tập trung vào hai nhóm nhiệm vụ bổ sung cho nhau:
1.  **Lý thuyết và minh họa** - Nắm vững nền tảng toán học của data fitting và phương pháp Ordinary Least Squares (OLS), sau đó minh họa các kết quả lý thuyết bằng code Python tự cài đặt.
2.  **Ứng dụng thực tế** - Vận dụng data fitting để phân tích một bộ dữ liệu thực, bao gồm tiền xử lý, xây dựng mô hình hồi quy và đánh giá kết quả một cách có hệ thống.

Sau khi hoàn thành đồ án, sinh viên có khả năng:
* Giải thích và chứng minh các tính chất cốt lõi của OLS (unbiasedness, BLUE, Gauss-Markov).
* Cài đặt pipeline data fitting hoàn chỉnh từ đầu bằng Python, có thể so sánh được với thư viện sklearn Linear Regression.
* Phân tích và xử lý bộ dữ liệu thực có missing values, outliers và các vấn đề thực tiễn.
* Đánh giá mô hình một cách toàn diện (hệ số $R^{2}$, residual analysis, cross-validation).

### Các công cụ cho phép sử dụng trong đồ án
* **Python 3.10+:** Ngôn ngữ cài đặt chính.
* **NumPy, SciPy:** Tính toán số; dùng để kiểm chứng, không dùng để thay thế cài đặt thuật toán.
* **Pandas:** Đọc, xử lý và thao tác dữ liệu.
* **Matplotlib, Seaborn:** Trực quan hóa dữ liệu và kết quả mô hình.
* **Scikit-learn:** Chỉ dùng để so sánh và kiểm chứng kết quả, không dùng để cài đặt OLS chính.
* **Jupyter Notebook:** Trình bày toàn bộ thực nghiệm.

> **Lưu ý:** Các hàm như `sklearn.linear_model.LinearRegression`, `numpy.linalg.lstsq` chỉ được dùng để kiểm chứng (verification). Phần cài đặt thuật toán chính phải được viết từ đầu dựa trên công thức toán học.

---

## 1. Phần 1: Lý Thuyết Data Fitting và Minh Họa

**Tóm tắt yêu cầu Phần 1:** Trình bày lại kiến thức đã học về data fitting và OLS. Với mỗi kết quả lý thuyết, sinh viên viết code Python để minh họa và kiểm chứng bằng dữ liệu giả lập (synthetic data).

### 1.1. Bài Toán Data Fitting

#### 1.1.1. Phát biểu bài toán tổng quát
**Định nghĩa 1.1 (Bài toán Data Fitting).** Cho tập dữ liệu $\mathcal{D}=\{(x_{i},y_{i})\}_{i=1}^{n}$ với $x_{i}\in\mathbb{R}^{p}$, $y_{i}\in\mathbb{R}$. Bài toán data fitting là tìm hàm $f:\mathbb{R}^{p}\rightarrow\mathbb{R}$ trong một lớp hàm cho trước sao cho $f$ xấp xỉ tốt nhất ánh xạ từ $x$ đến $y_{i}$ theo một tiêu chí đã định.

Trong mô hình hồi quy tuyến tính, ta giả thiết:
$$y_{i}=\beta_{0}+\beta_{1}x_{i1}+\beta_{2}x_{i2}+\cdot\cdot\cdot+\beta_{p}x_{ip}+\epsilon_{i}=x_{i}^{T}\beta+\epsilon_{i}$$
với $\beta=(\beta_{0},\beta_{1},...,\beta_{p})^{T}\in\mathbb{R}^{p+1}$ là vector tham số cần ước lượng và $\epsilon$ là nhiễu ngẫu nhiên.

Viết dưới dạng ma trận với $X\in\mathbb{R}^{n\times(p+1)}$ (ma trận design có cột đầu toàn 1):
$$y=X\beta+\epsilon$$

#### 1.1.2. Các Giả Thiết Gauss-Markov
Các giả thiết Gauss-Markov (GM1 - GM5)
* **GM1. Tuyến tính:** $y=X\beta+\epsilon$.
* **GM2. Không hoàn hảo đa cộng tuyến:** $rank(X)=p+1$ (các cột độc lập tuyến tính).
* **GM3. Ngoại sinh:** $\mathbb{E}[\epsilon|X]=0,$ tức $\mathbb{E}[\epsilon_{i}|x_{i}]=0$.
* **GM4. Đồng phương sai:** $Var(\epsilon|X)=\sigma^{2}I_{n}$ tức $Var(\epsilon_{i})=\sigma^{2}$ và $Cov(\epsilon_{i},\epsilon_{j})=0$ với $i\ne j$.
* **GM5. Phần dư Chuẩn:** $\epsilon|X\sim\mathcal{N}(0,\sigma^{2}I_{n})$.

### 1.2. Phương Pháp Ordinary Least Squares (OLS)

#### 1.2.1. Hàm mất mát và nghiệm OLS
OLS tìm $\hat{\beta}$ tối thiểu hóa tổng bình phương phần dư (Residual Sum of Squares):
$$RSS(\beta)=||y-X\beta||_{2}^{2}=\sum_{i=1}^{n}(y_{i}-x_{i}^{T}\beta)^{2}$$

**Định lý 1.1 (Nghiệm OLS - Normal Equations).** Nếu $X^{T}X$ khả nghịch, nghiệm OLS duy nhất là:
$$\hat{\beta}_{OLS}=(X^{T}X)^{-1}X^{T}y$$

*Chứng minh.* Tính đạo hàm và cho đạo hàm bằng không:
$$\nabla_{\beta}RSS=-2X^{T}(y-X\beta)=0$$
Sau đó giải ra: $X^{T}X\beta=X^{T}y$.

#### 1.2.2. Ma Trận Chiếu và Hat Matrix
**Định nghĩa 1.2 (Hat Matrix).** Ma trận chiếu (projection matrix hay hat matrix) là:
$$H=X(X^{T}X)^{-1}X^{T}\in\mathbb{R}^{n\times n}$$

**Mệnh đề 1.1 (Tính chất của H).**
* (i) $H^{2}=H$ (idempotent).
* (ii) $H^{T}=H$ (đối xứng).
* (iii) Giá trị riêng của H: chỉ là 0 hoặc 1.
* (iv) $rank(H)=p+1$.
* (v) Giá trị fitted: $\hat{y}=Hy;$ phần dư: $\hat{\epsilon}=(I-H)y$

#### 1.2.3. Định Lý Gauss-Markov
**Định lý 1.2 (Gauss-Markov).** Dưới các giả thiết GM1-GM4, ước lượng OLS $\hat{\beta}_{OLS}$ là ước lượng tuyến tính không chệch tốt nhất (Best Linear Unbiased Estimator - BLUE):
* (i) Không chệch: $\mathbb{E}[\hat{\beta}_{OLS}]=\beta$.
* (ii) Tốt nhất (phương sai nhỏ nhất): Với mọi ước lượng tuyến tính không chệch $\beta$ khác, ta có $Var(\beta_{j})\ge Var(\beta_{j}^{OLS})$ với mọi j.

Ma trận hiệp phương sai của $\hat{\beta}_{OLS}$:
$$Var(\hat{\beta}_{OLS}|X)=\sigma^{2}(X^{T}X)^{-1}$$

#### 1.2.4. Ước Lượng Phương Sai Nhiễu
Ước lượng không chệch của $\sigma^{2}$:
$$\hat{\sigma}^{2}=\frac{RSS}{n-p-1}=\frac{||y-X\hat{\beta}||^{2}}{n-p-1}$$

### 1.3. Đánh Giá Mô Hình

#### 1.3.1. Hệ số xác định $R^{2}$ và $R^{2}$ hiệu chỉnh
**Định nghĩa 1.3 (Hệ số xác định).** Hệ số xác định $R^{2}$ được định nghĩa như sau:
$$R^{2}=1-\frac{RSS}{TSS}=1-\frac{\sum_{i}(y_{i}-\hat{y}_{i})^{2}}{\sum_{i}(y_{i}-\overline{y})^{2}}$$
Với $R^{2}\in[0,1]$.

$R^{2}$ luôn tăng khi thêm biến. Để so sánh các mô hình có số biến khác nhau, ta hay dùng $R^{2}$ hiệu chỉnh:
$$\overline{R}^{2}=1-\frac{n-1}{n-p-1}(1-R^{2})$$

#### 1.3.2. Kiểm Định Giả Thuyết
Dưới giả thiết chuẩn GM5, $\hat{\beta}\sim\mathcal{N}(\beta,\sigma^{2}(X^{T}X)^{-1})$.
* **Kiểm định Student, t-test** - Kiểm định ý nghĩa của từng đặc trưng đối với mô hình:
    $$t_{j}=\frac{\beta_{j}}{\hat{\sigma}\sqrt{[(X^{T}X)^{-1}]_{jj}}}\sim t_{n-p-1} \text{ (với } H_{0}:\beta_{j}=0)$$
* **Kiểm định F cho mô hình tổng thể** – Kiểm định ý nghĩa của mô hình:
    $$F=\frac{(TSS-RSS)/p}{RSS/(n-p-1)}\sim F_{p,n-p-1} \text{ (với } H_{0}:\beta_{1}=\cdot\cdot\cdot=\beta_{p}=0)$$

### 1.4. Các Vấn Đề Nâng Cao trong Data Fitting

#### 1.4.1. Đa cộng tuyến (Multicollinearity)
Đa cộng tuyến xảy ra khi các cột của X có tương quan cao, khiến $X^{T}X$ gần suy biến. Để phát hiện hiện tượng này, chúng ta có thể sử dụng hệ số phóng đại (Variance Inflation Factor):
$$VIF_{j}=\frac{1}{1-R_{j}^{2}}$$
trong đó $R_{j}^{2}$ là $R^{2}$ khi hồi quy biến $X_{j}$ theo các biến còn lại. $VIF>10$ cho thấy đa cộng tuyến nghiêm trọng.

#### 1.4.2. Hồi Quy Ridge và Lasso (Regularization)
Khi dữ liệu có nhiều đặc trưng hoặc đa cộng tuyến, ta cần thêm thành phần chính quy hoá (regularization):
* **Ridge Regression ($L_{2}$):**
    $$\hat{\beta}_{ridge}=\arg\min_{\beta}\{||y-X\beta||^{2}+\lambda||\beta||_{2}^{2}\}=(X^{T}X+\lambda I)^{-1}X^{T}y$$
* **Lasso Regression ($L_{1}$):**
    $$\hat{\beta}_{lasso}=\arg\min_{\beta}\{||y-X\beta||^{2}+\lambda||\beta||_{1}\}$$

> Lasso không có nghiệm closed-form; giải bằng coordinate descent hoặc các phương pháp dưới gradient (subgradient methods).

#### 1.4.3. Phân Tích Phần Dư (Residual Analysis)
Sử dụng các công cụ thống kê mô tả để kiểm tra sai số của mô hình:
* **Residuals vs Fitted:** Kiểm tra tính tuyến tính và đồng phương sai.
* **Q-Q Plot:** Kiểm tra tính chuẩn của phần dư.
* **Scale-Location:** Kiểm tra phương sai đồng đều (homoscedasticity).
* **Cook's Distance:** Xác định các quan sát có ảnh hưởng lớn (influential points).

#### 1.4.4. Cross-Validation và Lựa Chọn Mô Hình
**k-Fold Cross-Validation:** Chia dữ liệu thành $k$ phần bằng nhau. Mỗi lần dùng $k-1$ phần để huấn luyện, 1 phần để kiểm tra. Lặp k lần và lấy trung bình:
$$CV_{(k)}=\frac{1}{k}\sum_{i=1}^{k}MSE_{i}$$

**Tiêu chí lựa chọn mô hình (AIC/BIC):**
$$AIC=n\ln\left(\frac{RSS}{n}\right)+2(p+2)$$
$$BIC=n\ln\left(\frac{RSS}{n}\right)+(p+2)\ln(n)$$

### 1.5. Yêu Cầu Cài Đặt Python - Phần 1

Với mỗi mục dưới đây, sinh viên phải: (a) trình bày công thức toán học, (b) cài đặt Python từ đầu, (c) minh họa bằng dữ liệu giả lập, (d) kiểm chứng với NumPy/sklearn.

1.  `ols_fit(X, y)` - Tính $\hat{\beta} = (X^{T}X)^{-1}X^{T}y$ và $\hat{\sigma}^{2}$.
2.  `hat_matrix(X)` - Tính $H=X(X^{T}X)^{-1}X^{T}$ kiểm tra idempotent.
3.  `model_metrics(y, y_hat, p)` - Tính RSS, TSS, $R^{2}$, $\overline{R}^{2}$, kiểm định F.
4.  `coef_inference(X, y, beta_hat, sigma2)` - Tính standard errors, t-statistics, p-values và khoảng tin cậy 95%.
5.  `vif(X)` - Tính VIF cho từng biến.
6.  `ridge_fit(X, y, lam)` - Cài đặt Ridge Regression, vẽ ridge trace.
7.  `residual_plots(X, y, beta_hat)` - Vẽ 4 biểu đồ phân tích phần dư.
8.  `kfold_cv(X, y, k)` - Cài đặt k-fold cross-validation, tính CV score.
9.  Minh họa định lý Gauss-Markov: Mô phỏng Monte Carlo để kiểm chứng $\mathbb{E}[\hat{\beta}]=\beta$ và OLS có phương sai nhỏ nhất.

### 1.6. Tiêu Chí Đánh Giá - Phần 1

| Tiêu chí | Mô tả | Điểm |
| :--- | :--- | :--- |
| Trình bày lý thuyết OLS | Đúng, đầy đủ công thức, có chứng minh | 1.0 |
| Cài đặt OLS từ đầu | Đúng, kiểm chứng với NumPy | 1.0 |
| Hat matrix và tính chất | Cài đặt, kiểm tra idempotent | 0.5 |
| Kiểm định hệ số (t, F) | Tính đúng t-stat, p-value | 0.5 |
| Regularization (Ridge/Lasso) | Cài đặt, vẽ ridge trace | 1.0 |
| Phân tích phần dư | 4 biểu đồ đầy đủ, nhận xét | 0.5 |
| Cross-validation | Cài k-fold CV, so sánh mô hình | 0.5 |
| Minh họa Gauss-Markov | Monte Carlo rõ ràng, nhận xét | 0.5 |
| Trình bày Notebook | Rõ ràng, có markdown giải thích | 0.5 |
| **Tổng Phần 1** | | **6.0** |

---

## 2. Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế

**Tóm tắt yêu cầu Phần 2:** Chọn ít nhất một bộ dữ liệu thực có missing values, thực hiện tiền xử lý, áp dụng data fitting để giải bài toán hồi quy, đánh giá và phân tích kết quả.

### 2.1. Tiêu Chí Chọn Bộ Dữ Liệu
Bộ dữ liệu phải thỏa mãn đồng thời các điều kiện:
1.  **Dữ liệu thực (real-world):** Thu thập từ quan sát thực tế, không phải dữ liệu tổng hợp (synthetic) hay dữ liệu toy.
2.  **Có missing values:** Dữ liệu gốc phải chứa ít nhất một cột có giá trị bị thiếu ($\ge5\%$ dữ liệu bị thiếu).
3.  **Biến mục tiêu liên tục:** Bài toán hồi quy (regression), không phải phân loại (classification).
4.  **Kích thước hợp lý:** $n\ge200$ quan trắc, $p\ge3$ biến đặc trưng.
5.  **Nguồn đáng tin cậy:** Kaggle, UCI ML Repository, data.gov, World Bank, v.v.

*Gợi ý:* House Prices, UCI Auto MPG, World Bank Open Data, v.v.

### 2.2. Tiền Xử Lý Dữ Liệu

#### 2.2.1. Khảo Sát Dữ Liệu (Exploratory Data Analysis - EDA)
* Thống kê mô tả: mean, median, std, min, max, quartiles.
* Phân phối từng biến: histogram, boxplot.
* Ma trận tương quan: heatmap.
* Kiểm tra dữ liệu trùng lắp và phân tích missing values.
* Phát hiện outliers: phương pháp IQR, z-score.

#### 2.2.2. Xử Lý Missing Values
* **MV1. Listwise deletion:** Xóa toàn bộ hàng có giá trị thiếu.
* **MV2. Mean/Median/Mode imputation:** Thay giá trị thiếu bằng thống kê của cột.
* **MV3. Regression imputation:** Dự đoán giá trị thiếu bằng hồi quy.
* **MV4. k-NN imputation:** Thay giá trị thiếu bằng trung bình k quan sát gần nhất.
* **MV5. Multiple Imputation (MICE):** Gộp kết quả theo quy tắc Rubin.

#### 2.2.3. Các Bước Tiền Xử Lý Khác
* **Feature engineering:** Tạo biến mới, biến đổi phi tuyến (log, đa thức).
* **Encoding biến phân loại:** One-hot encoding hoặc ordinal encoding.
* **Chuẩn hóa:** (z-score standardization) $x_{j}^{std}=\frac{x_{j}-\overline{x_{j}}}{s_{j}}$.
* **Xử lý outliers:** Winsorization hoặc loại bỏ có căn cứ.
* **Kiểm tra đa cộng tuyến:** VIF.

### 2.3. Xây Dựng và Đánh Giá Mô Hình

Quy trình xây dựng mô hình tuân theo pipeline: `EDA -> Tiền xử lý -> Train/Test Split -> Xây dựng mô hình -> Đánh giá -> Tinh chỉnh -> Báo cáo kết quả`.

#### Các Mô Hình Cần Thử Nghiệm
Sinh viên xây dựng và so sánh ít nhất 3 mô hình:

| Mô hình | Loại | Mô tả |
| :--- | :--- | :--- |
| **OLS cơ bản** | Bắt buộc | Hồi quy với tất cả các biến (sau tiền xử lý) |
| **OLS chọn biến** | Bắt buộc | Loại bỏ biến dựa trên p-value hoặc VIF |
| **Ridge / Lasso** | Bắt buộc | Regularization, chọn $\lambda$ qua CV |
| **Polynomial / Interaction** | Tùy chọn | Thêm đặc trưng phi tuyến |
| **Kernel / Bayesian** | Nâng cao | Xem mục 2.4 |

#### Tiêu Chí So Sánh Mô Hình
Mỗi mô hình được đánh giá trên tập test bằng MAE, RMSE và $R^2$:
$$MAE=\frac{1}{n_{test}}\sum_{i}|y_{i}-\hat{y}_{i}|$$
$$RMSE=\sqrt{\frac{1}{n_{test}}\sum_{i}(y_{i}-\hat{y}_{i})^{2}}$$
$$R_{test}^{2}=1-\frac{RSS_{test}}{TSS_{test}}$$

### 2.4. Kỹ Thuật Nâng Cao (Tùy Chọn)

**Kernel Regression**
Mở rộng OLS sang không gian đặc trưng phi tuyến thông qua kernel trick:
$$\hat{y}(x)=k(x)^{T}(K+\lambda I)^{-1}y$$
với $K_{ij}=k(x_{i},x_{j})$ là ma trận Gram và $k(\cdot,\cdot)$ là hàm kernel (vd: RBF):
$$k_{RBF}(x,x^{\prime})=\exp\left(-\frac{||x-x^{\prime}||^{2}}{2l^{2}}\right)$$

**Bayesian Linear Regression**
Phân phối hậu nghiệm (conjugate):
$$\beta|X,y\sim\mathcal{N}(m_{n},S_{n})$$
$$S_{n}=\left(S_{0}^{-1}+\frac{1}{\sigma^{2}}X^{T}X\right)^{-1}, \quad m_{n}=S_{n}\left(S_{0}^{-1}m_{0}+\frac{1}{\sigma^{2}}X^{T}y\right)$$

### 2.5. Yêu Cầu Cài Đặt Python - Phần 2
1.  **Pipeline tiền xử lý:** Viết class `DataPipeline` xử lý missing values, encoding, chuẩn hóa theo thứ tự (fit trên train, transform trên test).
2.  **So sánh 3+ mô hình:** Bảng tổng hợp MAE, RMSE, $R^{2}$ trên test set.
3.  **Cross-validation:** Dùng k-fold ($k=5$ hoặc $k=10$) chọn siêu tham số $\lambda$ cho Ridge/Lasso.
4.  **Phân tích phần dư:** Vẽ 4 biểu đồ chẩn đoán cho mô hình tốt nhất.
5.  **Feature importance:** Vẽ biểu đồ hệ số hồi quy để giải thích mô hình.
6.  **Nhận xét:** Giải thích kết quả theo ngữ cảnh dữ liệu.

### 2.6. Tiêu Chí Đánh Giá - Phần 2

| Tiêu chí | Mô tả | Điểm |
| :--- | :--- | :--- |
| Chọn và mô tả dữ liệu | Đúng tiêu chí, mô tả rõ nguồn gốc | 0.5 |
| EDA | Đầy đủ thống kê mô tả, biểu đồ | 0.5 |
| Xử lý missing values | Đúng phương pháp, có giải thích | 1.0 |
| Tiền xử lý tổng thể | Pipeline đầy đủ, fit/transform đúng | 0.5 |
| Xây dựng $\ge3$ mô hình | OLS, Ridge/Lasso, một mô hình khác | 1.5 |
| Đánh giá trên test set | MAE, RMSE, $R^{2}$, phân tích phần dư | 1.0 |
| Nhận xét và kết luận | Phân tích có chiều sâu, liên hệ thực tế | 0.5 |
| Kỹ thuật nâng cao | Kernel / Bayesian (tùy chọn, bonus) | +0.5 |
| **Tổng Phần 2** | | **5.5 (+0.5)** |

---

## 3. Yêu Cầu Chung và Hướng Dẫn Nộp Bài

### 3.1. Cấu Trúc Báo Cáo
Báo cáo viết bằng LaTeX hoặc Markdown (xuất ra PDF), bao gồm:
1. Trang bìa.
2. Mục lục.
3. Phần 1: Lý thuyết và minh họa.
4. Phần 2: Ứng dụng thực tế.
5. Kết luận.
6. Tài liệu tham khảo (Ít nhất 5 tài liệu).
7. Phụ lục (Bảng số liệu, biểu đồ bổ sung).

### 3.2. Cấu Trúc Thư Mục Nộp Bài
```text
Group_<ID>/
|-- README.md
|-- requirements.txt
|-- report/
|   |-- report.pdf
|   |-- report.tex
|-- part1/
|   |-- ols_implementation.py
|   |-- ridge_lasso.py
|   |-- residual_analysis.py
|   |-- cross_validation.py
|   |-- part1_notebook.ipynb
|-- part2/
|   |-- data/
|   |   |-- <ten_dataset>.csv
|   |-- data_pipeline.py
|   |-- model_comparison.py
|   |-- advanced_methods.py
|   |-- part2_notebook.ipynb

### 3.3. Yêu Cầu Kỹ Thuật

Sử dụng Python 3.10+, viết code rõ ràng (clean code), chú thích code nếu thật sự cần thiết.
* Tất cả biểu đồ phải có tiêu đề, nhãn trục, chú thích đầy đủ.

Mọi quyết định (chọn $\lambda$, chọn k, xử lý outlier) phải được giải thích bằng lý luận, không phải thử-sai ngẫu nhiên.
* Kết quả phải tái lập được (reproducible): đặt `random_state` / `seed` cụ thể.
* Mỗi hàm có ít nhất 2 unit test kiểm tra kết quả trên dữ liệu đã biết.

### 3.4. Phân Công Nhóm và Đạo Đức Học Thuật

**Lưu ý**
* Báo cáo phải ghi rõ phân công công việc của từng thành viên.
* Giảng viên sẽ chọn lựa một số nhóm để vấn đáp nếu cần thiết.
* Nghiêm cấm sao chép code hoặc báo cáo từ nhóm khác mà không trích dẫn nguồn.
* Sử dụng AI (ChatGPT, Copilot, v.v.) để gợi ý là được phép, nhưng phải hiểu và giải thích được toàn bộ code nộp.
* Vi phạm đạo đức học thuật dẫn đến điểm 0 toàn bộ đồ án.

### 3.5. Thang Điểm Tổng Hợp

| Phần | Nội dung | Điểm tối đa | Trọng số |
| :--- | :--- | :--- | :--- |
| **1** | Lý thuyết, minh họa, cài đặt OLS | 6.0 | 52% |
| **2** | Ứng dụng dữ liệu thực | 5.5 | 48% |
| **Bonus** | Kỹ thuật nâng cao (Kernel/Bayesian) | +0.5 | |
| **Tổng cộng** | | **11.5 (+0.5)** | **100%** |

Điểm cuối cùng $= \min(\text{Tổng}/1.15, 10)$, quy về thang điểm 10.

---

### Tóm tắt sản phẩm nộp bài

* **Báo cáo `report.pdf`** (bắt buộc)
* **Source code đầy đủ** kèm `README.md` và `requirements.txt`
* **Jupyter Notebooks:** `part1_notebook.ipynb` và `part2_notebook.ipynb`
* **Dữ liệu gốc:** file `.csv` hoặc link download trong README
* **Nộp qua:** Moodle của Khoa
* **Hạn nộp: [ngày 30/5/2026, trước 23:59]**

---

## Tài Liệu Tham Khảo

[1] Gilbert Strang. *Introduction to Linear Algebra*, 6th ed. Wellesley-Cambridge Press, 2023.
[2] Gareth James, Daniela Witten, Trevor Hastie & Robert Tibshirani. *An Introduction to Statistical Learning*, 2nd ed. Springer, 2021. Truy cập miễn phí: https://www.statlearning.com
[3] Trevor Hastie, Robert Tibshirani & Jerome Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. Truy cập miễn phí: https://hastie.su.domains/ElemStatLearn/
[4] Christopher M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006. (Chương 3: Linear Models for Regression)
[5] Kevin P. Murphy. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022. Truy cập miễn phí: https://probml.github.io/pml-book/book1.html
[6] Jake VanderPlas. *Python Data Science Handbook*. O'Reilly, 2016. https://jakevdp.github.io/PythonDataScienceHandbook/
[7] Wes McKinney. *Python for Data Analysis*, 3rd ed. O'Reilly, 2022.