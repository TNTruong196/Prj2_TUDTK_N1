### A. Bài Toán Data Fitting

#### 1. Tổng quát bài toán

**Định nghĩa Bài toán Data Fitting:** Cho tập dữ liệu $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ với $\mathbf{x}_i \in \mathbb{R}^p, y_i \in \mathbb{R}$. Bài toán data fitting là quá trình tìm kiếm một hàm số $f : \mathbb{R}^p \to \mathbb{R}$ nằm trong một lớp hàm cho trước sao cho $f$ xấp xỉ tốt nhất ánh xạ từ biến đầu vào $\mathbf{x}_i$ đến biến mục tiêu $y_i$ dựa trên một tiêu chí đánh giá (hàm mất mát) đã định.

Trong mô hình **hồi quy tuyến tính**, ta thiết lập giả thiết mối quan hệ giữa các biến là tuyến tính:

$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i = \mathbf{x}_i^T \boldsymbol{\beta} + \varepsilon_i, \quad (1)$$

Trong đó chi tiết các thành phần bao gồm:
* $\mathbf{x}_i = (1, x_{i1}, x_{i2}, \dots, x_{ip})^T \in \mathbb{R}^{p+1}$ là vector đặc trưng của quan sát thứ $i$ (thành phần đầu tiên bằng 1 tương ứng với hệ số chặn).
* $\boldsymbol{\beta} = (\beta_0, \beta_1, \dots, \beta_p)^T \in \mathbb{R}^{p+1}$ là vector tham số cần được ước lượng từ tập dữ liệu.
* $\varepsilon_i$ là nhiễu ngẫu nhiên (sai số quan sát).

Khi biểu diễn toàn bộ tập dữ liệu dưới dạng ma trận với $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ được gọi là ma trận thiết kế (design matrix - có cột đầu tiên toàn 1), ta có phương trình tổng quát:

$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}. \quad (2)$$

Với $\mathbf{y} \in \mathbb{R}^n$ là vector cột chứa các giá trị quan sát $y_i$, và $\boldsymbol{\varepsilon} \in \mathbb{R}^n$ là vector sai số tương ứng.

---

#### 2. Các Giả Thiết Gauss–Markov

Để các ước lượng của mô hình hồi quy tuyến tính đạt được các tính chất thống kê tốt nhất, nền tảng lý thuyết yêu cầu phải thỏa mãn **Các giả thiết Gauss–Markov (GM1 – GM5)**:

* **GM1. Tuyến tính (Linear in parameters):** $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$.  
    Mô hình thực tế phải tuyến tính đối với các tham số $\boldsymbol{\beta}$, dù có thể phi tuyến đối với các biến độc lập.

* **GM2. Không hoàn hảo đa cộng tuyến (No perfect multicollinearity):** $\text{rank}(\mathbf{X}) = p + 1$.  
    Các cột của ma trận thiết kế $\mathbf{X}$ phải độc lập tuyến tính. Điều kiện này đảm bảo ma trận vuông $\mathbf{X}^T\mathbf{X}$ có khả năng nghịch đảo, từ đó hệ phương trình mới có nghiệm duy nhất.

* **GM3. Ngoại sinh (Strict Exogeneity):** $\mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}$, tức $\mathbb{E}[\varepsilon_i \mid \mathbf{x}_i] = 0$.  
    Giá trị kỳ vọng của sai số ngẫu nhiên bằng 0, mang ý nghĩa các nhiễu này không mang thông tin có tính hệ thống liên quan đến bộ biến độc lập $\mathbf{X}$.

* **GM4. Đồng phương sai (Homoscedasticity) và Không tự tương quan:** $\text{Var}(\boldsymbol{\varepsilon} \mid \mathbf{X}) = \sigma^2 \mathbf{I}_n$.  
    Phát biểu này tương đương với hai yếu tố:
    1. $\text{Var}(\varepsilon_i) = \sigma^2$: Phương sai của phần dư là hằng số trên toàn bộ tập dữ liệu (không thay đổi theo sự thay đổi của $\mathbf{X}$).
    2. $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ với mọi $i \neq j$: Không có hiện tượng tự tương quan giữa phần dư của các quan sát khác nhau.

* **GM5. Phần dư Chuẩn (Normality of Errors):** $\boldsymbol{\varepsilon} \mid \mathbf{X} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)$.  
    Các sai số ngẫu nhiên có phân phối chuẩn. Đây là giả thiết không bắt buộc để tìm nghiệm OLS tối ưu, nhưng là điều kiện cần thiết để thực hiện các thủ tục suy diễn thống kê như xây dựng khoảng tin cậy và tiến hành kiểm định giả thuyết (kiểm định t, kiểm định F).

### B. Phương Pháp Ordinary Least Squares (OLS)

#### 1. Hàm mất mát và nghiệm OLS

Phương pháp Bình phương tối thiểu thông thường (OLS) xác định vector tham số $\hat{\boldsymbol{\beta}}$ bằng cách tối thiểu hóa tổng bình phương các sai số giữa giá trị thực tế và giá trị dự báo. Đại lượng này được gọi là **Tổng bình phương phần dư (Residual Sum of Squares - RSS)**:

$$RSS(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = \sum_{i=1}^{n}(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2. \quad (3)$$

**Định lý:** Nếu ma trận $\mathbf{X}^T\mathbf{X}$ là ma trận khả nghịch, nghiệm OLS duy nhất được xác định bởi công thức:

$$\hat{\boldsymbol{\beta}}_{OLS} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}. \quad (4)$$

**Chứng minh:**
Để tìm giá trị cực tiểu của $RSS(\boldsymbol{\beta})$, ta tính đạo hàm (gradient) của hàm mất mát theo $\boldsymbol{\beta}$ và giải phương trình bằng 0:
1. Khai triển hàm RSS: $RSS(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$.
2. Tính đạo hàm: $\nabla_{\boldsymbol{\beta}}RSS = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$.
3. Giải phương trình đạo hàm bằng không:
   $$-2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0 \iff \mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}.$$
Đây được gọi là **hệ phương trình chuẩn (Normal Equations)**. Khi $\mathbf{X}^T\mathbf{X}$ khả nghịch, ta thu được nghiệm (4).

---

#### 2. Ma Trận Chiếu và Hat Matrix

**Định nghĩa Hat Matrix:** Ma trận chiếu (projection matrix), hay còn gọi là hat matrix, được ký hiệu là $\mathbf{H}$, dùng để ánh xạ từ vector quan sát thực tế $\mathbf{y}$ đến vector giá trị dự báo $\hat{\mathbf{y}}$:

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \in \mathbb{R}^{n \times n}. \quad (5)$$

**Tính chất của ma trận H:**
* **(i) Tính lũy đẳng (Idempotent):** $\mathbf{H}^2 = \mathbf{H}$. Việc chiếu một vector hai lần lên cùng một không gian không làm thay đổi kết quả sau lần chiếu đầu tiên.
* **(ii) Tính đối xứng:** $\mathbf{H}^T = \mathbf{H}$.
* **(iii) Giá trị riêng (Eigenvalues):** Các giá trị riêng của $\mathbf{H}$ chỉ có thể là 0 hoặc 1.
* **(iv) Hạng (Rank):** $\text{rank}(\mathbf{H}) = p + 1$.
* **(v) Giá trị dự báo và phần dư:**
  * Vector dự báo: $\hat{\mathbf{y}} = \mathbf{Hy}$.
  * Vector phần dư: $\hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y}$.

Về mặt hình học, $\hat{\mathbf{y}}$ là hình chiếu vuông góc của $\mathbf{y}$ lên không gian cột của ma trận thiết kế $\mathbf{X}$.

---

#### 3. Định Lý Gauss-Markov

**Định lý:** Dưới các giả thiết từ **GM1 đến GM4**, ước lượng OLS $\hat{\boldsymbol{\beta}}_{OLS}$ là **Ước lượng Tuyến tính Không chệch Tốt nhất (Best Linear Unbiased Estimator - BLUE)**. Điều này có nghĩa là:

* **(i) Tính không chệch:** Kỳ vọng của ước lượng bằng giá trị thực của tham số, $\mathbb{E}[\hat{\boldsymbol{\beta}}_{OLS}] = \boldsymbol{\beta}$.
* **(ii) Tính tốt nhất (Hiệu quả):** Trong số tất cả các ước lượng tuyến tính và không chệch của $\boldsymbol{\beta}$, ước lượng OLS có phương sai nhỏ nhất. Với bất kỳ ước lượng không chệch $\tilde{\beta}_j$ nào khác, ta luôn có $\text{Var}(\tilde{\beta}_j) \ge \text{Var}(\hat{\beta}_{j}^{OLS})$.

Ma trận hiệp phương sai của ước lượng OLS được xác định bởi:
$$\text{Var}(\hat{\boldsymbol{\beta}}_{OLS} \mid \mathbf{X}) = \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}. \quad (6)$$

---

#### 4. Ước Lượng Phương Sai Nhiễu

Vì phương sai của nhiễu $\sigma^2$ thường không được biết trước trong thực tế, ta sử dụng một ước lượng không chệch dựa trên phần dư của mô hình:

$$\hat{\sigma}^2 = \frac{RSS}{n - p - 1} = \frac{\\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}\\|^2}{n - p - 1}. \quad (7)$$

Trong đó $n - p - 1$ đại diện cho bậc tự do của mô hình (số lượng quan sát trừ đi số lượng tham số cần ước lượng bao gồm cả hệ số chặn). Đây là thành phần quan trọng để tính toán sai số tiêu chuẩn và thực hiện các kiểm định thống kê sau này.