# Phân công
#### 1. Nhật
#### 2. Việt
#### 3. Trường 
#### 4. Tân
#### 5. Phúc


# Công việc chi tiết
# Note:
#### mỗi hàm code tính nắng thì cần làm ít nhất 2 unit test
#### làm báo cáo thì (task yêu cầu làm trong report.tex) thì chỉ cần làm ra 1 file md để người 5 lấy nội dung
#### làm phần nào thì có báo cáo phần đó phục vụ cho người 5 làm báo cáo. 

# Giai đoạn 1 (Deadline: 23h 13/5)

## 1
* Viết và chứng minh lý thuyết toán học của data fitting và OLS bằng LaTeX (chuẩn bị trước cho báo cáo). *(Phần 1 | File: `report.tex`)* 
* Code ols_fit(X, y) từ đầu (tính ma trận $(X^T X)^{-1} X^T y$ và $\hat{\sigma}^2$). *(Phần 1 | File: `part1/ols_implementation.py`)* 
* Code hat_matrix(X) (kiểm tra tính idempotent). *(Phần 1 | File: `part1/ols_implementation.py`)* 
* Viết Unit Test (ít nhất 2 test/hàm) cho ols_fit, hat_matrix và **tất cả các hàm tự cài đặt sau này**, kiểm tra trên dữ liệu đã biết để đảm bảo khớp 100% với NumPy/sklearn. *(Phần 1 | File: `part1/ols_implementation.py` và `part1_notebook.ipynb`)* 

## 3
* Code EDA cho bộ dữ liệu đã có: **thống kê mô tả, vẽ phân phối, heatmap tương quan, và bắt buộc thực hiện lọc outliers bằng IQR/z-score**. Phân tích và ghi chú rõ cơ chế Missing Value (MCAR, MAR hay MNAR) làm cơ sở chọn phương pháp xử lý. *(Phần 2 | File: `part2_notebook.ipynb`)* 
* Bắt đầu viết class DataPipeline với các hàm fit() và transform() (tập trung xử lý missing values trước dựa trên biện luận). *(Phần 2 | File: `part2/data_pipeline.py`)*

## Cả nhóm
* Thống nhất một random_state (seed) dùng chung cho toàn bộ project để đảm bảo kết quả hoàn toàn tái lập được (reproducible). *(Chung | File: Tất cả các file `.py` và `.ipynb`)*

---

# Giai đoạn 2 (Deadline: 23h 17/5)

## 2
* Code model_metrics(y, y_hat, p) (tính RSS, TSS, $R^2$, $\bar{R}^2$, kiểm định F). *(Phần 1 | File: `part1/ols_implementation.py`)* 
* Code coef_inference(X, y, beta_hat, sigma2) (tính standard errors, t-statistics, p-values, khoảng tin cậy 95%). *(Phần 1 | File: `part1/ols_implementation.py`)* 
* Viết Unit Test (ít nhất 2 test/hàm) cho các hàm đo lường và kiểm định thống kê. *(Phần 1 | File: `part1/ols_implementation.py` và `part1_notebook.ipynb`)* 

## 3
* Hoàn thiện class DataPipeline (thêm Encoding và Chuẩn hóa Z-score). *(Phần 2 | File: `part2/data_pipeline.py`)* 
* Chia Train/Test Split trước (sử dụng random_state đã chốt). Sau đó fit DataPipeline chỉ trên tập Train và transform trên cả tập Train và tập Test (đảm bảo tuyệt đối không xảy ra hiện tượng rò rỉ dữ liệu - Data Leakage).

---

# Giai đoạn 3  (Deadline: 23h 22/5)

## 1
* Setup vòng lặp Monte Carlo để minh họa định lý Gauss-Markov: chứng minh tính không chệch ($\mathbb{E}[\hat{\beta}] = \beta$) và **chứng minh OLS có phương sai nhỏ nhất** bằng cách so sánh với một ước lượng tuyến tính không chệch khác. *(Phần 1 | File: `part1_notebook.ipynb`)*

## 2
* Code hàm vif(X) để kiểm tra đa cộng tuyến. *(Phần 1 | File: `part1/ols_implementation.py`)* 

## 4
* Code ridge_fit(X, y, lam) và thuật toán kfold_cv(X, y, k) **(đảm bảo hàm K-fold có trả về điểm CV - MSE trung bình)**. *(Phần 1 | File: `part1/ridge_lasso.py` và `part1/cross_validation.py`)* 
* Bổ sung chức năng vẽ biểu đồ Ridge Trace (sự thay đổi của hệ số theo $\lambda$). *(Phần 1 | File: `part1/ridge_lasso.py`)* 
* **Sử dụng k-fold CV (k=5 hoặc k=10) để tự động chọn siêu tham số $\lambda$ tốt nhất cho mô hình Ridge.** *(Phần 2 | File: `part2/model_comparison.py`)* 
* Áp dụng các mô hình này để train trên tập dữ liệu đã qua tiền xử lý của Dev 3. *(Phần 2 | File: `part2/model_comparison.py` và `part2_notebook.ipynb`)* 
* Viết Unit Test (ít nhất 2 test/hàm) cho ridge_fit và kfold_cv kiểm tra trên dữ liệu đã biết. (Phần 1 | File: part1/ridge_lasso.py và part1/cross_validation.py) *

## 5
* Nhận beta_hat từ Dev 1 & Dev 4 để code residual_plots(X, y, beta_hat) (vẽ đầy đủ 4 biểu đồ chẩn đoán: **Residuals vs Fitted, Q-Q Plot, Scale-Location, Cook's Distance và phải ghi nhận xét cho từng biểu đồ** cho tập dữ liệu giả lập Phần 1). *(Phần 1 | File: `part1/residual_analysis.py`)*

## 2 & 4 (Phối hợp)
* Code phần kỹ thuật nâng cao (Kernel Regression hoặc Bayesian Linear Regression) để lấy 0.5 điểm Bonus. *(Phần 2 | File: `part2/advanced_methods.py`)*

---

# Giai đoạn 4 (Deadline: 23h 26/5)

## 4
* **Triển khai OLS chọn biến bằng cách loại bỏ các biến độc lập dựa trên p-value cao hoặc VIF > 10.** *(Phần 2 | File: `part2/model_comparison.py`)*
* Chạy script so sánh $\ge 4$ mô hình trên tập Test (**OLS cơ bản, OLS chọn biến, Ridge với $\lambda$ tối ưu, Kernel Regression hoặc Bayesian Linear Regression **) và tổng hợp bảng MAE, RMSE, $R^2$. *(Phần 2 | File: `part2/model_comparison.py` và `part2_notebook.ipynb`)*

## 5
* Sử dụng code vẽ Feature Importance (biểu đồ hệ số hồi quy sau chuẩn hóa) để giải thích mô hình Phần 2. *(Phần 2 | File: `part2_notebook.ipynb`)* 
* Tái sử dụng hàm residual_plots để vẽ và phân tích 4 biểu đồ phần dư **áp dụng riêng cho mô hình có kết quả tốt nhất của Phần 2 (bắt buộc ghi nhận xét chẩn đoán lỗi mô hình dựa trên 4 đồ thị này)**. *(Phần 2 | File: `part2_notebook.ipynb`)*

---

# Final (Deadline: 23h 29/5)

## 1
* Bổ sung Markdown bên trong file part1_notebook.ipynb để giải thích rõ ràng các công thức toán học cốt lõi (nghiệm OLS, tính chất ma trận Hat) và viết phân tích chi tiết cho kết quả minh họa định lý Gauss-Markov (từ vòng lặp Monte Carlo) thay vì chỉ để code trần. (Phần 1 | File: part1_notebook.ipynb)*

## 2
* Bổ sung Markdown bên trong file part1_notebook.ipynb để diễn giải ý nghĩa của các bước kiểm định thống kê (F-test, t-test, khoảng tin cậy), giải thích các chỉ số đo lường (RSS, $R^2$, VIF) và ghi chú nhận xét trực tiếp dưới 4 biểu đồ phần dư của tập dữ liệu giả lập. (Phần 1 | File: part1_notebook.ipynb)*

## 3
* Soạn thảo phần mô tả rõ nguồn gốc dữ liệu và chứng minh dữ liệu thỏa mãn các điều kiện ($n \ge 200$, $p \ge 3$, biến mục tiêu liên tục, $\ge 5\%$ missing values) để đưa vào báo cáo. *(Phần 2 | File: `report.tex`)*
* Bổ sung Markdown bên trong file `part2_notebook.ipynb` để giải thích chi tiết quá trình Khảo sát dữ liệu (EDA), lập luận cơ chế Missing Value và luồng hoạt động của DataPipeline. *(Phần 2 | File: `part2_notebook.ipynb`)*

## 4
* Soạn thảo phần đánh giá kết quả, viết nhận xét và kết luận giải thích hiệu suất của các mô hình theo đúng ngữ cảnh thực tế của bộ dữ liệu để đưa vào báo cáo. *(Phần 2 | File: `report.tex`)*
* Bổ sung Markdown bên trong file `part2_notebook.ipynb` để giải thích quá trình chọn siêu tham số và so sánh các mô hình trên tập Test. *(Phần 2 | File: `part2_notebook.ipynb`)*

## 5
* Đóng gói source code theo đúng cấu trúc thư mục quy định (chia rõ part1/, part2/, data/). *(Chung | Folder: `Group_<ID>/`)* * Tạo file requirements.txt và viết README.md hướng dẫn chạy code chi tiết. *(Chung | File: `requirements.txt` và `README.md`)*
* Tổng hợp các bản nháp nhận xét từ Người 3 và 4 vào file báo cáo chính. Lập bảng phân công công việc chi tiết cho từng thành viên trong nhóm. *(Chung | File: `report.tex`)*
* Kiểm tra format tổng thể, hoàn thiện và xuất file `report.pdf` cuối cùng để nộp. *(Chung | File: `report.pdf`)*
* làm báo cáo hoàn chỉnh.