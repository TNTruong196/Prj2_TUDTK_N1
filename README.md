# Đồ Án 2 — Data Fitting và Phương Pháp OLS

<div align="center">

**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051) &nbsp;|&nbsp; **Học kỳ 2, 2025–2026**

**Trường Đại học Khoa học Tự nhiên — ĐHQG TP. Hồ Chí Minh**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)

</div>

---

## Mục lục

- [Đồ Án 2 — Data Fitting và Phương Pháp OLS](#đồ-án-2--data-fitting-và-phương-pháp-ols)
  - [Mục lục](#mục-lục)
  - [Giới thiệu](#giới-thiệu)
  - [Thông tin nhóm](#thông-tin-nhóm)
  - [Cấu trúc thư mục](#cấu-trúc-thư-mục)
  - [🔧 Yêu cầu môi trường](#-yêu-cầu-môi-trường)
  - [Cài đặt](#cài-đặt)
  - [Hướng dẫn chạy](#hướng-dẫn-chạy)
    - [Notebook (khuyến nghị)](#notebook-khuyến-nghị)
    - [Chạy script so sánh mô hình (Phần 2)](#chạy-script-so-sánh-mô-hình-phần-2)
    - [Chạy từng module độc lập](#chạy-từng-module-độc-lập)
  - [Tổng quan kỹ thuật](#tổng-quan-kỹ-thuật)
    - [Phần 1 — Các hàm tự cài đặt](#phần-1--các-hàm-tự-cài-đặt)
    - [Phần 2 — Pipeline và so sánh mô hình](#phần-2--pipeline-và-so-sánh-mô-hình)
  - [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
  - [Quy ước chung](#quy-ước-chung)
  - [Kiểm thử (Unit Tests)](#kiểm-thử-unit-tests)
  - [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Đồ án thực hiện **tự cài đặt** các thuật toán hồi quy tuyến tính từ công thức toán học, không phụ thuộc vào thư viện ML (NumPy/sklearn chỉ dùng để kiểm chứng). Dự án gồm hai phần:

- **Phần 1 — Lý thuyết & Minh họa:** Cài đặt OLS, Ridge, Lasso, k-fold CV, VIF, phân tích phần dư và mô phỏng Monte Carlo (định lý Gauss–Markov). Toàn bộ dùng **Python 2D list** thuần.
- **Phần 2 — Ứng dụng thực tế:** Áp dụng pipeline tiền xử lý và so sánh 5 mô hình (OLS, OLS chọn biến, Ridge, Lasso, Bayesian Linear Regression) trên bộ dữ liệu **Air Quality** thực tế.

---

## Thông tin nhóm

| Họ và tên | MSSV |
|---|---|
| Trần Nhật Trường | 24120486 |
| Nguyễn Thanh Nhật | 24120111 |
| Nguyễn Ngọc Phúc | 24120215 |
| Trần Lê Đức Việt | 24120245 |
| Trần Nguyên Tân | 24120438 |

**Giảng viên hướng dẫn:** ThS. Lê Nhựt Nam &nbsp;|&nbsp; ThS. Võ Nam Thục Đoan

---

## Cấu trúc thư mục

```
Prj2_TUDTK_N1/
├── README.md
├── requirements.txt
│
├── anh_log_giaithich/              # Thư mục chứa log, đồ thị và giải thích kiểm thử
│   ├── plots/                      # Đồ thị phân tích dữ liệu, chẩn đoán mô hình
│   ├── logs/                       # Log terminal lưu các chỉ số đánh giá
│   └── giai_thich_chi_tiet/        # Tài liệu giải thích kịch bản test cho 16 hàm
│
├── part1/                          # Phần 1: Cài đặt thuật toán từ công thức
│   ├── matrix_helper.py            # Helper: nhân, chuyển vị, nghịch đảo ma trận (2D list)
│   ├── ols_implementation.py       # ols_fit, hat_matrix, model_metrics, coef_inference, vif
│   ├── ridge_lasso.py              # ridge_fit, lasso_fit (Coordinate Descent), ridge/lasso trace
│   ├── cross_validation.py         # kfold_cv (k-fold Cross-Validation)
│   ├── residual_analysis.py        # residual_plots (4 biểu đồ chẩn đoán)
│   └── part1_notebook.ipynb        # Notebook tổng hợp Phần 1
│
├── part2/                          # Phần 2: Ứng dụng dữ liệu thực
│   ├── data/
│   │   └── AirQuality.csv          # Bộ dữ liệu Air Quality (UCI ML Repository)
│   ├── data_pipeline.py            # class DataPipeline (fit/transform, imputation, encoding, Z-score)
│   ├── model_comparison.py         # So sánh 5 mô hình, chọn λ bằng CV, đánh giá MAE/RMSE/R²
│   ├── advanced_methods.py         # Bayesian Linear Regression (phần nâng cao / bonus)
│   └── part2_notebook.ipynb        # Notebook tổng hợp Phần 2
│
└── report/
    ├── report.tex                  # Báo cáo LaTeX
    └── report.pdf                  # Báo cáo xuất PDF
```

---

## 🔧 Yêu cầu môi trường

- **Python:** 3.10 trở lên
- **Jupyter Notebook** hoặc **VS Code** (với tiện ích mở rộng Jupyter)

Các gói thư viện cần thiết:

| Thư viện | Mục đích |
|---|---|
| `pandas` | Đọc và xử lý dữ liệu |
| `numpy` | Kiểm chứng kết quả (không dùng cho thuật toán chính) |
| `matplotlib` | Vẽ đồ thị |
| `seaborn` | Trực quan hóa EDA |
| `scipy` | Phân phối thống kê (t, F) để kiểm chứng |
| `scikit-learn` | Chia train/test reproducible, so sánh kết quả |
| `statsmodels` | Kiểm chứng thống kê |
| `ipykernel` | Chạy Jupyter Notebook |

---

## Cài đặt

**Bước 1:** Clone repository về máy:

```bash
git clone https://github.com/TNTruong196/Prj2_TUDTK_N1.git
cd Prj2_TUDTK_N1
```

**Bước 2 (khuyến nghị):** Tạo môi trường ảo:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**Bước 3:** Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

---

## Hướng dẫn chạy

### Notebook (khuyến nghị)

Mở và chạy lần lượt các cell trong Jupyter / VS Code:

```bash
# Phần 1 — Lý thuyết và minh họa
jupyter notebook part1/part1_notebook.ipynb

# Phần 2 — Ứng dụng thực tế
jupyter notebook part2/part2_notebook.ipynb
```

### Chạy script so sánh mô hình (Phần 2)

```bash
python part2/model_comparison.py
```

Script sẽ tự động:
1. Đọc dữ liệu `part2/data/AirQuality.csv`
2. Chia train/test (80/20, `random_state=42`)
3. Fit `DataPipeline` trên tập train, transform cả train và test
4. Chọn `λ` tối ưu cho **Ridge** và **Lasso** bằng 5-fold CV
5. Huấn luyện 5 mô hình: OLS, OLS chọn biến, Ridge, Lasso, Bayesian LR
6. In bảng **MAE / RMSE / R²** trên tập test

### Chạy từng module độc lập

```bash
# Phần 1
python part1/matrix_helper.py
python part1/ols_implementation.py
python part1/ridge_lasso.py
python part1/cross_validation.py
python part1/residual_analysis.py

# Phần 2
python part2/data_pipeline.py
python part2/advanced_methods.py
```

---

## Tổng quan kỹ thuật

### Phần 1 — Các hàm tự cài đặt

| Hàm | File | Mô tả |
|---|---|---|
| `ols_fit(X, y)` | `ols_implementation.py` | Tính `β̂ = (XᵀX)⁻¹Xᵀy` và `σ̂²` |
| `hat_matrix(X)` | `ols_implementation.py` | Ma trận chiếu `H = X(XᵀX)⁻¹Xᵀ`, kiểm tra idempotent |
| `model_metrics(y, y_hat, p)` | `ols_implementation.py` | RSS, TSS, R², R̄², F-test |
| `coef_inference(X, y, β, σ²)` | `ols_implementation.py` | Standard errors, t-statistics, p-values, CI 95% |
| `vif(X)` | `ols_implementation.py` | Variance Inflation Factor cho từng biến |
| `ridge_fit(X, y, λ)` | `ridge_lasso.py` | `β̂_ridge = (XᵀX + λI')⁻¹Xᵀy` |
| `lasso_fit(X, y, λ)` | `ridge_lasso.py` | Coordinate Descent với soft-threshold |
| `kfold_cv(X, y, k)` | `cross_validation.py` | k-fold CV, hỗ trợ custom `fit_func` |
| `residual_plots(X, y, β)` | `residual_analysis.py` | 4 biểu đồ: Residuals vs Fitted, Q-Q, Scale-Location, Cook's Distance |

> **Lưu ý:** Toàn bộ các hàm trên dùng Python 2D list thuần (`list[list[float]]`). NumPy/sklearn chỉ được gọi trong phần kiểm chứng.

### Phần 2 — Pipeline và so sánh mô hình

**`DataPipeline`** thực hiện 6 bước theo thứ tự, chỉ `fit` trên tập train để tránh data leakage:

1. **Feature engineering** — trích xuất `Month`, `DayOfWeek`, `Hour` từ `Date/Time`
2. **Loại cột missing** — loại bỏ cột có tỉ lệ thiếu > 50%
3. **Imputation** — median cho biến số thực, mode cho biến phân loại
4. **IQR Capping** (Winsorization) — xử lý giá trị ngoại lai
5. **One-hot encoding** + reindex — đảm bảo train/test đồng cấu trúc cột
6. **Z-score standardization** — `x_std = (x − μ) / σ`

---

## Kết quả thực nghiệm

**Bộ dữ liệu:** Air Quality — [UCI ML Repository](https://archive.ics.uci.edu/dataset/360/air+quality)  
**Target:** `C6H6(GT)` (nồng độ benzene)  
**Cấu hình:** `random_state=42`, chia train/test 80/20, chọn λ bằng 5-fold CV

| Mô hình | MAE | RMSE | R² |
|---|---:|---:|---:|
| OLS (Đầy đủ biến) | 0.8900 | 1.4461 | 0.9636 |
| OLS (Chọn biến) | 1.4285 | 2.0726 | 0.9252 |
| Ridge (λ = 0.1) | 0.8900 | 1.4461 | 0.9636 |
| **Lasso (λ = 0.0001)** | **0.8899** | **1.4460** | **0.9636** |
| Bayesian Linear Regression | 0.8900 | 1.4461 | 0.9636 |

> Mô hình **Lasso (λ = 0.0001)** đạt hiệu suất tốt nhất với MAE và RMSE thấp nhất. Tuy nhiên, chênh lệch hiệu năng giữa Ridge, Lasso, Bayesian LR và OLS là rất nhỏ do quan hệ tuyến tính của dữ liệu gốc rất mạnh (R² ≈ 0.964). Ở chiều ngược lại, OLS chọn biến (loại bỏ `PT08.S2(NMHC)` có VIF = 31.7 và `T` có VIF = 14.5 để giảm đa cộng tuyến) làm giảm hiệu suất đáng kể do loại đi các biến mang tín hiệu dự báo mạnh.

---

## Quy ước chung

| Quy ước | Giá trị |
|---|---|
| Random seed (toàn dự án) | `42` |
| Target column | `C6H6(GT)` |
| Train / Test split | 80% / 20% |
| k-fold CV | k = 5 |
| VIF threshold (chọn biến) | > 10 |
| Missing threshold (pipeline) | > 50% |
| Biểu diễn ma trận | Python 2D list (`list[list[float]]`) |

---

## Kiểm thử (Unit Tests)

Mỗi hàm tự cài đặt có **ít nhất 2 unit test** kiểm tra trên dữ liệu đã biết, đối chiếu với NumPy/sklearn.

```bash
# Chạy toàn bộ unit test Phần 1
python -m unittest part1.matrix_helper
python -m unittest part1.ols_implementation
python -m unittest part1.ridge_lasso
python -m unittest part1.cross_validation
python -m unittest part1.residual_analysis

# Chạy toàn bộ unit test Phần 2
python -m unittest part2.data_pipeline
python -m unittest part2.advanced_methods
python -m unittest part2.model_comparison
```

---

## Tài liệu tham khảo

1. Gilbert Strang. *Introduction to Linear Algebra*, 6th ed. Wellesley-Cambridge Press, 2023.
2. James, Witten, Hastie & Tibshirani. *An Introduction to Statistical Learning*, 2nd ed. Springer, 2021. — https://www.statlearning.com
3. Hastie, Tibshirani & Friedman. *The Elements of Statistical Learning*, 2nd ed. Springer, 2009. — https://hastie.su.domains/ElemStatLearn/
4. Christopher M. Bishop. *Pattern Recognition and Machine Learning*. Springer, 2006.
5. Kevin P. Murphy. *Probabilistic Machine Learning: An Introduction*. MIT Press, 2022. — https://probml.github.io/pml-book/book1.html

---

<div align="center">

*Khoa Công nghệ Thông tin — Trường Đại học Khoa học Tự nhiên TP. HCM*  
*Học kỳ 2, 2025–2026*

</div>
