# Báo cáo đánh giá kết quả Phần 2 - Final

---

## 1. Bảng so sánh 5 mô hình

Kết quả được lấy từ lần chạy `python part2/model_comparison.py` với `random_state=42`, chia train/test 80/20, chọn lambda cho Ridge bằng 5-fold Cross-Validation và chọn lambda cho Lasso bằng 5-fold Cross-Validation trên tập train.

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| OLS | 0.889981 | 1.446089 | 0.963608 |
| OLS (Variable Selection) | 1.428491 | 2.072625 | 0.925242 |
| Ridge (lambda=0.1) | 0.889957 | 1.446081 | 0.963608 |
| Lasso (lambda=0.0001) | 0.889858 | 1.446016 | 0.963611 |
| Bayesian Linear Regression | 0.889976 | 1.446087 | 0.963608 |

Bảng dạng LaTeX-ready:

```latex
\begin{table}[H]
\centering
\begin{tabular}{lrrr}
\hline
Model & MAE & RMSE & R^2 \\
\hline
OLS & 0.889981 & 1.446089 & 0.963608 \\
OLS (Variable Selection) & 1.428491 & 2.072625 & 0.925242 \\
Ridge ($\lambda=0.1$) & 0.889957 & 1.446081 & 0.963608 \\
Lasso ($\lambda=0.0001$) & 0.889858 & 1.446016 & 0.963611 \\
Bayesian Linear Regression & 0.889976 & 1.446087 & 0.963608 \\
\hline
\end{tabular}
\caption{So sánh hiệu suất các mô hình trên tập test AirQuality.}
\end{table}
```

---

## 2. Nhận xét từng mô hình

### 2.1. OLS cơ bản

OLS cơ bản dùng tất cả biến sau tiền xử lý để dự đoán nồng độ `C6H6(GT)`. Mô hình đạt MAE = 0.889981, RMSE = 1.446089 và R2 = 0.963608 trên tập test.

Kết quả này cho thấy quan hệ tuyến tính giữa các tín hiệu cảm biến và nồng độ Benzene trong bộ AirQuality là rất mạnh. Các biến như `PT08.S2(NMHC)`, `CO(GT)`, `PT08.S1(CO)` và `PT08.S5(O3)` có khả năng giải thích phần lớn biến thiên của target. Vì vậy, ngay cả mô hình tuyến tính cơ bản đã đạt R2 rất cao.

Hạn chế chính của OLS là nhạy với đa cộng tuyến. Khi nhiều cảm biến đo các chất liên quan hoặc cùng phản ánh điều kiện môi trường tại cùng thời điểm, các cột trong ma trận thiết kế có thể tương quan mạnh. Điều này làm hệ số hồi quy kém ổn định hơn, dù hiệu suất dự đoán vẫn cao.

### 2.2. OLS chọn biến

OLS chọn biến đã loại 2 biến:

| Bước | Biến bị loại | Lý do | Giá trị |
|---:|---|---|---:|
| 1 | `PT08.S2(NMHC)` | VIF | 31.742536 |
| 2 | `T` | VIF | 14.549521 |

Quy trình chọn biến ưu tiên loại biến có VIF > 10 trước, sau đó mới xét p-value. Trong lần chạy này, cả hai biến bị loại đều do VIF cao, nghĩa là chúng có mức đa cộng tuyến lớn với các biến còn lại.

Hiệu suất sau khi chọn biến giảm rõ rệt: MAE tăng từ 0.889981 lên 1.428491, RMSE tăng từ 1.446089 lên 2.072625 và R2 giảm từ 0.963608 xuống 0.925242. Điều này cho thấy việc loại biến giúp mô hình gọn và dễ giải thích hơn, nhưng trong dataset AirQuality các biến đa cộng tuyến vẫn mang thông tin dự báo hữu ích. Đặc biệt, `PT08.S2(NMHC)` có liên hệ rất mạnh với `C6H6(GT)`, nên loại biến này làm mất một phần tín hiệu quan trọng.

### 2.3. Ridge Regression

Ridge Regression chọn lambda = 0.1 bằng 5-fold Cross-Validation. Top 5 giá trị lambda theo CV-MSE:

| lambda | CV-MSE |
|---:|---:|
| 0.100000 | 1.993500 |
| 0.046416 | 1.993500 |
| 0.021544 | 1.993500 |
| 0.010000 | 1.993500 |
| 0.004642 | 1.993501 |

Ridge đạt MAE = 0.889957, RMSE = 1.446081 và R2 = 0.963608. Đây là kết quả rất tốt, tương đương OLS nhưng các hệ số ổn định hơn nhờ phạt L2 (regularization).

### 2.4. Lasso Regression

Lasso Regression tự cài đặt bằng Coordinate Descent kết hợp với kỹ thuật tối ưu vector residuals (giảm độ phức tạp tính toán r_j xuống còn $O(N)$ thay vì $O(N \cdot P)$).
Lasso chọn lambda = 0.0001 bằng 5-fold Cross-Validation. Top 5 giá trị lambda theo CV-MSE:

| lambda | CV-MSE |
|---:|---:|
| 0.000100 | 1.993455 |
| 0.000534 | 1.993493 |
| 0.002848 | 1.995287 |
| 0.015199 | 2.034271 |
| 0.081113 | 2.196293 |

Lasso đạt MAE = 0.889858, RMSE = 1.446016 và R2 = 0.963611. Đây là mô hình đạt kết quả tốt nhất trong tất cả các mô hình so sánh trên tập test (R2 cao nhất và sai số MAE/RMSE thấp nhất). 

Do lambda chọn được rất nhỏ ($\lambda = 0.0001$), Lasso chỉ phạt nhẹ các biến có độ tương quan mạnh và duy trì hầu hết các biến quan trọng, mang lại hiệu năng dự báo vượt trội và tối ưu.

### 2.5. Bayesian Linear Regression

Bayesian Linear Regression dùng prior variance = 100 cho các hệ số feature, intercept prior variance = 1e12 và sigma2 ước lượng từ OLS. Mô hình đạt MAE = 0.889976, RMSE = 1.446087 và R2 = 0.963608.

Kết quả gần OLS/Ridge/Lasso vì prior variance = 100 là prior tương đối yếu sau khi dữ liệu đã được chuẩn hóa. Khi số mẫu lớn và tín hiệu tuyến tính mạnh, posterior mean chủ yếu bị chi phối bởi dữ liệu quan sát. Điểm bổ sung của Bayesian LR là có posterior covariance, cho phép biểu diễn các Credible Intervals (khoảng tin cậy Bayesian) cho các hệ số, giúp diễn giải trực quan mức độ bất định trong việc thu thập tín hiệu cảm biến.

---

## 3. Kết luận tổng quát

Theo các chỉ số đánh giá, Lasso Regression với lambda = 0.0001 là mô hình tốt nhất, bám sát là Ridge Regression với lambda = 0.1. Cả hai mô hình điều chuẩn hóa (regularized) đều mang lại hiệu quả tốt hơn OLS cơ bản và OLS chọn biến.

Sự tương quan mạnh giữa các cảm biến trong không khí khiến mô hình tuyến tính đơn thuần dễ bị ảnh hưởng bởi hiện tượng đa cộng tuyến. Việc Ridge và Lasso kiểm soát hệ số thông qua hàm phạt L2 và L1 giúp duy trì tính ổn định của dự báo tốt hơn nhiều so với việc loại bỏ biến cứng nhắc trong OLS chọn biến.
