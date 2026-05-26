# Báo cáo đánh giá kết quả Phần 2 - Final

---

## 1. Bảng so sánh 4 mô hình

Kết quả được lấy từ lần chạy `python part2/model_comparison.py` với `random_state=42`, chia train/test 80/20 và chọn lambda Ridge bằng 5-fold Cross-Validation trên tập train.

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| OLS | 0.889981 | 1.446089 | 0.963608 |
| OLS (Variable Selection) | 1.428491 | 2.072625 | 0.925242 |
| Ridge (lambda=0.1) | 0.889957 | 1.446081 | 0.963608 |
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

Ridge đạt MAE = 0.889957, RMSE = 1.446081 và R2 = 0.963608. Đây là kết quả tốt nhất theo MAE và RMSE, nhưng mức cải thiện so với OLS rất nhỏ.

Lý do hợp lý là OLS đã fit rất tốt trên dataset này, trong khi Ridge chỉ thêm regularization L2 để co nhỏ hệ số. Ridge không loại biến như OLS chọn biến, mà giữ lại toàn bộ tín hiệu cảm biến và giảm variance của hệ số. Đây là bias-variance tradeoff: mô hình chấp nhận một lượng bias nhỏ từ regularization để đổi lấy hệ số ổn định hơn, đặc biệt khi các cảm biến có tương quan mạnh.

### 2.4. Bayesian Linear Regression

Bayesian Linear Regression dùng prior variance = 100 cho các hệ số feature, intercept prior variance = 1e12 và sigma2 ước lượng từ OLS. Mô hình đạt MAE = 0.889976, RMSE = 1.446087 và R2 = 0.963608.

Kết quả gần OLS/Ridge vì prior variance = 100 là prior tương đối yếu sau khi dữ liệu đã được chuẩn hóa. Khi số mẫu lớn và tín hiệu tuyến tính mạnh, posterior mean chủ yếu bị chi phối bởi dữ liệu quan sát, nên dự đoán gần với nghiệm OLS. Điểm bổ sung của Bayesian LR là có posterior covariance, từ đó có thể diễn giải độ bất định của hệ số thay vì chỉ đưa ra một ước lượng điểm.

---

## 3. Kết luận tổng quát

Theo MAE và RMSE, Ridge Regression với lambda = 0.1 là mô hình tốt nhất, nhưng khoảng cách so với OLS và Bayesian Linear Regression gần như không đáng kể. Về mặt thực tế, cả ba mô hình OLS, Ridge và Bayesian LR đều phù hợp để dự đoán nồng độ Benzene `C6H6(GT)` trên bộ AirQuality.

Dataset AirQuality có đặc trưng là nhiều biến cảm biến đo các thành phần khí và điều kiện môi trường có tương quan cao với nhau. Điều này tạo ra hai hiện tượng cùng lúc:

- Quan hệ tuyến tính với target rất mạnh, giúp OLS đạt R2 cao.
- Đa cộng tuyến giữa các biến đầu vào, làm hệ số OLS có thể kém ổn định.

Ridge là lựa chọn cân bằng nhất trong bộ so sánh này vì xử lý đa cộng tuyến bằng cách co nhỏ hệ số thay vì loại bỏ tín hiệu. OLS chọn biến hữu ích nếu mục tiêu ưu tiên mô hình gọn và giải thích qua ít biến hơn, nhưng kết quả test cho thấy việc loại `PT08.S2(NMHC)` và `T` làm mất đáng kể độ chính xác dự báo.

Hạn chế của thực nghiệm hiện tại là các mô hình vẫn chủ yếu nằm trong họ tuyến tính. Hướng cải thiện có thể gồm: kiểm tra residual plots cho mô hình tốt nhất, thử thêm biến tương tác hoặc biến phi tuyến có kiểm soát, đánh giá độ ổn định của chọn biến qua nhiều split, và so sánh thêm các mô hình regularization khác nếu mục tiêu là tăng khả năng dự báo.

---
