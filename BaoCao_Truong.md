# Báo cáo công việc Giai đoạn 1 & 2 - Người 3: Tiền xử lý dữ liệu (EDA & Data Pipeline)

---

## 1. Khảo sát dữ liệu (EDA) và Phân tích cơ chế Missing Value

Bộ dữ liệu được sử dụng là **Air Quality** (chất lượng không khí). Quá trình Khảo sát Dữ liệu (EDA) đã được thực hiện để đánh giá tổng quan về phân phối, mức độ tương quan và tình trạng khuyết thiếu dữ liệu. 

### 1.1. Phân tích cơ chế giá trị khuyết thiếu (Missing Values)
Dữ liệu từ các cảm biến đo lường thường xuyên bị thiếu sót. Qua phân tích, cơ chế missing value của bộ dữ liệu này được xác định chủ yếu là **MAR (Missing At Random - Khuyết thiếu ngẫu nhiên)**. 
- **Biện luận:** Sự cố của một cảm biến (ví dụ: cảm biến CO) thường không phụ thuộc vào chính nồng độ khí CO lúc đó, mà phụ thuộc vào các yếu tố ngoại cảnh khác như nhiệt độ hoặc độ ẩm quá cao/quá thấp khiến cảm biến bị lỗi tạm thời.
- **Chiến lược xử lý:** Vì lý do trên, phương pháp **Listwise Deletion (xóa dòng)** bị loại bỏ vì sẽ làm mất mát một lượng lớn thông tin quan trọng. Thay vào đó, nhóm quyết định áp dụng chiến lược **Imputation (Điền khuyết)** để bảo toàn dữ liệu. Các cột có tỉ lệ missing quá cao ($> 50\%$) mới bị loại bỏ hoàn toàn do độ tin cậy không còn đảm bảo.

### 1.2. Phân tích giá trị ngoại lai (Outliers)
Biểu đồ Boxplot và phân phối cho thấy dữ liệu cảm biến chứa rất nhiều giá trị ngoại lai (những thời điểm nồng độ khí tăng vọt bất thường). 
- **Chiến lược xử lý:** Nhóm quyết định không xóa các điểm ngoại lai này (vì sẽ mâu thuẫn với nỗ lực điền khuyết bảo toàn dữ liệu ở trên), mà sử dụng phương pháp **Winsorization (Cắt ngọn - IQR Capping)**. Ngoài ra, việc tồn tại nhiều outliers cũng là cơ sở để nhóm quyết định dùng **Median (Trung vị)** thay vì Mean (Trung bình) trong quá trình điền khuyết các biến số thực, do Median mang tính bền vững (robust) hơn trước outlier.

---

## 2. Thiết kế và Xây dựng Data Pipeline

Để đảm bảo tính nhất quán và tự động hóa, nhóm đã thiết kế class `DataPipeline` theo mô hình Hướng đối tượng, gồm 2 hàm cốt lõi là `fit()` và `transform()`. Pipeline thực hiện 6 bước tuần tự:

### Bước 1: Feature Engineering (Trích xuất đặc trưng thời gian)
Các cột `Date` và `Time` được chuyển đổi định dạng và trích xuất thành các biến có tính chu kỳ nhằm giúp mô hình học được yếu tố thời gian:
- `Date` $\rightarrow$ `Month` (Tháng), `DayOfWeek` (Ngày trong tuần).
- `Time` $\rightarrow$ `Hour` (Giờ).
Sau đó, các cột `Date` và `Time` gốc được loại bỏ.

### Bước 2: Loại bỏ các cột thiếu dữ liệu trầm trọng
Tính tỉ lệ missing cho từng cột. Cột nào có tỉ lệ missing $> 50\%$ (ngưỡng `missing_threshold`) sẽ được lưu vào danh sách `cols_to_drop` để loại bỏ hoàn toàn.

### Bước 3: Điền khuyết dữ liệu (Imputation)
- **Biến phân loại (Categorical):** Điền bằng giá trị xuất hiện nhiều nhất (Mode).
- **Biến thời gian (Month, DayOfWeek, Hour):** Điền bằng Median để đảm bảo kết quả là số nguyên.
- **Biến số thực liên tục:** Điền bằng Median (chiến lược mặc định) nhằm giảm thiểu sai lệch do ảnh hưởng của outliers.

### Bước 4: Xử lý Outlier bằng IQR Capping (Winsorization)
Sử dụng phương pháp Tứ phân vị (Interquartile Range) để giới hạn các giá trị cực đoan:
$$ IQR = Q_3 - Q_1 $$
$$ \text{Lower Bound} = Q_1 - 1.5 \times IQR $$
$$ \text{Upper Bound} = Q_3 + 1.5 \times IQR $$
Tất cả các giá trị vượt ra ngoài khoảng $[\text{Lower Bound}, \text{Upper Bound}]$ sẽ được gán (clip) bằng chính giá trị của 2 đầu mút này.

### Bước 5: Mã hóa biến phân loại (One-Hot Encoding)
Sử dụng `pd.get_dummies()` để tạo các biến dummy (0/1) cho các cột có kiểu dữ liệu là chuỗi (object). Quá trình này giúp mô hình tuyến tính có thể xử lý được các biến hạng mục.

### Bước 6: Chuẩn hóa dữ liệu (Z-score Standardization)
Đưa các biến về cùng một thang đo nhằm giúp các thuật toán (đặc biệt là Ridge) hội tụ tốt hơn và không bị chệch hệ số do sự chênh lệch đơn vị đo:
$$ x_{std} = \frac{x - \mu}{\sigma} $$

---

## 3. Ngăn chặn Rò rỉ dữ liệu (Data Leakage)

Thiết kế của `DataPipeline` tuân thủ nghiêm ngặt nguyên tắc chống rò rỉ dữ liệu (Data Leakage) – một trong những lỗi nghiêm trọng nhất trong Machine Learning:
1. **Quá trình `fit()` chỉ được gọi duy nhất trên tập Huấn luyện (Train set).** Toàn bộ các tham số thống kê bao gồm: danh sách cột cần xóa, giá trị median/mode để điền khuyết, biên giới hạn IQR (Lower/Upper bounds), tên các cột sau khi one-hot encoding, Mean ($\mu$) và Std ($\sigma$) đều được tính toán và **lưu trữ lại từ tập Train**.
2. **Quá trình `transform()` áp dụng cho cả Train và Test set.** Nó sử dụng lại nguyên xi các tham số đã học ở quá trình `fit()` để biến đổi tập Test. 
3. **Xử lý lệch cột One-hot:** Hàm `transform()` sử dụng `reindex(columns=self.encoding_columns, fill_value=0)` để ép tập Test phải có cấu trúc cột giống hệt tập Train, ngăn chặn lỗi thiếu/thừa cột dummy khi tập Test không chứa đủ các category.

---

## 4. Kết quả kiểm thử (Unit Tests)

Class `DataPipeline` đã được kiểm thử toàn diện thông qua script `part2/test_pipline.py` và đã **Pass 100% các bài test**:
- **Test rò rỉ cấu trúc:** Tập Train và Test sau khi qua pipeline có số lượng và cấu trúc cột giống hệt nhau.
- **Test Missing Values:** Đã triệt tiêu hoàn toàn NaN, Null và Vô cực (Infinity) trong dữ liệu.
- **Test Chuẩn hóa:** Trung bình (Mean) của các cột số thực xấp xỉ 0 và Độ lệch chuẩn (Std) xấp xỉ 1.
- **Test Outliers:** Không còn giá trị nào nằm ngoài biên giới hạn Winsorization.

Dữ liệu đầu ra là những ma trận hoàn toàn sạch, được chia tách theo đúng chuẩn và được trả về dưới dạng `2D list` (đã đính kèm cột hệ số chặn Intercept) để sẵn sàng đưa vào các thuật toán tự cài đặt OLS, Ridge và Bayesian.