# Quy trình Backtest và Đánh giá Mô hình Dự báo Nhu cầu (Phase 3)

Tài liệu này mô tả chi tiết phương pháp luận kiểm thử ngược (backtest) và các thước đo sai số được sử dụng để đánh giá công bằng các mô hình dự báo trong đề tài nghiên cứu.

---

## 1. Nguyên tắc thiết kế Backtest

Để mô phỏng chính xác điều kiện triển khai thực tế và ngăn ngừa rò rỉ dữ liệu tương lai (data leakage), nghiên cứu áp dụng kỹ thuật **Rolling-Origin với Expanding Window** (gốc thời gian trượt, cửa sổ huấn luyện mở rộng) trên giai đoạn hold-out 364 ngày:

- **Phạm vi dữ liệu**: 60 SKU đại diện tại cửa hàng `CA_1` từ tập dữ liệu M5 Walmart.
- **Tập hold-out**: $d_{1578}$ đến $d_{1941}$ (2015-05-25 đến 2016-05-22, đúng 52 tuần = 364 ngày).
- **Tần suất & Horizon**: Dự báo theo **NGÀY**, chu kỳ dự báo cố định $H = 28$ ngày (phù hợp với việc tính toán tồn kho theo lead time ngày ở Phase 5).
- **Số lượng fold**: $364 / 28 = 13$ fold liên tiếp, không chồng lấn giai đoạn đánh giá.

### Cấu hình từng Fold
Tại mỗi fold $k \in \{0, 1, \dots, 12\}$:
- Điểm cắt (origin): $o_k = 1577 + 28 \times k$.
- Tập huấn luyện (expanding history): từ ngày $d_1$ đến ngày $d_{o_k}$.
- Giai đoạn dự báo & đánh giá: từ ngày $d_{o_k + 1}$ đến ngày $d_{o_k + 28}$.
- Fold cuối cùng ($k = 12$): origin $o_{12} = 1913$, dự báo $d_{1914} \dots d_{1941}$ (kết thúc đúng ngày cuối cùng của bộ dữ liệu M5).

> **Quy tắc bất biến**: Mọi tinh chỉnh siêu tham số (hyperparameter tuning) của các mô hình học máy (Phase 3b) chỉ được sử dụng dữ liệu trong giai đoạn trước $d_{1577}$, tuyệt đối không dùng 13 fold hold-out để chọn tham số.

---

## 2. Các mô hình Baseline (Cơ sở so sánh)

Mọi mô hình học máy phức tạp (LightGBM/XGBoost) hay Time-Series Foundation Model (Chronos/TimesFM) đều phải được so sánh với 3 mô hình cơ sở:

1. **Naive**: Lấy giá trị bán thực tế của ngày cuối cùng trước origin ($t = o_k$) và lặp lại cho toàn bộ 28 ngày dự báo.
2. **Seasonal Naive (lag 7)**: Lấy giá trị bán của 7 ngày cuối cùng trước origin và lặp lại theo đúng thứ trong tuần qua 4 tuần tiếp theo.
3. **Moving Average (window 28)**: Tính trung bình số học lượng bán trong 28 ngày gần nhất trước origin, lặp lại giá trị trung bình này cho 28 ngày dự báo.

---

## 3. Hệ thống chỉ số đánh giá (Metrics)

Theo quy định tại Mục 2 của `AGENTS.md`, **cấm dùng MAPE** vì dữ liệu có 54.22% số ngày bán bằng 0 (gây lỗi chia cho 0). Hệ thống sử dụng 5 chỉ số:

| Chỉ số | Công thức | Ý nghĩa & Ứng dụng |
|---|---|---|
| **MAE** | $\frac{1}{N} \sum \|y_t - \hat{y}_t\|$ | Sai số tuyệt đối trung bình (đơn vị: sản phẩm/ngày). |
| **RMSE** | $\sqrt{\frac{1}{N} \sum (y_t - \hat{y}_t)^2}$ | Phạt nặng các sai số lớn; hữu ích khi chi phí thiếu hàng/tồn kho phi tuyến. |
| **WAPE** | $\frac{\sum \|y_t - \hat{y}_t\|}{\sum y_t}$ | Sai số phần trăm có trọng số; không bị lỗi chia cho 0 khi từng ngày riêng lẻ bằng 0. |
| **Bias** | $\frac{1}{N} \sum (\hat{y}_t - y_t)$ | Xu hướng lệch: $>0$ là dự báo thừa (tăng chi phí lưu kho), $<0$ là dự báo thiếu (tăng nguy cơ hết hàng). **Quy ước dấu**: $\text{dự báo} - \text{thực tế}$ ($\hat{y} - y$); dương = over-forecast, âm = under-forecast. |
| **MASE** | Xem chi tiết bên dưới | Sai số tuyệt đối chuẩn hóa theo thang đo Naive 1 bước trên tập train. $\text{MASE} < 1$ thể hiện mô hình tốt hơn Naive cơ bản. |

### Cách tính và gộp chỉ số MASE (Mean Absolute Scaled Error)
1. Với mỗi chuỗi SKU $i$ tại origin $o_k$, thang đo chuẩn hóa (in-sample scale) được tính trên toàn bộ lịch sử bán hàng đến $o_k$:
   $$\text{scale}_{i,k} = \frac{1}{T_k - 1} \sum_{t=2}^{T_k} |y_{i,t} - y_{i,t-1}|$$
   Nếu $\text{scale}_{i,k} = 0$ (chuỗi không biến thiên), hệ thống dừng và báo lỗi theo quy định.
2. MASE của chuỗi $i$ tại fold $k$:
   $$\text{MASE}_{i,k} = \frac{\text{MAE}_{i,k}}{\text{scale}_{i,k}} = \frac{\frac{1}{28} \sum_{h=1}^{28} |y_{i, o_k+h} - \hat{y}_{i, o_k+h}|}{\text{scale}_{i,k}}$$
3. **Cơ chế gộp MASE (Aggregation)**: MASE của một nhóm hoặc của toàn bộ 13 fold được tính bằng **trung bình số học (unweighted mean)** trên toàn bộ các cặp $(\text{origin}, \text{SKU})$:
   $$\text{MASE}_{\text{overall}} = \frac{1}{K \times M} \sum_{k=0}^{12} \sum_{i=1}^{60} \text{MASE}_{i,k}$$
   *Lưu ý cốt lõi*: Trong cách tính này, **mỗi SKU có trọng số ngang nhau** (1 phiếu bình đẳng unweighted), bất kể SKU đó bán được 100 sản phẩm/ngày hay chỉ 0.1 sản phẩm/ngày.

---

## 4. Các chiều phân tích kết quả

Kết quả dự báo trên 13 fold ($21.840$ điểm dữ liệu cho mỗi mô hình) được tổng hợp và phân tích theo 4 góc độ:

1. **Tổng thể (Overall)**: Đánh giá hiệu năng bình quân trên toàn bộ 60 SKU và 13 fold ($60 \times 28 \times 13 = 21.840$ lượt dự báo).
2. **Theo từng Fold (By Fold)**: Theo dõi độ ổn định của mô hình qua các mùa trong năm (13 fold trải dài từ tháng 5/2015 đến tháng 5/2016).
3. **Theo Ngành hàng (By Category)**: Phân tách riêng cho `FOODS` (20 SKU), `HOBBIES` (20 SKU), `HOUSEHOLD` (20 SKU).
4. **Theo Mức độ gián đoạn (By Zero-demand Tier)**: Chia 60 SKU thành 3 nhóm dựa trên tỷ lệ ngày bán bằng 0 trong tập train ($d_1 \dots d_{1577}$):
   - `smooth` ($< 33\%$ ngày bằng 0): **11 SKU** (tỷ lệ ngày 0 trung bình: 13.9%, doanh số trung bình cao: 11.1 sản phẩm/ngày).
   - `medium` ($33\% - 66\%$ ngày bằng 0): **29 SKU** (tỷ lệ ngày 0 trung bình: 50.4%, doanh số trung bình: 1.3 sản phẩm/ngày).
   - `intermittent` ($> 66\%$ ngày bằng 0): **20 SKU** (tỷ lệ ngày 0 trung bình: 81.9%, doanh số trung bình thấp: 0.3 sản phẩm/ngày).

---

## 5. Mô hình LightGBM (Phase 3b)

### Chiến lược dự báo

LightGBM được huấn luyện theo chiến lược **Direct Multi-Step Forecasting**: một mô hình duy nhất cho toàn bộ 60 SKU, dự báo trực tiếp 28 ngày (không đệ quy). Mỗi hàng trong tập huấn luyện là một bộ `(SKU, origin_t, horizon_h)` với $h = 1 \dots 28$; `horizon_h` được đưa vào làm feature riêng biệt để mô hình học được hình dạng dự báo khác nhau theo từng bước horizon.

### Tập Feature (21 features)

**Nhóm A — Tại origin t (chỉ dùng lịch sử $\le t$, đảm bảo không rò rỉ dữ liệu):**

| Feature | Mô tả |
|---|---|
| `lag_t0`, `lag_t6`, `lag_t13`, `lag_t27` | Doanh số tại ngày $t, t-6, t-13, t-27$ |
| `roll_mean_7/28/56`, `roll_std_7/28/56` | Trung bình & độ lệch chuẩn cuộn 7/28/56 ngày cuối tính đến $t$ |
| `zero_ratio_28` | Tỷ lệ ngày bán = 0 trong 28 ngày cuối tính đến $t$ |
| `days_since_last_sale` | Số ngày kể từ lần bán gần nhất trước $t$ |
| `cat_encoded` | Mã hóa số ngành hàng: `FOODS=0`, `HOBBIES=1`, `HOUSEHOLD=2` |

**Nhóm B — Ngày đích t+h (biết trước tại thời điểm dự báo — oracle features):**

| Feature | Mô tả |
|---|---|
| `horizon_h` | Bước dự báo ($1 \dots 28$) |
| `weekday_h`, `month_h`, `dayofmonth_h` | Thứ trong tuần ($0 \dots 6$), tháng ($1 \dots 12$), ngày trong tháng ($1 \dots 31$) |
| `has_event_h` | Cờ nhị phân sự kiện (1 nếu có sự kiện, 0 nếu không) |
| `snap_CA_h` | Cờ trợ cấp SNAP tại bang California |
| `sell_price_h` | Giá bán của SKU tại ngày đích |
| `price_ratio_h` | Tỷ lệ giá bán ngày đích so với giá bán tại origin: $\text{sell\_price}_h / \text{sell\_price}_t$ |

> [!IMPORTANT]
> **Giả định ngoại sinh biết trước (Oracle Features)**: Giá bán và lịch sự kiện của 28 ngày tiếp theo được giả định là đã biết trước tại thời điểm dự báo. Giả định này phù hợp với bối cảnh bán lẻ siêu thị (giá tuần và lịch sự kiện khuyến mãi được lên kế hoạch trước). Tuy nhiên, các mô hình baseline (Naive, SeasonalNaive, MovingAverage) **không sử dụng các biến ngoại sinh này**, nên đây là một lợi thế cấu trúc của LightGBM cần được ghi nhận minh bạch khi báo cáo so sánh.

### Quy trình Tuning Siêu Tham Số

- **Phạm vi dữ liệu**: Chỉ sử dụng dữ liệu trong giai đoạn trước $d_{1577}$. Tuyệt đối không dùng các fold hold-out.
- **2 fold nội bộ**:
  - Fold A: origin = 1521, giai đoạn validation: $d_{1522} \dots d_{1549}$.
  - Fold B: origin = 1549, giai đoạn validation: $d_{1550} \dots d_{1577}$.
  - Cả 2 origin đều thỏa $d \equiv 1577 \pmod 7$ (cùng thứ trong tuần) để đồng nhất với backtest chính.
- **Lưới tìm kiếm (8 cấu hình)**:
  - `objective`: `poisson` vs `tweedie` (với `tweedie_variance_power = 1.5`).
  - `num_leaves`: `[15, 31]`.
  - `learning_rate`: `[0.03, 0.05]`.
  - Cố định: `n_estimators = 500`, `min_child_samples = 20`, `seed = 42`.
- **Cách chọn origin huấn luyện**: Mỗi fold sử dụng **20 origin gần nhất** tính lùi từ origin với bước nhảy $\text{step} = 7$ ngày (đảm bảo đủ ít nhất 56 ngày lịch sử để tính rolling 56).
- **Kết quả tuning**: Cấu hình đạt WAPE trung bình 2 fold nhỏ nhất là **0.5997**:
  - `objective: tweedie` (`tweedie_variance_power: 1.5`)
  - `num_leaves: 31`
  - `learning_rate: 0.05`
  - Tham số được lưu cố định tại `configs/lightgbm.yaml`.

### Quy trình Backtest Hold-out (13 Folds)

Tại mỗi fold $k \in \{0, 1, \dots, 12\}$ trên tập hold-out:
1. **Lịch sử**: Dữ liệu expanding từ $d_1$ đến $d_{o_k}$ ($o_k = 1577 + 28k$).
2. **Quy tắc chọn origin huấn luyện**: Nhất quán hoàn toàn với quy trình tuning, mỗi fold hold-out sử dụng **20 origin gần nhất** (cách nhau 7 ngày) kết thúc tại $o_k - 28$, nhằm đảm bảo nhãn tương lai 28 ngày của mọi mẫu huấn luyện nằm trọn vẹn trong giai đoạn lịch sử trước $o_k$, loại trừ hoàn toàn nguy cơ rò rỉ dữ liệu.
3. **Huấn luyện lại**: Huấn luyện mới từ đầu (retrain from scratch) với siêu tham số cố định từ `configs/lightgbm.yaml`.
4. **Dự báo**: Dự báo 28 ngày với `future_context_df`. Giá trị dự báo được cắt chặn dưới $\ge 0$ (clip), không làm tròn.
5. **Kiểm tra an toàn**: Nếu WAPE tổng thể $< 0.3$, script tự động dừng và cảnh báo nghi ngờ rò rỉ.

---

## 6. So sánh Kết quả Thực nghiệm: Baselines vs LightGBM vs Chronos

### Bảng tổng hợp hiệu năng 5 mô hình (Hold-out 13 Folds, 21.840 lượt dự báo)

| Mô hình | Loại mô hình | WAPE | MASE | MAE | RMSE | Bias | Số fold thắng MA (WAPE) | Số fold thắng MA (MASE) |
|---|---|---|---|---|---|---|---|---|
| **Chronos** | TSFM (Zero-shot) | **0.6269** | **0.7850** | **1.5340** | 4.7433 | -0.5627 | **12 / 13** | **13 / 13** |
| **LightGBM** | Machine Learning | 0.6476 | 0.9182 | 1.5845 | **4.5063** | -0.1419 | 11 / 13 | 6 / 13 |
| **MovingAverage** | Baseline (28d) | 0.7306 | 0.9179 | 1.7877 | 5.5607 | -0.0079 | 0 / 13 | 0 / 13 |
| **SeasonalNaive** | Baseline (lag 7) | 0.7908 | 1.0270 | 1.9349 | 5.6084 | -0.1277 | 3 / 13 | 0 / 13 |
| **Naive** | Baseline (lag 1) | 0.8843 | 1.1382 | 2.1637 | 6.4648 | +0.2866 | 2 / 13 | 0 / 13 |

---

## 7. Phân tích Chi tiết Kết quả Thực nghiệm theo Zero-demand Tier

| Nhóm | Số SKU | Mô hình | WAPE | MASE | MAE | RMSE | Bias |
|---|---|---|---|---|---|---|---|
| **`smooth`** | 11 | **LightGBM** | **0.5210** | **0.7991** | **5.0870** | **10.1119** | -0.8460 |
| | | Chronos | 0.5225 | 0.8200 | 5.1022 | 10.6056 | -1.3755 |
| | | MovingAverage | 0.6364 | 0.8883 | 6.2138 | 12.6459 | -0.0322 |
| | | SeasonalNaive | 0.6570 | 1.0210 | 6.4148 | 12.5350 | -0.6526 |
| | | Naive | 0.7305 | 1.1001 | 7.1331 | 14.3682 | +0.8089 |
| **`medium`** | 29 | **Chronos** | **0.8922** | **0.7832** | **1.0312** | 1.8785 | -0.4506 |
| | | LightGBM | 0.9167 | 0.8026 | 1.0595 | **1.7178** | +0.0175 |
| | | MovingAverage | 0.9169 | 0.8068 | 1.0597 | 1.7411 | -0.0001 |
| | | SeasonalNaive | 1.1016 | 0.9658 | 1.2733 | 2.2310 | -0.0156 |
| | | Naive | 1.2478 | 1.0682 | 1.4423 | 2.7599 | +0.2447 |
| **`intermittent`** | 20 | **Chronos** | **1.0224** | **0.7685** | **0.3004** | 0.7197 | -0.2781 |
| | | MovingAverage | 1.3918 | 1.0952 | 0.4089 | 0.6436 | -0.0058 |
| | | LightGBM | 1.4269 | 1.1514 | 0.4193 | **0.6355** | +0.0141 |
| | | SeasonalNaive | 1.4647 | 1.1192 | 0.4304 | 0.8522 | -0.0015 |
| | | Naive | 1.6218 | 1.2607 | 0.4765 | 0.8883 | +0.0600 |

### Nhận xét khoa học:
1. **Chronos (Zero-shot)**: Thể hiện năng lực khái quát hóa mạnh mẽ vượt bậc trên các chuỗi nhu cầu gián đoạn (`intermittent`), hạ MASE xuống **0.7685** (so với 1.0952 của MovingAverage và 1.1514 của LightGBM). Tuy nhiên, Chronos có xu hướng **under-forecasting** rõ rệt (Bias overall là -0.5627), điều này cần đặc biệt lưu tâm khi tính toán tồn kho an toàn ở Phase 5 để tránh nguy cơ stockout.
2. **LightGBM**: Duy trì ưu thế tốt nhất ở nhóm hàng bán nhanh (`smooth`) nhờ tận dụng được các biến ngoại sinh (giá bán và lịch sự kiện khuyến mãi SNAP), đạt RMSE thấp nhất trong tất cả các mô hình (**4.5063**). Bias của LightGBM (-0.1419) ổn định hơn nhiều so với Chronos.
3. **Cơ chế MASE vs WAPE**: Chronos thắng áp đảo về MASE (13/13 fold) vì giải quyết triệt để 20 SKU intermittent chiếm số lượng lớn. Trong khi đó, ở WAPE, mức độ cải thiện của Chronos và LightGBM khá sát nhau (0.6269 so với 0.6476) do WAPE chịu ảnh hưởng chủ đạo từ nhóm hàng `smooth`.
