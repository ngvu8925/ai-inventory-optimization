# Định nghĩa chính sách tồn kho & chỉ số mô phỏng (Phase 5)

Tài liệu này chốt định nghĩa đang dùng trong `src/inventory/`. Mọi tham số lấy từ
`configs/assumptions.yaml`; các giá trị chi phí hiện là **giả định tạm, chưa được
nhóm duyệt**.

---

## 1. Các chính sách

Ký hiệu: `L = lead_time_days` (mặc định 7), `z = Φ⁻¹(service_level)` (tính tự động,
không hard-code), `q_days = reorder_quantity_days` (mặc định 14).

### 1.1. `Baseline` (chính, dùng cửa sổ gần)

- Chỉ dùng dữ liệu **đến hết ngày origin** của từng fold.
- Cửa sổ: `baseline_window_days` ngày gần nhất (mặc định 91).
- `d̄ = mean(sales trong cửa sổ)`
- `σ = std(sales trong cửa sổ, ddof=1)`
- `SS = z × σ × √L`
- `ROP = d̄ × L + SS`
- `Q = d̄ × q_days`

### 1.2. `Baseline_full_history` (đối chiếu, không phải chỉ số chính)

Giống `Baseline` nhưng cửa sổ = toàn bộ lịch sử từ `d_1` đến origin. Giữ lại để
đo ảnh hưởng của việc chọn cửa sổ; không dùng để kết luận "baseline tốt hơn/kém hơn".

### 1.3. `point_forecast`

- `d̄ = mean(forecast[1..L])` — trung bình điểm dự báo (median) trong L ngày đầu.
- `σ = std(toàn bộ lịch sử đến origin, ddof=1)` — **chưa** dùng sai số dự báo.
- `SS = z × σ × √L`; `ROP = d̄ × L + SS`.
- `Q = kỳ vọng nhu cầu trong q_days ngày` (tổng forecast điểm của q_days ngày đầu;
  nếu horizon ngắn hơn q_days thì dùng trung bình × q_days).
- Với kỳ vọng, tổng của các ngày bằng tổng kỳ vọng từng ngày nên phép cộng hợp lệ.

### 1.3.1. `point_forecast_window_sigma` — biến thể (a), tách nguồn σ

Giống `point_forecast` ở phần ROP kỳ vọng và Q (lấy từ dự báo điểm), nhưng:

```text
σ = std(sales trong baseline_window_days ngày gần nhất, ddof=1)   # mặc định 91
SS = z × σ × √L
ROP = mean(forecast[1..L]) × L + SS
```

Mục đích: giữ nguyên vị trí ROP theo dự báo, chỉ đổi nguồn σ sang đúng cửa sổ
91 ngày như Baseline. Nhờ vậy đo riêng phần chênh lệch do nguồn σ.

### 1.3.2. `point_forecast_error_sigma` — biến thể (b), σ từ sai số dự báo

Giống `point_forecast` nhưng σ là độ lệch chuẩn của **sai số đã quan sát**:

```text
với fold k và SKU i:
  errors = { actual − forecast } của SKU i ở TẤT CẢ các fold < k
  σ_error = std(errors, ddof=1)          nếu có ≥ 2 quan sát
  σ_error = std(91 ngày gần nhất trước origin)   nếu chưa đủ (fold 0)
```

- Chỉ dùng dữ liệu đã quan sát **trước origin** của fold hiện tại → không rò rỉ
  tương lai. Fold 0 chưa có fold trước nên dùng fallback σ 91 ngày; điều này
  được ghi rõ trong `results/inventory/forecast_error_sigma.csv`
  (`fallback_used = True`, `n_error_obs = 0`).
- σ_error là độ lệch chuẩn sai số ngày, không phải RMSE và không phải σ của
  phân vị mô hình.
- Với kịch bản stress, σ_error vẫn lấy từ sai số lịch sử trước origin (chưa
  nhân hệ số stress); đây là lựa chọn có chủ ý để không dùng thông tin tương lai.

### 1.4. `quantile_forecast` (chỉ mô hình có q_0.1 và q_0.9)

Mục tiêu: ROP là **phân vị của tổng nhu cầu L ngày**, không phải tổng các phân vị
từng ngày.

Xấp xỉ đang dùng (ghi rõ giả định):

1. Với mỗi ngày `t`: `μ_t = q_0.5(t)`;
   `σ_t = max(q_0.9(t) − q_0.1(t), 0) / (2 × 1.2816)`
   (1.2816 = z của phân vị chuẩn 0.9).
2. Giả định nhu cầu các ngày **độc lập**:
   - `μ_L = Σ μ_t`
   - `σ_L = √(Σ σ_t²)`
3. `SS = z × σ_L`; `ROP = μ_L + SS`.
4. `Q = kỳ vọng nhu cầu trong q_days ngày` (dùng μ_t, không dùng q_0.9).

**Không dùng `Σ q_0.9(t)`** vì tổng phân vị từng ngày là cận trên bảo thủ
(trường hợp comonotonic), không phải phân vị của tổng khi các ngày không phụ
thuộc hoàn toàn. Xấp xỉ chuẩn ở trên cũng có hạn chế: bỏ qua tương quan giữa các
ngày và có thể kém chính xác với nhu cầu gián đoạn (nhiều ngày bán = 0); cần nêu
trong phần hạn chế của báo cáo.

### 1.4.1. `quantile_forecast_history_sigma` — biến thể (c), σ lịch sử toàn phần

Giữ nguyên μ_L và Q lấy từ q_0.5 như `quantile_forecast`, nhưng thay σ_L bằng
σ lịch sử toàn phần:

```text
μ_L = Σ q_0.5(t) trong L ngày đầu
σ = std(toàn bộ lịch sử đến origin, ddof=1)
SS = z × σ × √L
ROP = μ_L + SS
```

Khi dự báo điểm bằng q_0.5 (đúng quy ước của repo), biến thể này có cùng ROP
với `point_forecast`; khác biệt duy nhất còn lại là cách diễn giải σ. Đây là
phép kiểm tra đối chứng cho mục đích tách phần do ROP và phần do nguồn σ.

### 1.4.2. Tiền xử lý phân vị (bắt buộc trước khi mô phỏng)

1. **Monotone rearrangement**: với mỗi dòng dự báo, sắp xếp các giá trị
   `q_0.1, q_0.5, q_0.9, q_0.95` tăng dần rồi gán lại theo đúng mức. Bản gốc
   được giữ ở các cột `*_raw` trong bộ nhớ; file parquet gốc **không bị sửa**.
   Cột `forecast` được đồng bộ bằng q_0.5 sau sắp xếp vì điểm dự báo của mô
   hình được định nghĩa là trung vị.
   Số dòng/số ô bị sửa được ghi vào
   `results/inventory/quantile_rearrangement.csv`.
2. **Kiểm tra khả dụng phân vị**: một mô hình chỉ được chạy chính sách quantile
   nếu thực sự xuất ra q_0.1, q_0.9 và q_0.95 **không phải bản sao** của q_0.9.
   `Chronos_Bolt` trong repo chép q_0.9 sang q_0.95
   (`src/forecasting/chronos_model.py`), nên chính sách quantile của Bolt bị
   đánh dấu **không khả dụng** và không được đưa vào mô phỏng. Kết quả ghi tại
   `results/inventory/quantile_availability.csv`.

### 1.5. `Oracle` — tham chiếu cận dưới chi phí (KHÔNG phải mô hình)

`Oracle` dùng **nhu cầu thực của tương lai** trong fold (perfect foresight) để
đặt hàng. Mục đích duy nhất: cho biết chi phí tồn kho thấp nhất có thể đạt được
trong cùng cấu trúc `(s, Q)` và cùng tồn kho ban đầu, làm mốc so sánh.

- `ROP_t = Σ nhu cầu thực của L ngày kế tiếp` (`t+1 … t+L`).
- `Q_t = Σ nhu cầu thực của q_days ngày bắt đầu từ lúc hàng về` (`t+L … t+L+q_days−1`).
- Không dùng safety stock, không dùng dự báo.

**Không được** xếp Oracle vào nhóm mô hình, không dùng để kết luận "mô hình nào
tốt hơn". Trong bảng kết quả, Oracle được ghi rõ
`Oracle (lower bound, not a model)` và policy `oracle_reference`.

### 1.6. Tồn kho ban đầu (dùng chung mọi chính sách)

```
tồn kho ban đầu = mean(nhu cầu THỰC trong initial_inventory_lookback_days ngày
                       gần nhất trước origin) × initial_inventory_days
```

Chỉ dùng dữ liệu thực đã biết tại origin, **không dùng dự báo**, để bias của mô
hình không xâm nhập thêm vào tồn kho đầu kỳ.

---

## 2. Mô phỏng

- Chính sách `(s, Q)` rà soát liên tục, **lost sales** (không giao bù).
- Mỗi ngày: nhận hàng → bán → ghi thiếu → kiểm tra đặt hàng.
- Đặt khi `on_hand + on_order ≤ ROP` và không có đơn đang trên đường.
- Đơn đặt ngày `t` về ngày `t + L` (L cố định, không trễ ngẫu nhiên).

---

## 3. Chỉ số

### 3.1. Fill rate — chỉ số CHÍNH là fill rate gộp

```
fill_rate_pooled = 1 − Σ total_short / Σ total_demand
```

Đây là tỷ lệ đơn vị nhu cầu được đáp ứng trên toàn bộ 60 SKU × 13 fold.

Chỉ số phụ (giữ để đối chiếu, **không** dùng làm chỉ số chính):

```
fill_rate_mean_per_sku_fold = mean(fill_rate của từng SKU×fold)
```

Hai chỉ số này có thể chênh nhau rất lớn và thậm chí đảo thứ hạng khi SKU bán ít
có fill rate thấp. Báo cáo phải nêu rõ đang dùng chỉ số nào.

### 3.1.1. Bảng theo `zero_tier` — BẮT BUỘC đi kèm fill gộp

Không được chỉ báo cáo fill gộp. Mỗi lần báo cáo kết quả mô phỏng phải kèm bảng
theo nhóm nhu cầu `smooth / medium / intermittent` (phân loại theo tỷ lệ ngày
`sales = 0` ở `configs/backtest.yaml`) với **fill rate gộp, tổng thiếu hàng và
tổng chi phí** cho từng chính sách. Lý do: nhu cầu gián đoạn chỉ chiếm ~4% tổng
nhu cầu nhưng chiếm tỷ trọng thiếu hàng lớn hơn nhiều, nên fill gộp có thể che
lấp khác biệt giữa các chính sách.

Nguồn số liệu đầy đủ: `results/inventory/simulation_by_zero_tier_<scenario>.csv`.
Bảng tóm tắt bắt buộc cho kịch bản `normal`:

<!-- ZERO_TIER_TABLE_START -->
Bảng dưới đây là kịch bản `normal`, đủ 22 chính sách (gồm cả 3 biến thể σ).
`Hạng*` xếp theo `cost_total` trong từng tier, **không tính Oracle**; Oracle chỉ
là cận dưới chi phí, không phải mô hình. Dấu `—` là Oracle.

| Tier | Hạng* | Model | Policy | Fill gộp | Tổng thiếu | Tổng chi phí |
|---|---:|---|---|---:|---:|---:|
| smooth | — | Oracle (lower bound, not a model) | Oracle* | 0.9541 | 1,794.50 | 5,420.31 |
| smooth | 1 | Baseline_full_history | Baseline_full | 0.9436 | 2,206.91 | 7,225.92 |
| smooth | 2 | Baseline | Baseline | 0.9397 | 2,356.77 | 7,371.48 |
| smooth | 3 | Chronos2_Covariates | point (σ full hist) | 0.9258 | 2,899.84 | 8,650.48 |
| smooth | 3 | Chronos2_Covariates | quantile (σ hist) | 0.9258 | 2,899.84 | 8,650.48 |
| smooth | 5 | LightGBM | point (σ full hist) | 0.9210 | 3,089.75 | 8,900.38 |
| smooth | 6 | Chronos2_Covariates | point (σ 91) | 0.9204 | 3,110.33 | 8,996.60 |
| smooth | 7 | Chronos2_Covariates | point (σ error) | 0.9169 | 3,249.41 | 9,227.33 |
| smooth | 8 | Chronos2_Covariates | quantile | 0.9130 | 3,403.38 | 9,489.25 |
| smooth | 9 | LightGBM | point (σ 91) | 0.9118 | 3,447.93 | 9,535.40 |
| smooth | 10 | LightGBM | point (σ error) | 0.9103 | 3,505.59 | 9,640.62 |
| smooth | 11 | Chronos2_History | point (σ full hist) | 0.9127 | 3,413.08 | 9,657.13 |
| smooth | 11 | Chronos2_History | quantile (σ hist) | 0.9127 | 3,413.08 | 9,657.13 |
| smooth | 13 | Chronos2_History | point (σ 91) | 0.9086 | 3,573.22 | 9,901.22 |
| smooth | 14 | Chronos2_History | point (σ error) | 0.9072 | 3,629.38 | 9,984.67 |
| smooth | 15 | Chronos2_History | quantile | 0.9042 | 3,745.77 | 10,172.95 |
| smooth | 16 | MovingAverage | point (σ full hist) | 0.8981 | 3,983.00 | 10,703.46 |
| smooth | 17 | Chronos_Bolt | point (σ full hist) | 0.8984 | 3,970.54 | 10,757.15 |
| smooth | 18 | MovingAverage | point (σ error) | 0.8968 | 4,033.50 | 10,767.25 |
| smooth | 19 | MovingAverage | point (σ 91) | 0.8960 | 4,066.50 | 10,810.51 |
| smooth | 20 | Chronos_Bolt | point (σ 91) | 0.8911 | 4,258.09 | 11,252.78 |
| smooth | 21 | Chronos_Bolt | point (σ error) | 0.8906 | 4,278.74 | 11,293.95 |
| medium | — | Oracle (lower bound, not a model) | Oracle* | 0.9373 | 765.00 | 4,993.10 |
| medium | 1 | LightGBM | point (σ 91) | 0.9183 | 996.42 | 7,372.57 |
| medium | 2 | LightGBM | point (σ error) | 0.9182 | 998.37 | 7,410.26 |
| medium | 3 | Baseline_full_history | Baseline_full | 0.9144 | 1,043.97 | 7,488.66 |
| medium | 4 | Baseline | Baseline | 0.9129 | 1,062.42 | 7,498.07 |
| medium | 5 | LightGBM | point (σ full hist) | 0.9227 | 942.70 | 7,689.91 |
| medium | 6 | MovingAverage | point (σ error) | 0.9086 | 1,114.75 | 7,759.08 |
| medium | 7 | MovingAverage | point (σ 91) | 0.9053 | 1,155.25 | 7,819.44 |
| medium | 8 | MovingAverage | point (σ full hist) | 0.9103 | 1,095.00 | 8,060.92 |
| medium | 9 | Chronos_Bolt | point (σ 91) | 0.7785 | 2,702.49 | 12,191.72 |
| medium | 10 | Chronos_Bolt | point (σ error) | 0.7772 | 2,718.89 | 12,248.50 |
| medium | 11 | Chronos2_History | quantile | 0.7594 | 2,934.96 | 12,254.86 |
| medium | 12 | Chronos2_Covariates | quantile | 0.7589 | 2,942.20 | 12,304.31 |
| medium | 13 | Chronos2_History | point (σ 91) | 0.7647 | 2,870.55 | 12,332.11 |
| medium | 14 | Chronos2_History | point (σ error) | 0.7654 | 2,862.10 | 12,348.65 |
| medium | 15 | Chronos2_Covariates | point (σ 91) | 0.7621 | 2,903.17 | 12,406.60 |
| medium | 16 | Chronos2_Covariates | point (σ error) | 0.7626 | 2,896.48 | 12,437.07 |
| medium | 17 | Chronos_Bolt | point (σ full hist) | 0.7800 | 2,683.64 | 12,445.20 |
| medium | 18 | Chronos2_History | quantile (σ hist) | 0.7686 | 2,822.90 | 12,547.55 |
| medium | 18 | Chronos2_History | point (σ full hist) | 0.7686 | 2,822.90 | 12,547.55 |
| medium | 20 | Chronos2_Covariates | point (σ full hist) | 0.7654 | 2,862.58 | 12,641.22 |
| medium | 20 | Chronos2_Covariates | quantile (σ hist) | 0.7654 | 2,862.58 | 12,641.22 |
| intermittent | — | Oracle (lower bound, not a model) | Oracle* | 0.9165 | 178.50 | 2,432.06 |
| intermittent | 1 | LightGBM | point (σ 91) | 0.8690 | 280.14 | 4,148.89 |
| intermittent | 2 | LightGBM | point (σ error) | 0.8693 | 279.49 | 4,218.27 |
| intermittent | 3 | Baseline | Baseline | 0.8666 | 285.38 | 4,261.73 |
| intermittent | 4 | LightGBM | point (σ full hist) | 0.8711 | 275.71 | 4,294.98 |
| intermittent | 5 | MovingAverage | point (σ 91) | 0.8463 | 328.75 | 4,348.40 |
| intermittent | 6 | MovingAverage | point (σ error) | 0.8474 | 326.50 | 4,414.03 |
| intermittent | 7 | Baseline_full_history | Baseline_full | 0.8578 | 304.26 | 4,468.48 |
| intermittent | 8 | MovingAverage | point (σ full hist) | 0.8518 | 317.00 | 4,569.89 |
| intermittent | 9 | Chronos2_History | quantile | 0.5156 | 1,036.20 | 6,131.26 |
| intermittent | 10 | Chronos2_Covariates | quantile | 0.4925 | 1,085.46 | 6,218.77 |
| intermittent | 11 | Chronos_Bolt | point (σ 91) | 0.5481 | 966.66 | 6,450.20 |
| intermittent | 12 | Chronos2_History | point (σ 91) | 0.5336 | 997.70 | 6,481.10 |
| intermittent | 13 | Chronos_Bolt | point (σ error) | 0.5503 | 961.85 | 6,485.60 |
| intermittent | 14 | Chronos2_History | point (σ error) | 0.5338 | 997.20 | 6,530.25 |
| intermittent | 15 | Chronos2_Covariates | point (σ 91) | 0.5161 | 1,035.16 | 6,565.07 |
| intermittent | 16 | Chronos_Bolt | point (σ full hist) | 0.5494 | 963.77 | 6,624.74 |
| intermittent | 17 | Chronos2_Covariates | point (σ error) | 0.5141 | 1,039.39 | 6,628.65 |
| intermittent | 18 | Chronos2_History | quantile (σ hist) | 0.5340 | 996.68 | 6,674.41 |
| intermittent | 18 | Chronos2_History | point (σ full hist) | 0.5340 | 996.68 | 6,674.41 |
| intermittent | 20 | Chronos2_Covariates | point (σ full hist) | 0.5160 | 1,035.30 | 6,775.83 |
| intermittent | 20 | Chronos2_Covariates | quantile (σ hist) | 0.5160 | 1,035.30 | 6,775.83 |

Ghi chú đọc bảng:

- Fill gộp theo tier của `intermittent` thấp hơn hẳn `smooth`/`medium`; đây là
  nơi khác biệt giữa các chính sách lộ rõ nhất, dù tier này chỉ chiếm ~4% tổng
  nhu cầu.
- Các cặp `point (σ full hist)` và `quantile (σ hist)` trùng số vì cùng μ_L,
  cùng Q và cùng σ; đây là kết quả kiểm tra đối chứng, không phải hai mô hình
  độc lập.
- Bảng đầy đủ cho cả 3 kịch bản nằm trong
  `results/inventory/simulation_by_zero_tier_<scenario>.csv`.
<!-- ZERO_TIER_TABLE_END -->

### 3.2. Stockout rate

```
stockout_rate = số ngày có units_short > 0 / tổng số ngày
```

Không dùng "ngày tồn kho cuối ngày = 0", vì một ngày có thể hết sạch hàng nhưng
vẫn bán đủ nhu cầu.

### 3.3. Chi phí

- `holding = h × Σ max(inventory_end, 0)` (đơn vị-ngày).
- `ordering = K × số lần đặt`.
- `shortage = p × Σ units_short` (lost sales).
- `total = holding + ordering + shortage`.

Lưu ý: mô hình **chưa** trừ doanh thu/margin bị mất khi thiếu hàng; `p` hiện chỉ
là chi phí phạt. Cần nhóm chốt lại ý nghĩa của `p` trước khi dùng `cost_total`
để kết luận.

---

## 4. Tham số cấu hình liên quan

| Tham số | Giá trị hiện tại | Ghi chú |
|---|---:|---|
| `lead_time_days` | 7 | L cố định |
| `service_level` | 0,95 | z tính tự động |
| `baseline_window_days` | 91 | Cửa sổ baseline chính |
| `reorder_quantity_days` | 14 | Chu kỳ đặt hàng cho Q |
| `initial_inventory_days` | 7 | Số ngày tồn kho ban đầu |
| `initial_inventory_lookback_days` | 28 | Cửa sổ ước lượng nhu cầu cho tồn kho đầu |
| `ordering_cost` | 5,0 | Giả định tạm |
| `holding_cost_per_unit_per_day` | 0,002032 | **Tạm (mức B, 1,5%/tháng)** — xem cách tính dưới |
| `shortage_cost_per_unit` | 2,0 | Giả định tạm, chưa gồm margin mất |

### Cách tính `holding_cost_per_unit_per_day` (mức B)

```text
giá bán trung bình 60 SKU = 4,064 (data/processed/sales_daily.parquet)
h = (1,5% / tháng) × 4,064 / 30 ngày
  = 0,002032 /đơn vị/ngày
  ≈ 18% /năm tính trên giá bán (chưa dùng giá vốn/COGS)
```

Ba mức đã báo cáo để nhóm chọn: A = 0,001355 (1,0%/tháng),
B = 0,002032 (1,5%/tháng), C = 0,002709 (2,0%/tháng). Mức B là **giả định tạm,
chưa được nhóm duyệt**; ma trận độ nhạy trong
`results/inventory/sensitivity_matrix.csv` chạy cả A/B/C.

Toàn bộ giá trị chi phí và vận hành là **giả định tạm, chưa được nhóm duyệt**.

---

## 5. Ma trận độ nhạy (một tham số)

Script `src/inventory/sensitivity.py` chạy kịch bản `normal`, mỗi lần chỉ đổi MỘT
tham số quanh mức cơ sở (h = B, p = 2, K = 5, service level = 0,95). Ma trận
phân biệt hai loại khối:

**A. Độ nhạy kế toán chi phí — KHÔNG đổi hành vi**

| Tham số | Các mức | Cách tính |
|---|---|---|
| `holding_cost_per_unit_per_day` (h) | 0,001355 / 0,002032 / 0,002709 | Tính lại từ đơn vị-ngày tồn kho đã lưu |
| `shortage_cost_per_unit` (p) | 1,0 / 2,0 / 4,0 | Tính lại từ `total_short` đã lưu |
| `ordering_cost` (K) | 2,5 / 5,0 / 10,0 | Tính lại từ `num_orders` đã lưu |

Ba tham số này chỉ đổi trọng số quy đổi chi phí. **ROP, Q, fill rate, tổng thiếu
hàng và số đơn không đổi**; vì vậy không chạy lại mô phỏng mà tính lại trực tiếp
từ `simulation_detail_normal.csv`. Thứ hạng `cost_total` vẫn có thể đổi vì trọng
số chi phí thay đổi — đó là đổi hạng do kế toán, không phải do hành vi đặt hàng.

**B. Độ nhạy hành vi — CÓ đổi ROP/SS/Q**

| Tham số | Các mức | Cách tính |
|---|---|---|
| `service_level` | 0,90 / 0,95 / 0,99 | Chạy lại mô phỏng cho đúng 3 mức |

`service_level` đổi z, do đó đổi SS, ROP và hành vi đặt hàng; phải chạy lại mô
phỏng.

Đầu ra (mỗi cấu hình 22 chính sách, kèm hạng theo `cost_total`):

- `results/inventory/sensitivity_matrix.csv` — toàn bộ 264 dòng.
- `results/inventory/sensitivity_rankings.csv` — bảng xếp hạng gọn.
- `results/inventory/sensitivity_inversions.csv` — các cặp đảo hạng so với cơ sở.
- `results/inventory/sensitivity_matrix.md` — bảng markdown có ghi rõ loại khối.

`Oracle` luôn đứng hạng 1 nhưng chỉ là **cận dưới chi phí**, không phải mô hình.
Các cặp đảo hạng chỉ nằm giữa các chính sách thật (không liên quan Oracle).

