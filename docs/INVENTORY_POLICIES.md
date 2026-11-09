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
tham số quanh mức cơ sở (h = B, p = 2, K = 5, service level = 0,95):

| Tham số | Các mức đã chạy |
|---|---|
| `holding_cost_per_unit_per_day` | 0,001355 / 0,002032 / 0,002709 |
| `shortage_cost_per_unit` | 1,0 / 2,0 / 4,0 |
| `ordering_cost` | 2,5 / 5,0 / 10,0 |
| `service_level` | 0,90 / 0,95 / 0,99 |

Đầu ra (mỗi cấu hình 11 chính sách, kèm hạng theo `cost_total`):

- `results/inventory/sensitivity_matrix.csv` — toàn bộ 132 dòng.
- `results/inventory/sensitivity_rankings.csv` — bảng xếp hạng gọn.
- `results/inventory/sensitivity_inversions.csv` — các cặp đảo hạng so với cơ sở.
- `results/inventory/sensitivity_matrix.md` — bảng markdown.

`Oracle` luôn đứng hạng 1 nhưng chỉ là **cận dưới chi phí**, không phải mô hình.
Các cặp đảo hạng chỉ nằm giữa các chính sách thật (không liên quan Oracle).

