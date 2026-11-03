# Danh mục Mô hình Dự báo & Cấu hình Thực nghiệm (Phase 1–4)

Tài liệu này mô tả chi tiết tất cả các mô hình dự báo trong hệ thống, bao gồm baseline, LightGBM (Phase 3) và 3 biến thể Chronos Zero-shot (Phase 4).

---

## 1. Tổng quan các Mô hình & Giấy phép

| Mô hình | Định danh trong code | Checkpoint / Gói | Giấy phép | Mô tả tóm tắt |
|---|---|---|---|---|
| **Naive** | `Naive` | Tự cài đặt (`src/forecasting/baselines.py`) | Mã nguồn dự án | Dự báo giá trị ngày cuối cùng trước origin cho cả 28 ngày |
| **Seasonal Naive** | `SeasonalNaive` | Tự cài đặt (`src/forecasting/baselines.py`) | Mã nguồn dự án | Lặp lại chu kỳ 7 ngày gần nhất trước origin (lag-7) |
| **Moving Average** | `MovingAverage` | Tự cài đặt (`src/forecasting/baselines.py`) | Mã nguồn dự án | Trung bình trượt 28 ngày gần nhất trước origin |
| **LightGBM** | `LightGBM` | `lightgbm==4.6.0` | MIT License | GBDT đệ quy đa biến với 25 features (lag, rolling, calendar, price, SNAP, event) |
| **Chronos-2 History** | `Chronos2_History` | `amazon/chronos-2` (~120M params, ~478 MB) | Apache-2.0 | Time-Series Foundation Model đơn biến, chỉ nhận lịch sử bán hàng (`sales`) |
| **Chronos-2 Covariates** | `Chronos2_Covariates` | `amazon/chronos-2` (~120M params, ~478 MB) | Apache-2.0 | Time-Series Foundation Model đa biến nhận lịch sử bán hàng + biến ngoại sinh biết trước 28 ngày |
| **Chronos-Bolt** | `Chronos_Bolt` | `amazon/chronos-bolt-mini` (~21.2M params, ~85 MB) | Apache-2.0 | Foundation model dạng patch/encoder-only tốc độ cao, chỉ nhận lịch sử bán hàng |

---

## 2. Quy ước Điểm Dự báo & Phân vị (Probabilistic Forecasts)

- **Điểm dự báo (Point forecast)**:
  - Tất cả 3 biến thể Chronos sử dụng **trung vị (quantile 0.5)** làm điểm dự báo chính (`forecast = q_0.5`).
  - Điểm dự báo được cắt không âm: `forecast = max(0.0, forecast)`.
  - Không làm tròn về số nguyên để giữ nguyên độ chính xác cho việc tính toán tồn kho an toàn và chi phí.
- **Phân vị dự báo (Quantiles)**:
  - Được lưu đầy đủ vào file Parquet: `q_0.1`, `q_0.5`, `q_0.9`, `q_0.95`.
  - Riêng `Chronos-Bolt` được huấn luyện xuất tối đa quantile 0.90; theo quy định an toàn, cột `q_0.95` được gán bằng `q_0.90` (clamped).
  - Phân vị được đánh giá riêng qua Pinball Loss và lưu tại `results/forecasting/chronos_pinball_loss.csv`. Tuyệt đối không đưa Pinball Loss vào bảng so sánh điểm (MAE/RMSE/WAPE/MASE) để đảm bảo tính công bằng.

---

## 3. Giả định Biến Ngoại sinh Biết trước (Chronos2_Covariates & LightGBM)

Biến thể `Chronos2_Covariates` sử dụng cùng tập biến ngoại sinh và cùng giả định **"biết trước 28 ngày" (oracle/known-future)** giống như `LightGBM` (so sánh công bằng):
1. **Giá bán (`sell_price`)**: Giả định biết trước giá niêm yết trong chu kỳ bổ sung 28 ngày (nếu thiếu, fallback về giá ngày gần nhất tại origin).
2. **Lịch sự kiện (`has_event`)**: Cờ nhị phân (1/0) biểu thị ngày có sự kiện văn hóa/tôn giáo/quốc gia đã lên lịch trước.
3. **Trợ cấp SNAP California (`snap_CA`)**: Cờ nhị phân (1/0) theo lịch cấp phát tem phiếu thực phẩm cố định của bang California.
4. **Lịch thứ trong tuần (`wday`)**: Số nguyên 1..7 (bắt đầu từ thứ Bảy theo quy ước M5).
5. **Tháng trong năm (`month`)**: Số nguyên 1..12.

Nguyên tắc chống rò rỉ: Lịch sử chỉ lấy tối đa 1024 ngày gần nhất tính đến thời điểm origin; tuyệt đối không dùng bất kỳ số liệu doanh số (`sales`) nào sau origin.

---

## 4. Khung Đánh giá & Thiết lập Thực nghiệm (Reproducibility)

- **Dữ liệu**: 60 SKUs thuộc cửa hàng `CA_1` (Walmart M5), dải thời gian $d_1 \dots d_{1941}$.
- **Giai đoạn hold-out backtest**: 13 fold rolling-origin, step 28 ngày:
  - Fold 0: origin $d_{1577}$, dự báo $d_{1578} \dots d_{1605}$
  - ...
  - Fold 12: origin $d_{1913}$, dự báo $d_{1914} \dots d_{1941}$
  - Tổng số mẫu đánh giá: 13 folds $\times$ 60 SKUs $\times$ 28 days = **21,840 quan sát**.
- **Context Length**: Cố định tối đa 1024 ngày gần nhất trước origin cho Chronos, không tune.
- **Zero-shot**: Cố định hoàn toàn trọng số pretrained từ Hugging Face, **không fine-tune**, **không điều chỉnh siêu tham số trên 13 fold hold-out**.
- **Cố định Seed**: Seed = 42 (`torch.manual_seed(42)`, `np.random.seed(42)`).

---

## 5. Thời gian Chạy Thực tế (Inference Time trên CPU)

Phần cứng kiểm thử: Intel Core i7-1255U (10 nhân, 12 luồng), 16 GB RAM, OS Windows.

| Biến thể | Checkpoint | Thời gian 1 fold (tb) | Tổng thời gian 13 folds | Tốc độ tương đối |
|---|---|---|---|---|
| `Chronos_Bolt` | `amazon/chronos-bolt-mini` | ~1.0 s | **13.1 s (0.22 phút)** | Nhanh nhất (~52x so với Chronos2_Covariates) |
| `Chronos2_History` | `amazon/chronos-2` | ~7.2 s | **93.8 s (1.56 phút)** | Nhanh (~7x so với Chronos2_Covariates) |
| `Chronos2_Covariates` | `amazon/chronos-2` | ~53.0 s | **689.3 s (11.49 phút)** | Tốn tài nguyên nhất do xử lý cross-attention đa biến |

---

## 6. Môi trường Thực thi & Lệnh Chạy lại

Dự án tuân thủ nghiêm ngặt việc cách ly môi trường:
- `.venv`: Môi trường chính (chạy baselines, LightGBM, so sánh 7 mô hình, kiểm thử chung). Không cài PyTorch hay Chronos.
- `.venv-tsfm`: Môi trường TSFM riêng biệt (chứa `torch 2.15.0+cpu`, `chronos-forecasting 2.3.2`, `transformers 5.22.1`).

### Lệnh chạy lại Backtest Chronos (.venv-tsfm):
```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'

# Chạy thử 1 fold (ví dụ Fold 0)
.\.venv-tsfm\Scripts\python src/forecasting/run_chronos.py --test-fold 0

# Chạy toàn bộ 13 folds cho cả 3 biến thể
.\.venv-tsfm\Scripts\python src/forecasting/run_chronos.py --variant all

# Chạy bộ test tích hợp kiểm tra rò rỉ và định dạng
.\.venv-tsfm\Scripts\pytest -m slow -v tests/test_chronos_slow.py
```

### Lệnh cập nhật và kiểm tra so sánh 7 mô hình (.venv chính):
```powershell
$env:PYTHONPATH = '.'
$env:PYTHONIOENCODING = 'utf-8'

# Cập nhật lại toàn bộ 4 bảng so sánh và tính Pinball Loss
.\.venv\Scripts\python -c "from src.forecasting.run_chronos_backtest import update_model_comparisons; update_model_comparisons()"

# Chạy toàn bộ test suite cơ bản (không gồm test slow)
.\.venv\Scripts\pytest -q
```
