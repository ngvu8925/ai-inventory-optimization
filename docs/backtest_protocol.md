# Quy trình Backtest và Đánh giá Mô hình Dự báo Nhu cầu (Phase 3)

Tài liệu này mô tả chi tiết phương pháp luận kiểm thử ngược (backtest) và các thước đo sai số được sử dụng để đánh giá công bằng các mô hình dự báo trong đề tài nghiên cứu.

---

## 1. Nguyên tắc thiết kế Backtest

Để mô phỏng chính xác điều kiện triển khai thực tế và ngăn ngừa rò rỉ dữ liệu tương lai (data leakage), nghiên cứu áp dụng kỹ thuật **Rolling-Origin với Expanding Window** (gốc thời gian trượt, cửa sổ huan luyện mở rộng) trên giai đoạn hold-out 364 ngày:

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
| **Bias** | $\frac{1}{N} \sum (\hat{y}_t - y_t)$ | Xu hướng lệch: $>0$ là dự báo thừa (tăng chi phí lưu kho), $<0$ là dự báo thiếu (tăng nguy cơ hết hàng). |
| **MASE** | $\frac{\text{MAE}}{\frac{1}{T-1}\sum_{t=2}^T \|y_t - y_{t-1}\|}$ | Sai số tương đối so với mô hình Naive 1 bước trên tập train. $\text{MASE} < 1$ thể hiện mô hình tốt hơn Naive cơ bản. |

---

## 4. Các chiều phân tích kết quả

Kết quả dự báo trên 13 fold ($21.840$ điểm dữ liệu cho mỗi mô hình) được tổng hợp và phân tích theo 4 góc độ:

1. **Tổng thể (Overall)**: Đánh giá hiệu năng bình quân trên toàn bộ 60 SKU và 13 fold.
2. **Theo từng Fold (By Fold)**: Theo dõi độ ổn định của mô hình qua các mùa trong năm (từ tháng 5/2015 đến tháng 5/2016).
3. **Theo Ngành hàng (By Category)**: Phân tách riêng cho `FOODS`, `HOBBIES`, `HOUSEHOLD` để nhận diện thế mạnh của từng thuật toán trên các nhóm mặt hàng khác nhau.
4. **Theo Mức độ gián đoạn (By Zero-demand Tier)**: Chia 60 SKU thành 3 nhóm dựa trên tỷ lệ ngày bán bằng 0 trong tập train ($d_1 \dots d_{1577}$):
   - `smooth` ($< 33\%$ ngày bằng 0): Hàng bán nhanh, nhu cầu liên tục.
   - `medium` ($33\% - 66\%$ ngày bằng 0): Nhu cầu trung bình.
   - `intermittent` ($> 66\%$ ngày bằng 0): Nhu cầu gián đoạn mạnh, nhiều ngày liên tiếp không có lượt mua.
