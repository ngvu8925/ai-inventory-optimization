# Data Card — M5 Walmart Forecasting

## Nguồn

- Bộ dữ liệu: [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
- Trang tổng quan: [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview)
- Đơn vị cung cấp trên Kaggle: University of Nicosia; dữ liệu bán hàng của Walmart.
- Kaggle ghi giấy phép là **Subject to Competition Rules**. Đây không phải giấy phép mở độc lập. Người tải cần đăng nhập Kaggle và chấp nhận quy định của cuộc thi. Chỉ sử dụng, lưu trữ, chia sẻ hoặc công bố dữ liệu trong phạm vi được các quy định đó cho phép; không đưa dữ liệu thô vào Git hay phân phối lại.
- Ngày truy cập thông tin nguồn: 2026-10-27.

## Nội dung và cấu trúc

Các file là CSV. Kaggle mô tả năm file trong bộ tải xuống. Các cột dưới đây tóm tắt cấu trúc được tài liệu cuộc thi mô tả; số ngày trong hai file sales khác nhau.

| File | Cấu trúc và ý nghĩa |
|---|---|
| `calendar.csv` | Một dòng cho mỗi ngày, với `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, các cột sự kiện (`event_name_1/2`, `event_type_1/2`) và các cờ SNAP theo bang (`snap_CA`, `snap_TX`, `snap_WI`). Cột `d` ánh xạ ngày sang chỉ số dạng `d_1` … `d_1969`. |
| `sell_prices.csv` | Một dòng theo `store_id`, `item_id` và `wm_yr_wk`, có `sell_price`. Giá được cung cấp theo tuần; tổ hợp không có giá trong tuần có nghĩa sản phẩm không được bán trong tuần đó, theo mô tả thảo luận của cuộc thi. |
| `sales_train_validation.csv` | Một dòng cho mỗi tổ hợp sản phẩm–cửa hàng; cột định danh: `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`; cột nhu cầu ngày: `d_1` … `d_1913`. Đây là phần lịch sử trước kỳ validation của cuộc thi. |
| `sales_train_evaluation.csv` | Cùng cấu trúc định danh và cột sales dạng rộng; chứa `d_1` … `d_1941`, tức bổ sung 28 ngày sau mốc kết thúc của `sales_train_validation.csv`. Kaggle mô tả đây là file có nhãn dùng cho leaderboard công khai. |
| `sample_submission.csv` | Mẫu định dạng dự báo 28 ngày, với `id` và `F1` … `F28`; không phải dữ liệu lịch sử bán hàng. |

## Phạm vi thời gian và quy mô

- Lịch ngày đầy đủ trong `calendar.csv`: `d_1` bắt đầu **2011-01-29**; `d_1969` là **2016-06-19** (1.969 ngày liên tục).
- `sales_train_validation.csv`: `d_1` … `d_1913`, tương ứng **2011-01-29 – 2016-04-24**.
- `sales_train_evaluation.csv`: `d_1` … `d_1941`, tương ứng **2011-01-29 – 2016-05-22**.
- Bài toán cuộc thi đánh giá các kỳ 28 ngày tiếp theo; chỉ số lịch chạy đến `d_1969` (**2016-06-19**).
- M5 có **3.049 SKU** (mã `item_id`) và **10 cửa hàng** (`store_id`) tại California, Texas và Wisconsin; cấp SKU × cửa hàng tạo **30.490 chuỗi**. Có 3 ngành hàng (`cat_id`) và 7 phòng ban (`dept_id`).

## Giới hạn và lưu ý nghiên cứu

- Nhu cầu là **số lượng bán ra**, không phải nhu cầu tiềm ẩn hay dữ liệu đơn hàng chưa được đáp ứng. Do không có dữ liệu tồn kho, không thể xác định trực tiếp ngày hết hàng hoặc lượng cầu bị mất.
- Dữ liệu không cung cấp **lead time**, lượng hàng đang về, lượng đã cam kết, chi phí đặt hàng, lưu kho hay thiếu hàng. Các trường hợp này phải được mô phỏng bằng giả định có cấu hình và báo cáo riêng; không xem chúng là quan sát thực tế từ Walmart.
- Có nhiều giá trị sales bằng 0 và chuỗi nhu cầu gián đoạn, nhất là ở cấp SKU–cửa hàng. Giữ các số 0 như quan sát; không diễn giải chúng tự động là stockout hay thiếu dữ liệu.
- Giá chỉ được ghi theo tuần, trong khi sales theo ngày. Thiếu giá trong một tổ hợp cửa hàng–SKU–tuần được mô tả là sản phẩm không được bán tuần đó; không tự nội suy giá nếu chưa có quy tắc được nêu rõ.
- Dữ liệu thuộc Walmart tại ba bang Hoa Kỳ và giai đoạn lịch sử cụ thể; kết quả không tự động khái quát cho nhà bán lẻ hoặc giai đoạn khác.
- Dùng `sales_train_validation.csv` cho mô hình hóa ban đầu để tránh vô tình sử dụng các nhãn mở rộng có trong `sales_train_evaluation.csv`. Mọi mốc train/validation/test trong nghiên cứu phải được xác định theo thời gian; không dùng random split.

## Tập dữ liệu con nghiên cứu (Subset 60 SKU tại CA_1 — Áp dụng từ Phase 2)

### Khảo sát dữ liệu thực tế tại CA_1 (3.049 SKU)
Trong quá trình tiền xử lý, toàn bộ 3.049 SKU tại cửa hàng `CA_1` được phân tích về thời điểm mở bán và lịch sử giá:
- **1.204 SKU có mặt từ ngày đầu tiên ($d_1$, tuần 11101)**: Có đầy đủ 278 tuần giá liên tục từ 2011 đến 2016, không có dòng nào khuyết giá.
- **1.835 SKU mở bán giữa chừng trong giai đoạn train ($11101 < \text{wm\_yr\_wk} \le 11517$)**: Trước tuần mở bán, các SKU này có `sell_price` bị khuyết (NaN) và doanh số bằng 0 do hàng chưa được bày bán tại siêu thị.
- **10 SKU mở bán sau ngày $d_{1577}$**: Chỉ xuất hiện từ giai đoạn hold-out ($d_{1578} \dots d_{1941}$); trong toàn bộ giai đoạn train ($d_1 \dots d_{1577}$), 10 SKU này hoàn toàn không có giá và sales = 0.

### Quy tắc chọn 60 SKU (Phương án 1)
- **Tập ứng viên**: Chỉ lấy các SKU tại `CA_1` có giá bán ngay từ tuần đầu tiên (`wm_yr_wk == 11101`). Điều kiện lọc này chỉ dựa trên thông tin tại mốc $d_1$, hoàn toàn không nhìn vào dữ liệu sau $d_{1577}$.
- **Lý do chọn**: Mục tiêu đề tài là so sánh công bằng các mô hình dự báo chuỗi thời gian (Phase 3–4) và mô phỏng chính sách tồn kho (Phase 5). Việc đưa các SKU mở bán muộn sẽ làm phát sinh hàng trăm ngày "chưa bán" (sales = 0 giả tạo kết hợp NaN giá), lẫn lộn với nhu cầu gián đoạn thật sự và làm sai lệch cả thước đo sai số (WAPE, MASE) lẫn chi phí tồn kho. Bài toán ra mắt sản phẩm mới (cold-start) nằm ngoài phạm vi thuyết minh. Phương án 1 vẫn bảo đảm tính đa dạng cao với tỷ lệ ngày bán bằng 0 trải rộng từ ~2% đến ~96%.
- **Quy tắc phân vị đều**:
  Trong mỗi ngành hàng (`FOODS`, `HOBBIES`, `HOUSEHOLD`), tính tổng doanh số bán trong tập train $d_1 \dots d_{1577}$. Sắp xếp tăng dần theo `(total_sales_train, item_id)` (tiêu chí phân định khi trùng là mã `item_id` tăng dần). Với $N$ ứng viên, chọn 20 chỉ số:
  $$\text{index}[i] = \text{round}\left(i \times \frac{N - 1}{19}\right), \quad i = 0 \dots 19$$
  Quy tắc được lưu tại `configs/data_config.yaml`, danh sách 60 SKU lưu tại `configs/selected_skus.csv`.

### Số liệu thực đo trên subset 60 SKU
- **File lưu trữ**: `data/processed/sales_daily.parquet` và `data/processed/sales_weekly.parquet` (không đưa vào Git).
- **Quy mô dòng**:
  - `sales_daily.parquet`: **116.460 dòng** ($60 \text{ SKU} \times 1.941 \text{ ngày}$).
    - Giai đoạn train ($d_1 \dots d_{1577}$, 2011-01-29 đến 2015-05-24): **94.620 dòng**.
    - Giai đoạn hold-out ($d_{1578} \dots d_{1941}$, 2015-05-25 đến 2016-05-22, 364 ngày): **21.840 dòng**.
  - `sales_weekly.parquet`: **16.680 dòng** ($60 \text{ SKU} \times 278 \text{ tuần}$).
- **Giá trị thiếu (Missing values)**:
  - `sales`: **0 dòng thiếu** (100% đầy đủ).
  - `sell_price`: **0 dòng thiếu** (100% đầy đủ giá liên tục cho cả 60 SKU).
- **Tỷ lệ ngày bán bằng 0 (% Zero-demand) trên tập train $d_1 \dots d_{1577}$**:
  - Toàn bộ 60 SKU: **54.22%** (toàn bộ 1.941 ngày là **54.08%**).
  - Ngành hàng `FOODS` (20 SKU): **43.72%** (SKU thấp nhất: 2.16%, SKU cao nhất: 95.81%).
  - Ngành hàng `HOBBIES` (20 SKU): **58.04%** (SKU thấp nhất: 22.32%, SKU cao nhất: 94.04%).
  - Ngành hàng `HOUSEHOLD` (20 SKU): **60.90%** (SKU thấp nhất: 7.93%, SKU cao nhất: 94.48%).
- **Thống kê doanh số và giá bán (tập train)**:
  - Doanh số bán trung bình: **2.83 đơn vị/ngày** (độ lệch chuẩn 12.01; trung vị = 0; giá trị lớn nhất 599).
  - Giá bán trung bình: **$4.06** (nhỏ nhất $0.42, lớn nhất $15.28).

### Hạn chế nghiên cứu đã ghi nhận
- **Chỉ gồm SKU mở bán từ đầu**: Không bao gồm sản phẩm mới ra mắt giữa chừng (cold-start). Kết quả dự báo và chính sách tồn kho phản ánh hành vi của các mặt hàng đã có lịch sử lưu hành ổn định.
- **Phạm vi cửa hàng**: Chỉ thực hiện trên cửa hàng `CA_1` tại California; chưa khái quát hóa cho các cửa hàng khác của Walmart.

## Tài liệu tham khảo

1. Kaggle, [M5 Forecasting — Accuracy: Data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) — mô tả file và giấy phép theo quy định cuộc thi.
2. Kaggle, [M5 Forecasting — Accuracy: Overview](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview) — phạm vi Walmart, bang, và kỳ dự báo.
3. Kaggle, [thảo luận về mô tả dữ liệu M5](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/134910) — trường calendar, giá tuần và cấu trúc sales.
4. Makridakis et al., [M5 accuracy competition: Results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013) — bối cảnh và quy mô chuỗi của cuộc thi.
5. Makridakis et al., [Product sales probabilistic forecasting: An empirical evaluation using the M5 competition data](https://doi.org/10.1016/j.ijpe.2021.108237) — thảo luận đặc tính nhu cầu gián đoạn và các chuỗi bán hàng ở cấp sản phẩm–cửa hàng.
