# Tải dữ liệu M5

1. Đăng nhập hoặc tạo tài khoản tại [Kaggle](https://www.kaggle.com/).
2. Mở trang [M5 Forecasting — Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
3. Đọc và chấp nhận Competition Rules/điều khoản trên Kaggle để được phép tải dữ liệu.
4. Tải các file CSV từ trang Data.
5. Giải nén nếu Kaggle tải về file nén, rồi chép các CSV vào `data/raw/` ở repo này. Không đổi tên file gốc.

> Các file thô bị loại khỏi Git bởi `.gitignore`. Không tải lên Git, gửi lại hoặc phân phối dữ liệu; hãy làm theo Competition Rules và mục giấy phép trong [DATA_CARD.md](DATA_CARD.md).

Các file cần thiết cho khảo sát M5:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sample_submission.csv` (mẫu định dạng dự báo)

Tài liệu cấu trúc, phạm vi, nguồn và giới hạn: [DATA_CARD.md](DATA_CARD.md).
