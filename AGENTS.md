# AGENTS.md — Hệ thống dự báo nhu cầu & tối ưu tồn kho (ML + TSFM + LLM)

> Đặt file này ở thư mục gốc repo. Antigravity / Cursor / Codex đọc `AGENTS.md`.
> Claude Code đọc `CLAUDE.md` → tạo file đó chỉ gồm một dòng: `@AGENTS.md`

## 1. Bối cảnh
Đề tài NCKH (Đại học Đông Á, nhóm lớp ST23D): xây hệ thống web nguyên mẫu gồm 3 phần tách biệt:
1. **Dự báo nhu cầu** (baseline, ML, Time-Series Foundation Model).
2. **Tối ưu tồn kho** (ROP, safety stock, lượng bổ sung, mô phỏng chính sách).
3. **Trợ lý LLM** giải thích kết quả và trả lời câu hỏi bằng ngôn ngữ tự nhiên.

Đây là dự án **nghiên cứu**: kết quả phải kiểm chứng được và công bằng giữa các mô hình. Không phải sản phẩm thương mại.

## 2. LUẬT BẮT BUỘC (không được vi phạm)
1. **Chia dữ liệu theo thời gian** (train < val < test). Cấm random split / shuffle. Dùng backtesting (rolling origin) nhiều giai đoạn.
2. **Cấm rò rỉ dữ liệu tương lai**: feature chỉ dùng thông tin có tại thời điểm dự báo (lag, rolling mean phải `shift` trước khi tính).
3. **Mọi mô hình so sánh trên cùng dữ liệu, cùng horizon, cùng quy trình đánh giá.** Luôn có baseline (naive kỳ trước, seasonal naive).
4. **Foundation model (Chronos/TimesFM): zero-shot trước.** Chỉ fine-tune khi tôi yêu cầu rõ.
5. **Chỉ số**: MAE, RMSE, WAPE hoặc MASE. **Không dùng MAPE** nếu dữ liệu có nhu cầu = 0 (chia cho 0).
6. **LLM không tự tính hay bịa số.** Mọi số liệu (dự báo, ROP, SS, lượng đặt) do backend tính bằng code có test; LLM chỉ nhận số đó qua tool calling và diễn giải. Nếu không có dữ liệu → LLM phải nói "không có dữ liệu", không đoán.
7. **Không tự khẳng định** "mô hình X tốt hơn" hay "tiết kiệm Y%". Chỉ báo cáo số đo thực tế kèm giả định. Không đặt trước tỷ lệ cải thiện.
8. **Mọi giả định** (chi phí đặt hàng/lưu kho/thiếu hàng, lead time, quy tắc đặt hàng) nằm trong file config (`configs/`), không hard-code trong code.
9. **Reproducible**: cố định seed, ghi phiên bản thư viện, lưu kết quả thực nghiệm vào `results/` kèm config đã dùng.

## 3. Công thức tồn kho (theo thuyết minh)
- `ROP = d̄ × L + SS` (d̄: nhu cầu TB/ngày, L: lead time ngày, SS: tồn kho an toàn)
- Tồn kho khả dụng = hàng thực có + hàng đang về − hàng đã cam kết
- SS mặc định: `z × σ_d × sqrt(L)` (z theo service level, cấu hình được). Đây là giả định mặc định, ghi rõ trong báo cáo.
- Mô phỏng so sánh **ít nhất 2 chính sách**: (a) cơ sở (ví dụ ROP từ trung bình lịch sử), (b) dựa trên dự báo. Chỉ số: tổng chi phí mô phỏng, Fill Rate, tỷ lệ thiếu hàng.
- Có kịch bản stress: nhu cầu tăng đột biến, lead time kéo dài.

## 4. Stack
- Python 3.11+, Pandas, NumPy, Scikit-learn, XGBoost hoặc LightGBM
- TSFM: Chronos hoặc TimesFM (chọn theo tương thích + tài nguyên máy)
- Backend: FastAPI (+ Pydantic, pytest). Frontend: ReactJS. Biểu đồ: Plotly
- DB: chọn **một** hệ chính (PostgreSQL hoặc MongoDB) — hỏi tôi trước khi chốt
- LLM: một API hoặc model open-source; RAG chỉ thêm khi cần tra tài liệu nghiệp vụ

## 5. Cấu trúc repo
```text
.
├── AGENTS.md
├── README.md
├── configs/            # giả định chi phí, lead time, horizon, seed
├── data/
│   ├── raw/            # không sửa, không commit file lớn
│   ├── processed/
│   └── DATA_CARD.md    # nguồn, cấu trúc, phạm vi thời gian, số SKU
├── notebooks/          # EDA, thử nghiệm
├── src/
│   ├── data/           # load, clean, aggregate (ngày/tuần)
│   ├── forecasting/    # baselines.py, ml_model.py, tsfm_model.py, backtest.py, metrics.py
│   ├── inventory/      # policies.py, simulator.py, costs.py
│   ├── llm/            # tools.py (tool calling), prompts.py, assistant.py, eval.py
│   └── api/            # FastAPI routers, schemas
├── frontend/           # React dashboard
├── results/            # bảng/biểu đồ thực nghiệm (kèm config)
├── tests/
└── docs/               # ghi chú để viết báo cáo
```

## 6. Lộ trình (16 tuần) — làm tuần tự, xong phase nào báo cáo phase đó
| Phase | Tuần | Việc | Xong khi |
|---|---|---|---|
| 1 | 1–2 | Khảo sát tài liệu, chọn dữ liệu | Có `DATA_CARD.md` + danh sách nguồn |
| 2 | 3–4 | Làm sạch, EDA | Pipeline xử lý chạy lại được + biểu đồ EDA (xu hướng, mùa vụ, nhu cầu gián đoạn) |
| 3 | 5–7 | Baseline + XGBoost/LightGBM | Bảng MAE/RMSE/WAPE-MASE qua backtesting |
| 4 | 8–9 | Chronos/TimesFM zero-shot | So sánh cùng quy trình với phase 3 |
| 5 | 10–11 | Mô-đun tồn kho + mô phỏng | Báo cáo chi phí, Fill Rate, thiếu hàng cho ≥2 chính sách |
| 6 | 12–13 | LLM + API | Bộ test kiểm tra câu trả lời khớp số backend |
| 7 | 14–15 | Dashboard + tích hợp | App chạy end-to-end |
| 8 | 16 | Báo cáo | Tổng hợp kết quả, hạn chế, hướng phát triển |

## 7. Cách làm việc với tôi
- Tôi là sinh viên, học qua làm. **Làm từng bước nhỏ**, giải thích ngắn gọn bằng tiếng Việt, đơn giản.
- Trước khi code phase mới: tóm tắt kế hoạch (3–6 gạch đầu dòng) và **chờ tôi đồng ý**.
- Mỗi bước xong: nói rõ file đã tạo/sửa, lệnh để chạy, kết quả mong đợi để tôi chụp màn hình xác nhận.
- Không cài thêm thư viện lớn, không xóa file, không đổi stack khi chưa hỏi.
- Việc nào phải chọn (dataset, DB, model, LLM) → đưa 2–3 phương án + ưu/nhược + đề xuất, để tôi chọn.
- Viết test cho phần tính toán (metrics, ROP, SS, simulator) trước khi nối vào API/LLM.
- Code: type hints, docstring ngắn, tên biến tiếng Anh, comment tiếng Việt khi logic không hiển nhiên.
- Commit nhỏ, message dạng `feat(forecasting): add seasonal naive baseline`.
- Nếu không chắc hoặc kết quả bất thường (sai số quá thấp, nghi rò rỉ dữ liệu) → **dừng và báo**, không tự "sửa cho đẹp".
- Dữ liệu chỉ dùng nguồn công khai hoặc được cung cấp hợp pháp; ghi nguồn + license vào `DATA_CARD.md`.
