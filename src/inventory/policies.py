"""Các chính sách tồn kho (Reorder Point & Safety Stock) — Phase 5.

3 chính sách được so sánh:
  1. 'baseline'        : ROP từ nhu cầu trung bình lịch sử (không dùng mô hình dự báo).
  2. 'point_forecast'  : ROP từ điểm dự báo (median) của mô hình ML/TSFM.
  3. 'quantile_forecast': SS tính từ phân vị xác suất của Chronos (không giả định chuẩn).

Công thức (theo thuyết minh AGENTS.md):
  ROP = d_bar * L + SS
  SS  = z * sigma_d * sqrt(L)   (chính sách baseline & point_forecast)
  SS  = q_{1-alpha}(horizon=L) - d_bar * L  (chính sách quantile, nếu có phân vị)

Tất cả tham số đọc từ config dict, KHÔNG hard-code.
"""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def load_assumptions(config_path: str = "configs/assumptions.yaml") -> dict:
    """Đọc file cấu hình giả định tồn kho."""
    import yaml
    from pathlib import Path
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_safety_stock_normal(
    sigma_d: float,
    lead_time_days: int,
    z_score: float,
) -> float:
    """Tính Safety Stock theo phân phối chuẩn (giả định mặc định).

    SS = z * sigma_d * sqrt(L)

    Args:
        sigma_d: Độ lệch chuẩn nhu cầu hàng ngày (đơn vị: units/ngày).
        lead_time_days: Lead time cố định (L, ngày).
        z_score: Hệ số z tương ứng service level mục tiêu.

    Returns:
        Giá trị Safety Stock (units), đã làm tròn không âm.
    """
    ss = z_score * sigma_d * np.sqrt(lead_time_days)
    return float(max(ss, 0.0))


def calc_rop(
    d_bar: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    """Tính Reorder Point.

    ROP = d_bar * L + SS

    Args:
        d_bar: Nhu cầu trung bình hàng ngày (units/ngày).
        lead_time_days: Lead time cố định (L, ngày).
        safety_stock: Safety stock (units).

    Returns:
        Giá trị ROP (units).
    """
    rop = d_bar * lead_time_days + safety_stock
    return float(max(rop, 0.0))


def policy_baseline(
    history_sales: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Chính sách cơ sở (Baseline): ROP từ trung bình & độ lệch chuẩn lịch sử.

    Không sử dụng bất kỳ mô hình dự báo nào.
    SS = z * std(history_sales) * sqrt(L).

    Args:
        history_sales: Chuỗi doanh số lịch sử hàng ngày (mảng 1-D).
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        dict với keys: d_bar, sigma_d, safety_stock, rop, order_qty.
    """
    L = int(cfg["lead_time_days"])
    z = float(cfg["z_score"])
    q_days = int(cfg["reorder_quantity_days"])

    d_bar = float(np.mean(history_sales))
    sigma_d = float(np.std(history_sales, ddof=1)) if len(history_sales) > 1 else 0.0

    ss = calc_safety_stock_normal(sigma_d, L, z)
    rop = calc_rop(d_bar, L, ss)
    order_qty = max(d_bar * q_days, 1.0)  # đặt ít nhất 1 đơn vị

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": round(sigma_d, 4),
        "safety_stock": round(ss, 4),
        "rop": round(rop, 4),
        "order_qty": round(order_qty, 4),
    }


def policy_point_forecast(
    forecast_values: np.ndarray,
    history_sales: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Chính sách điểm dự báo: ROP từ trung bình dự báo điểm + SS từ lịch sử.

    d_bar = mean(forecast) trong lead_time_days đầu tiên.
    SS    = z * std(history_sales) * sqrt(L)  [vẫn dùng lịch sử để ước lượng sigma].

    Args:
        forecast_values: Mảng dự báo điểm (median) cho horizon = 28 ngày.
        history_sales: Chuỗi doanh số lịch sử để ước lượng sigma_d.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        dict với keys: d_bar, sigma_d, safety_stock, rop, order_qty.
    """
    L = int(cfg["lead_time_days"])
    z = float(cfg["z_score"])
    q_days = int(cfg["reorder_quantity_days"])

    # Dùng dự báo trong L ngày đầu để ước lượng nhu cầu mong đợi trong lead time
    fc_lead = forecast_values[:L] if len(forecast_values) >= L else forecast_values
    d_bar = float(np.mean(fc_lead))
    sigma_d = float(np.std(history_sales, ddof=1)) if len(history_sales) > 1 else 0.0

    ss = calc_safety_stock_normal(sigma_d, L, z)
    rop = calc_rop(d_bar, L, ss)
    order_qty = max(float(np.mean(forecast_values)) * q_days, 1.0)

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": round(sigma_d, 4),
        "safety_stock": round(ss, 4),
        "rop": round(rop, 4),
        "order_qty": round(order_qty, 4),
    }


def policy_quantile_forecast(
    forecast_values: np.ndarray,
    quantile_90: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Chính sách xác suất: SS từ phân vị dự báo của Chronos, không giả định chuẩn.

    Ý tưởng: Thay vì dùng z * sigma * sqrt(L), dùng trực tiếp phân vị cao
    của dự báo trong lead time để làm ROP.

    ROP = sum(q_0.9, L ngày đầu) — đây là mức tồn kho đảm bảo service level ~90%.
    SS  = ROP - d_bar * L = phần đệm phân vị so với kỳ vọng.

    Lưu ý: Đây là giả định thô (quantile của tổng ≠ tổng quantile), nhưng được
    dùng vì: (1) không cần giả định phân phối chuẩn, (2) trực quan, (3) phục vụ
    so sánh định tính giữa các chính sách. Ghi rõ trong báo cáo.

    Args:
        forecast_values: Mảng dự báo điểm (q_0.5) cho horizon = 28 ngày.
        quantile_90: Mảng dự báo phân vị 0.9 cho horizon = 28 ngày.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        dict với keys: d_bar, safety_stock, rop, order_qty.
    """
    L = int(cfg["lead_time_days"])
    q_days = int(cfg["reorder_quantity_days"])

    fc_lead = forecast_values[:L] if len(forecast_values) >= L else forecast_values
    q90_lead = quantile_90[:L] if len(quantile_90) >= L else quantile_90

    d_bar = float(np.mean(fc_lead))
    # ROP = sum của phân vị 0.9 trong L ngày lead time (đơn giản hóa)
    rop = float(np.sum(q90_lead))
    ss = max(rop - d_bar * L, 0.0)

    order_qty = max(float(np.mean(forecast_values)) * q_days, 1.0)

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": None,  # không dùng trong chính sách này
        "safety_stock": round(ss, 4),
        "rop": round(rop, 4),
        "order_qty": round(order_qty, 4),
    }


def compute_policy_params(
    item_id: str,
    fold: int,
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    policy_name: str,
    cfg: dict,
) -> Dict[str, float]:
    """Helper tổng hợp: tính tham số chính sách cho 1 SKU và 1 fold.

    Args:
        item_id: Mã SKU.
        fold: Số fold backtest (0..12).
        history_df: DataFrame lịch sử với cột ['item_id', 'sales'].
        forecast_df: DataFrame dự báo của fold với cột ['item_id', 'forecast', 'q_0.9'] (optional).
        policy_name: Tên chính sách ('baseline', 'point_forecast', 'quantile_forecast').
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        Dict tham số chính sách.
    """
    hist = history_df[history_df["item_id"] == item_id]["sales"].values.astype(float)
    if len(hist) == 0:
        raise ValueError(f"Không có lịch sử bán hàng cho SKU {item_id}")

    if policy_name == "baseline":
        return policy_baseline(hist, cfg)

    fc_item = forecast_df[forecast_df["item_id"] == item_id].sort_values("date")
    fc_vals = fc_item["forecast"].values.astype(float)
    if len(fc_vals) == 0:
        raise ValueError(f"Không có dự báo cho SKU {item_id}, fold {fold}")

    if policy_name == "point_forecast":
        return policy_point_forecast(fc_vals, hist, cfg)

    if policy_name == "quantile_forecast":
        if "q_0.9" not in fc_item.columns:
            raise ValueError(
                f"Cột q_0.9 không tồn tại trong dự báo — chính sách quantile_forecast "
                f"chỉ áp dụng cho mô hình có phân vị (Chronos)."
            )
        q90 = fc_item["q_0.9"].values.astype(float)
        return policy_quantile_forecast(fc_vals, q90, cfg)

    raise ValueError(f"Tên chính sách không hợp lệ: '{policy_name}'. "
                     f"Chọn: 'baseline', 'point_forecast', 'quantile_forecast'.")
