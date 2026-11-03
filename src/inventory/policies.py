"""Các chính sách tồn kho (Reorder Point & Safety Stock) — Phase 5.

Các chính sách được so sánh:
  1. 'baseline'             : ROP từ nhu cầu trung bình & độ lệch chuẩn
                              trong cửa sổ gần nhất trước origin.
  2. 'baseline_full_history': như baseline nhưng dùng toàn bộ lịch sử
                              (giữ lại để đối chiếu, KHÔNG phải chỉ số chính).
  3. 'point_forecast'       : ROP từ điểm dự báo (median) của mô hình ML/TSFM.
  4. 'quantile_forecast'    : ROP là phân vị của TỔNG nhu cầu trong lead time,
                              xấp xỉ bằng phân phối chuẩn của tổng các ngày.

Công thức chung (theo thuyết minh AGENTS.md):
  ROP = d_bar * L + SS
  SS  = z * sigma_d * sqrt(L)   (baseline & point_forecast)

Lưu ý về phân vị của tổng:
  Không dùng sum(q_alpha của từng ngày) làm ROP vì đó không phải phân vị
  của tổng nhu cầu L ngày (trừ trường hợp comonotonic). Ở đây xấp xỉ tổng
  L ngày bằng phân phối chuẩn:
      mu_L    = sum_t q_0.5(t)
      sigma_t = (q_0.9(t) - q_0.1(t)) / (2 * z_0.9)
      sigma_L = sqrt(sum_t sigma_t^2)        (giả định độc lập giữa các ngày)
      ROP     = mu_L + z_service * sigma_L
  Đây là xấp xỉ có ghi rõ giả định, không phải phân vị chính xác.

Tất cả tham số đọc từ config dict, KHÔNG hard-code.
"""

from statistics import NormalDist
from typing import Dict, Optional
import numpy as np
import pandas as pd

# z của phân vị chuẩn 0.9, dùng để quy đổi khoảng q0.1..q0.9 thành sigma.
Z_QUANTILE_90 = NormalDist().inv_cdf(0.9)


def load_assumptions(config_path: str = "configs/assumptions.yaml") -> dict:
    """Đọc file cấu hình giả định tồn kho (UTF-8)."""
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def z_from_service_level(service_level: float) -> float:
    """Tính z từ mức dịch vụ (cycle service level) — không hard-code.

    Args:
        service_level: Xác suất không thiếu hàng trong chu kỳ bổ sung (0, 1).

    Returns:
        z sao cho P(Z <= z) = service_level với Z ~ N(0, 1).
    """
    level = float(service_level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"service_level phải nằm trong (0, 1), nhận: {level}")
    return float(NormalDist().inv_cdf(level))


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


def _recent_window(values: np.ndarray, window_days: Optional[int]) -> np.ndarray:
    """Lấy `window_days` quan sát gần nhất; window_days <= 0 nghĩa là toàn bộ."""
    arr = np.asarray(values, dtype=float)
    if window_days is None:
        return arr
    window = int(window_days)
    if window <= 0:
        return arr
    return arr[-window:]


def calc_initial_inventory(history_sales: np.ndarray, cfg: dict) -> float:
    """Tồn kho ban đầu dùng chung cho MỌI chính sách.

    Chỉ dùng nhu cầu THỰC đã biết trước origin, không dùng dự báo:
        tồn kho ban đầu = trung bình nhu cầu `initial_inventory_lookback_days`
                          ngày gần nhất * `initial_inventory_days`.

    Args:
        history_sales: Chuỗi doanh số thực hàng ngày đến hết ngày origin.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        Tồn kho ban đầu (units), không âm.
    """
    hist = np.asarray(history_sales, dtype=float)
    if len(hist) == 0:
        raise ValueError("Không có lịch sử để tính tồn kho ban đầu")

    lookback = int(cfg.get("initial_inventory_lookback_days", 28))
    days = int(cfg["initial_inventory_days"])
    recent = _recent_window(hist, lookback)
    init_inv = float(np.mean(recent)) * days
    return float(max(init_inv, 0.0))


def calc_order_qty(forecast_values: np.ndarray, cfg: dict) -> float:
    """Lượng đặt hàng Q = kỳ vọng nhu cầu trong `reorder_quantity_days` ngày.

    Với kỳ vọng, tổng của các ngày bằng tổng kỳ vọng từng ngày (tính tuyến
    tính). Vì vậy dùng tổng các điểm dự báo (median) của những ngày đầu là
    hợp lệ; KHÔNG dùng tổng phân vị cho Q.

    Args:
        forecast_values: Mảng dự báo điểm (median) cho toàn horizon.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        Q (units), tối thiểu 1.0.
    """
    q_days = int(cfg["reorder_quantity_days"])
    fc = np.asarray(forecast_values, dtype=float)
    if len(fc) == 0:
        return 1.0
    if len(fc) >= q_days:
        expected = float(np.sum(fc[:q_days]))
    else:
        expected = float(np.mean(fc) * q_days)
    return float(max(expected, 1.0))


def policy_baseline(
    history_sales: np.ndarray,
    cfg: dict,
    window_days: Optional[int] = None,
) -> Dict[str, float]:
    """Chính sách cơ sở: ROP từ trung bình & độ lệch chuẩn lịch sử gần đây.

    Không sử dụng bất kỳ mô hình dự báo nào. Chỉ dùng dữ liệu đến origin.
    SS = z * std(history) * sqrt(L).

    Args:
        history_sales: Chuỗi doanh số lịch sử hàng ngày đến hết origin.
        cfg: Dict cấu hình từ assumptions.yaml.
        window_days: Độ dài cửa sổ. None = lấy `baseline_window_days` trong
            config; 0 hoặc số âm = dùng toàn bộ lịch sử (biến thể đối chiếu).

    Returns:
        dict với keys: d_bar, sigma_d, safety_stock, rop, order_qty,
        window_days_used, n_history_used.
    """
    if window_days is None:
        window_days = int(cfg.get("baseline_window_days", 91))

    L = int(cfg["lead_time_days"])
    z = z_from_service_level(cfg["service_level"])
    q_days = int(cfg["reorder_quantity_days"])

    hist = _recent_window(history_sales, window_days)
    if len(hist) == 0:
        raise ValueError("Không có lịch sử bán hàng để tính baseline")

    d_bar = float(np.mean(hist))
    sigma_d = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0

    ss = calc_safety_stock_normal(sigma_d, L, z)
    rop = calc_rop(d_bar, L, ss)
    order_qty = max(d_bar * q_days, 1.0)

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": round(sigma_d, 4),
        "safety_stock": round(ss, 4),
        "rop": round(rop, 4),
        "order_qty": round(order_qty, 4),
        "window_days_used": int(window_days),
        "n_history_used": int(len(hist)),
    }


def policy_point_forecast(
    forecast_values: np.ndarray,
    history_sales: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Chính sách điểm dự báo: ROP từ trung bình dự báo điểm + SS từ lịch sử.

    d_bar = mean(forecast) trong `lead_time_days` ngày đầu tiên.
    SS    = z * std(history_sales) * sqrt(L)  [vẫn dùng lịch sử để ước lượng sigma].

    Args:
        forecast_values: Mảng dự báo điểm (median) cho horizon = 28 ngày.
        history_sales: Chuỗi doanh số lịch sử để ước lượng sigma_d.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        dict với keys: d_bar, sigma_d, safety_stock, rop, order_qty.
    """
    L = int(cfg["lead_time_days"])
    z = z_from_service_level(cfg["service_level"])

    fc = np.asarray(forecast_values, dtype=float)
    if len(fc) == 0:
        raise ValueError("Không có giá trị dự báo")

    # Dùng dự báo trong L ngày đầu để ước lượng nhu cầu mong đợi trong lead time
    fc_lead = fc[:L]
    d_bar = float(np.mean(fc_lead))

    hist = np.asarray(history_sales, dtype=float)
    sigma_d = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0

    ss = calc_safety_stock_normal(sigma_d, L, z)
    rop = calc_rop(d_bar, L, ss)
    order_qty = calc_order_qty(fc, cfg)

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": round(sigma_d, 4),
        "safety_stock": round(ss, 4),
        "rop": round(rop, 4),
        "order_qty": round(order_qty, 4),
    }


def policy_quantile_forecast(
    forecast_values: np.ndarray,
    quantile_10: np.ndarray,
    quantile_90: np.ndarray,
    cfg: dict,
) -> Dict[str, float]:
    """Chính sách phân vị: ROP là phân vị của TỔNG nhu cầu trong lead time.

    Cách xử lý (xấp xỉ có ghi rõ giả định):
      1. Mỗi ngày t, lấy mu_t = q_0.5(t) và
         sigma_t = (q_0.9(t) - q_0.1(t)) / (2 * 1.2816).
      2. Giả định nhu cầu các ngày độc lập, tổng L ngày có:
             mu_L    = sum_t mu_t
             sigma_L = sqrt(sum_t sigma_t^2)
      3. ROP = mu_L + z(service_level) * sigma_L
         SS  = z(service_level) * sigma_L

    Không dùng sum_t q_0.9(t): đó là cận trên bảo thủ (comonotonic), không
    phải phân vị của tổng khi các ngày không hoàn toàn phụ thuộc nhau.

    Args:
        forecast_values: Dự báo điểm (q_0.5) cho horizon.
        quantile_10: Dự báo q_0.1 cùng horizon.
        quantile_90: Dự báo q_0.9 cùng horizon.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        dict với keys: d_bar, sigma_d, safety_stock, rop, order_qty,
        sigma_lead_time, method.
    """
    L = int(cfg["lead_time_days"])
    z = z_from_service_level(cfg["service_level"])

    mu = np.maximum(np.asarray(forecast_values, dtype=float), 0.0)
    q10 = np.maximum(np.asarray(quantile_10, dtype=float), 0.0)
    q90 = np.maximum(np.asarray(quantile_90, dtype=float), 0.0)
    if not (len(mu) == len(q10) == len(q90)):
        raise ValueError("forecast, q_0.1 và q_0.9 phải cùng độ dài")
    if len(mu) == 0:
        raise ValueError("Không có giá trị dự báo")

    # Sigma từng ngày từ khoảng phân vị 0.1 - 0.9 (giả định chuẩn)
    sigma_daily = np.maximum((q90 - q10) / (2.0 * Z_QUANTILE_90), 0.0)

    mu_lead = float(np.sum(mu[:L]))
    var_lead = float(np.sum(np.square(sigma_daily[:L])))
    sigma_lead = float(np.sqrt(var_lead))

    d_bar = mu_lead / L
    ss = z * sigma_lead
    rop = mu_lead + ss
    order_qty = calc_order_qty(mu, cfg)

    return {
        "d_bar": round(d_bar, 4),
        "sigma_d": round(float(np.mean(sigma_daily[:L])), 4),
        "sigma_lead_time": round(sigma_lead, 4),
        "safety_stock": round(float(ss), 4),
        "rop": round(float(max(rop, 0.0)), 4),
        "order_qty": round(order_qty, 4),
        "method": "normal_sum_q10_q90",
    }


def build_oracle_paths(
    actual_demand: np.ndarray,
    lead_time_days: int,
    reorder_quantity_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Chính sách tham chiếu Oracle — dùng nhu cầu THỰC của tương lai.

    ĐÂY KHÔNG PHẢI LÀ MÔ HÌNH DỰ BÁO. Oracle chỉ là **cận dưới chi phí** để
    trả lời câu hỏi "nếu biết trước nhu cầu thì chi phí tồn kho tối thiểu
    trong cấu trúc (s, Q) này là bao nhiêu". Không được dùng Oracle để kết
    luận mô hình nào tốt hơn.

    Cách xây dựng (perfect foresight trong horizon của fold):
      - ROP_t = tổng nhu cầu thực của `lead_time_days` ngày kế tiếp,
                tức demand[t+1 : t+1+L] (nhu cầu trong thời gian chờ hàng).
      - Q_t   = tổng nhu cầu thực của `reorder_quantity_days` ngày bắt đầu
                từ lúc hàng về, tức demand[t+L : t+L+q_days].
      - Những ngày cuối fold không còn đủ dữ liệu tương lai: dùng phần còn
                lại; nếu rỗng thì ROP = 0 và Q = 1 (đơn đó không kịp về
                trong fold nên không ảnh hưởng kết quả).

    Args:
        actual_demand: Nhu cầu thực của fold (đã áp dụng stress nếu có).
        lead_time_days: Lead time L (ngày).
        reorder_quantity_days: Chu kỳ đặt hàng q_days (ngày).

    Returns:
        (rop_path, qty_path) — hai mảng cùng độ dài với actual_demand.
    """
    demand = np.asarray(actual_demand, dtype=float)
    T = len(demand)
    L = int(lead_time_days)
    q_days = int(reorder_quantity_days)
    if L < 1 or q_days < 1:
        raise ValueError("lead_time_days và reorder_quantity_days phải >= 1")

    rop_path = np.zeros(T, dtype=float)
    qty_path = np.zeros(T, dtype=float)

    for t in range(T):
        future_lt = demand[t + 1: t + 1 + L]
        rop_path[t] = float(np.sum(future_lt)) if len(future_lt) else 0.0

        future_q = demand[t + L: t + L + q_days]
        qty_path[t] = max(float(np.sum(future_q)), 1.0) if len(future_q) else 1.0

    return rop_path, qty_path


def compute_policy_params(
    item_id: str,
    fold: int,
    history_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame],
    policy_name: str,
    cfg: dict,
) -> Dict[str, float]:
    """Helper tổng hợp: tính tham số chính sách cho 1 SKU và 1 fold.

    Args:
        item_id: Mã SKU.
        fold: Số fold backtest (0..12).
        history_df: DataFrame lịch sử với cột ['item_id', 'sales'], ≤ origin.
        forecast_df: DataFrame dự báo của fold (None với baseline).
        policy_name: 'baseline', 'baseline_full_history', 'point_forecast',
            hoặc 'quantile_forecast'.
        cfg: Dict cấu hình từ assumptions.yaml.

    Returns:
        Dict tham số chính sách.
    """
    hist = history_df[history_df["item_id"] == item_id]["sales"].values.astype(float)
    if len(hist) == 0:
        raise ValueError(f"Không có lịch sử bán hàng cho SKU {item_id}")

    if policy_name == "baseline":
        return policy_baseline(hist, cfg)

    if policy_name == "baseline_full_history":
        # window_days=0: dùng toàn bộ lịch sử (biến thể đối chiếu)
        return policy_baseline(hist, cfg, window_days=0)

    if forecast_df is None:
        raise ValueError(f"Chính sách '{policy_name}' cần dữ liệu dự báo")

    fc_item = forecast_df[forecast_df["item_id"] == item_id].sort_values("date")
    fc_vals = fc_item["forecast"].values.astype(float)
    if len(fc_vals) == 0:
        raise ValueError(f"Không có dự báo cho SKU {item_id}, fold {fold}")

    if policy_name == "point_forecast":
        return policy_point_forecast(fc_vals, hist, cfg)

    if policy_name == "quantile_forecast":
        missing = [c for c in ("q_0.1", "q_0.9") if c not in fc_item.columns]
        if missing:
            raise ValueError(
                f"Thiếu cột {missing} — quantile_forecast chỉ áp dụng cho mô hình "
                f"có phân vị q_0.1 và q_0.9."
            )
        q10 = fc_item["q_0.1"].values.astype(float)
        q90 = fc_item["q_0.9"].values.astype(float)
        return policy_quantile_forecast(fc_vals, q10, q90, cfg)

    raise ValueError(
        f"Tên chính sách không hợp lệ: '{policy_name}'. "
        f"Chọn: 'baseline', 'baseline_full_history', 'point_forecast', "
        f"'quantile_forecast'."
    )
