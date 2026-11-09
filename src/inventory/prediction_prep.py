"""Chuẩn bị dự báo trước khi mô phỏng tồn kho — Phase 5 hậu chẩn đoán.

Module này KHÔNG sửa file dự báo gốc trên đĩa. Nó chỉ:
  1. Sắp xếp lại (monotone rearrangement) các phân vị bị đảo thứ tự trong
     từng dòng, giữ bản gốc ở các cột `*_raw`.
  2. Xác định mô hình có thực sự xuất ra phân vị cần dùng hay không
     (Chronos-Bolt không xuất q0.95 — code cũ chép q0.9 sang q0.95).
  3. Ước lượng sigma từ sai số dự báo đã quan sát ở các fold TRƯỚC origin
     (fold 0 dùng sigma 91 ngày làm fallback, không dùng dữ liệu tương lai).

Chạy kiểm tra nhanh:
  $env:PYTHONPATH='.'
  .\\.venv\\Scripts\\python -m src.inventory.prediction_prep
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

# Thứ tự phân vị tăng dần. Chỉ các cột có mặt mới được sắp xếp.
QUANTILE_COLS = ["q_0.1", "q_0.5", "q_0.9", "q_0.95"]

# Các cặp cạnh nhau dùng để đếm vi phạm thứ tự.
ADJACENT_PAIRS = [
    ("q_0.1", "q_0.5"),
    ("q_0.5", "q_0.9"),
    ("q_0.9", "q_0.95"),
]


def rearrange_quantiles(
    pred_df: pd.DataFrame,
    keep_original: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Sắp xếp lại phân vị từng dòng để đảm bảo q0.1 ≤ q0.5 ≤ q0.9 ≤ q0.95.

    Đây là "monotone rearrangement" thực dụng cho các mức phân vị đã cho:
    với mỗi dòng, sắp xếp các giá trị phân vị tăng dần rồi gán lại theo đúng
    thứ tự mức. Cách này không thay đổi tập giá trị, chỉ sửa các dòng bị đảo.

    Bản gốc được giữ ở các cột `<tên>_raw` nếu `keep_original=True`.
    Vì điểm dự báo của mô hình được định nghĩa là trung vị q0.5, sau khi sắp
    xếp cột `forecast` được đồng bộ bằng q_0.5 đã sắp xếp.

    Args:
        pred_df: DataFrame dự báo, có thể chứa các cột q_0.1/q_0.5/q_0.9/q_0.95.
        keep_original: Giữ bản gốc ở các cột `_raw`.

    Returns:
        (DataFrame đã sắp xếp, dict thống kê số dòng/số ô bị sửa).
    """
    df = pred_df.copy()
    present = [c for c in QUANTILE_COLS if c in df.columns]

    stats: Dict[str, float] = {
        "n_rows_total": int(len(df)),
        "n_quantile_columns": len(present),
        "n_rows_rearranged": 0,
        "n_cells_changed": 0,
        "forecast_changed_rows": 0,
    }
    for a, b in ADJACENT_PAIRS:
        stats[f"n_{a}_gt_{b}_before"] = int(
            (df[a] > df[b]).sum() if a in df.columns and b in df.columns else 0
        )

    if len(present) == 0:
        return df, stats

    original_values = df[present].to_numpy(dtype=float)
    sorted_values = np.sort(original_values, axis=1)

    stats["n_cells_changed"] = int(np.sum(original_values != sorted_values))
    stats["n_rows_rearranged"] = int(
        np.any(original_values != sorted_values, axis=1).sum()
    )

    original_forecast = df["forecast"].to_numpy(dtype=float).copy() if "forecast" in df.columns else None

    if keep_original:
        for col in present:
            df[f"{col}_raw"] = df[col]

    for j, col in enumerate(present):
        df[col] = sorted_values[:, j]

    # forecast là trung vị q0.5; đồng bộ sau sắp xếp để hai nguồn không lệch.
    if "q_0.5" in df.columns:
        df["forecast"] = df["q_0.5"]
        if original_forecast is not None:
            stats["forecast_changed_rows"] = int(
                np.sum(original_forecast != df["q_0.5"].to_numpy(dtype=float))
            )

    for a, b in ADJACENT_PAIRS:
        stats[f"n_{a}_gt_{b}_after"] = int(
            (df[a] > df[b]).sum() if a in df.columns and b in df.columns else 0
        )

    return df, stats


def quantile_availability(
    model_name: str,
    pred_df: pd.DataFrame,
) -> Dict[str, object]:
    """Kiểm tra mô hình có thực sự xuất ra bộ phân vị dùng được hay không.

    Quy tắc:
      - Cần có q_0.1 và q_0.9.
      - q_0.95 chỉ được coi là "có thật" nếu KHÔNG phải bản sao của q_0.9
        trên toàn bộ các dòng. Chronos-Bolt trong repo này chép q_0.9 sang
        q_0.95 (xem `src/forecasting/chronos_model.py`), nên bị đánh dấu
        không khả dụng thay vì giả vờ có q_0.95.

    Returns:
        Dict gồm các cờ và lý do để ghi vào file kết quả.
    """
    has_q10 = "q_0.1" in pred_df.columns
    has_q50 = "q_0.5" in pred_df.columns
    has_q90 = "q_0.9" in pred_df.columns
    has_q95 = "q_0.95" in pred_df.columns

    copy_share: Optional[float] = None
    if has_q90 and has_q95 and len(pred_df) > 0:
        copy_share = float((pred_df["q_0.9"] == pred_df["q_0.95"]).mean())

    if not (has_q10 and has_q90):
        available = False
        reason = "không có cột q_0.1/q_0.9"
    elif copy_share is not None and copy_share >= 1.0:
        available = False
        reason = "q_0.95 là bản sao của q_0.9 (mô hình không xuất q_0.95)"
    else:
        available = True
        reason = "có q_0.1/q_0.9 và q_0.95 không phải bản sao"

    return {
        "model": model_name,
        "has_q_0.1": has_q10,
        "has_q_0.5": has_q50,
        "has_q_0.9": has_q90,
        "has_q_0.95": has_q95,
        "share_q90_equal_q95": round(copy_share, 6) if copy_share is not None else np.nan,
        "quantile_policy_available": available,
        "reason": reason,
    }


def compute_error_sigmas(
    pred_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    bt_cfg: dict,
    model_name: str,
    baseline_window_days: int = 91,
) -> pd.DataFrame:
    """Ước lượng sigma ngày từ sai số dự báo của các fold TRƯỚC origin.

    Với mỗi (fold k, SKU):
      - Lấy tất cả sai số (actual − forecast) của SKU đó ở các fold < k.
      - sigma = std(errors, ddof=1).
      - Nếu chưa có đủ 2 quan sát (fold 0, hoặc SKU thiếu dữ liệu), fallback
        về std 91 ngày gần nhất trước origin của fold k.

    Không dùng actual của fold hiện tại hay tương lai → không rò rỉ dữ liệu.

    Args:
        pred_df: Dự báo đã chuẩn bị (có fold, item_id, actual, forecast).
        daily_df: Dữ liệu ngày đã có cột `d_num`, dùng cho fallback sigma 91.
        bt_cfg: Cấu hình backtest (initial_origin, step_size).
        model_name: Tên mô hình để ghi vào kết quả.
        baseline_window_days: Cửa sổ fallback (mặc định 91, giống Baseline).

    Returns:
        DataFrame: model, fold, item_id, sigma_error, n_error_obs, fallback_used.
    """
    pred = pred_df.sort_values(["fold", "item_id", "date"]).copy()
    pred["error"] = pred["actual"].astype(float) - pred["forecast"].astype(float)

    initial_origin = int(bt_cfg["initial_origin"])
    step = int(bt_cfg["step_size"])
    items = sorted(pred["item_id"].unique())
    folds = sorted(pred["fold"].unique())

    # Cache lịch sử theo (fold, item) để tránh lọc lặp lại.
    records = []
    for fold_k in folds:
        origin = initial_origin + int(fold_k) * step
        hist_df = daily_df[daily_df["d_num"] <= origin]
        prior = pred[pred["fold"] < fold_k]

        for item_id in items:
            item_prior = prior[prior["item_id"] == item_id]
            errors = item_prior["error"].to_numpy(dtype=float)
            errors = errors[np.isfinite(errors)]

            if len(errors) >= 2:
                sigma = float(np.std(errors, ddof=1))
                n_obs = int(len(errors))
                fallback = False
            else:
                hist = hist_df[hist_df["item_id"] == item_id]["sales"].to_numpy(dtype=float)
                hist = hist[-baseline_window_days:] if len(hist) > baseline_window_days else hist
                sigma = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0
                n_obs = int(len(errors))
                fallback = True

            records.append(
                {
                    "model": model_name,
                    "fold": int(fold_k),
                    "item_id": item_id,
                    "sigma_error": round(sigma, 6),
                    "n_error_obs": n_obs,
                    "fallback_used": fallback,
                }
            )

    return pd.DataFrame(records)


def prepare_model_predictions(
    model_name: str,
    pred_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    bt_cfg: dict,
    baseline_window_days: int = 91,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, Dict[str, float]]:
    """Chuẩn bị đầy đủ cho 1 mô hình: sắp xếp phân vị, kiểm tra khả dụng, sigma sai số."""
    availability = quantile_availability(model_name, pred_df)
    prepared, rearrange_stats = rearrange_quantiles(pred_df, keep_original=True)
    error_sigmas = compute_error_sigmas(
        prepared,
        daily_df,
        bt_cfg,
        model_name=model_name,
        baseline_window_days=baseline_window_days,
    )
    return prepared, availability, error_sigmas, rearrange_stats


if __name__ == "__main__":
    print("Module chuẩn bị dự báo — dùng qua run_simulation.py")
