"""Test cho các sửa nhỏ sau chẩn đoán Phase 5.

Bao phủ:
  (i)  MA91 + σ91 + Q91 phải trùng Baseline từng dòng.
  (ii) Khi σ_L của quantile bằng σ_hist·√L thì ROP quantile = ROP point.
  (iii) Monotone rearrangement sắp xếp phân vị và giữ bản gốc.
  (iv) Sigma sai số chỉ dùng các fold TRƯỚC origin; fold 0 fallback σ91.
  (v)  Chronos-Bolt không có q0.95 thật → chính sách quantile không khả dụng.
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_processed_daily
from src.forecasting.backtest import load_backtest_config
from src.forecasting.features import _ensure_d_num
from src.inventory.policies import (
    calc_initial_inventory,
    load_assumptions,
    policy_baseline,
    policy_point_forecast,
    policy_point_forecast_window_sigma,
    policy_quantile_forecast,
    policy_quantile_forecast_sigma,
)
from src.inventory.prediction_prep import (
    compute_error_sigmas,
    quantile_availability,
    rearrange_quantiles,
)
from src.inventory.simulator import run_simulation_for_item


def _simulate(params: dict, history: np.ndarray, actual: np.ndarray, cfg: dict, fold: int, item_id: str = "X"):
    """Chạy simulator với tham số chính sách đã tính."""
    return run_simulation_for_item(
        item_id=item_id,
        fold=fold,
        actual_demand=actual,
        rop=params["rop"],
        order_qty=params["order_qty"],
        lead_time=int(cfg["lead_time_days"]),
        initial_inventory=calc_initial_inventory(history, cfg),
        holding_cost_per_unit_per_day=float(cfg["holding_cost_per_unit_per_day"]),
        ordering_cost=float(cfg["ordering_cost"]),
        shortage_cost_per_unit=float(cfg["shortage_cost_per_unit"]),
    )


# =========================================================
# (i) MA91 + σ91 + Q91 == Baseline từng dòng
# =========================================================
def test_ma91_sigma91_q91_equals_baseline_row_by_row():
    """Dự báo MA91 hằng số + σ cửa sổ 91 + Q từ chính dự báo phải bằng Baseline."""
    daily = _ensure_d_num(load_processed_daily("data/processed/sales_daily.parquet"))
    bt = load_backtest_config("configs/backtest.yaml")
    cfg = load_assumptions("configs/assumptions.yaml")

    horizon = int(bt["horizon"])
    initial_origin = int(bt["initial_origin"])
    step = int(bt["step_size"])
    items = sorted(daily["item_id"].unique())
    window = int(cfg["baseline_window_days"])

    checked = 0
    mismatches = []
    for fold_k in range(int(bt["num_folds"])):
        origin = initial_origin + fold_k * step
        hist_df = daily[daily["d_num"] <= origin]
        for item_id in items:
            hist = hist_df[hist_df["item_id"] == item_id]["sales"].to_numpy(dtype=float)
            actual = daily[
                (daily["item_id"] == item_id)
                & (daily["d_num"] > origin)
                & (daily["d_num"] <= origin + horizon)
            ].sort_values("d_num")["sales"].to_numpy(dtype=float)
            if len(hist) == 0 or len(actual) == 0:
                continue

            mean91 = float(np.mean(hist[-window:]))
            forecast_ma91 = np.full(horizon, mean91, dtype=float)

            base = policy_baseline(hist, cfg, window_days=window)
            variant = policy_point_forecast_window_sigma(
                forecast_ma91, hist, cfg, window_days=window
            )

            same_params = (
                variant["rop"] == base["rop"]
                and variant["order_qty"] == base["order_qty"]
                and variant["safety_stock"] == base["safety_stock"]
            )
            sim_base = _simulate(base, hist, actual, cfg, fold_k, item_id)
            sim_variant = _simulate(variant, hist, actual, cfg, fold_k, item_id)
            same_rows = (
                sim_base["cost_total"] == sim_variant["cost_total"]
                and sim_base["total_short"] == sim_variant["total_short"]
                and sim_base["num_orders"] == sim_variant["num_orders"]
            )
            checked += 1
            if not (same_params and same_rows):
                mismatches.append(
                    {
                        "fold": fold_k,
                        "item_id": item_id,
                        "base_rop": base["rop"],
                        "variant_rop": variant["rop"],
                        "base_qty": base["order_qty"],
                        "variant_qty": variant["order_qty"],
                        "cost_diff": sim_variant["cost_total"] - sim_base["cost_total"],
                    }
                )

    assert checked == int(bt["num_folds"]) * len(items) == 780
    assert not mismatches, f"{len(mismatches)} dòng lệch Baseline: {mismatches[:3]}"


# =========================================================
# (ii) σ_L khớp σ_hist·√L → ROP quantile == ROP point
# =========================================================
def test_quantile_rop_equals_point_when_sigma_matches():
    """Craft q0.1/q0.9 sao cho σ_L = σ_hist·√L, rồi so ROP hai chính sách."""
    cfg = {
        "lead_time_days": 7,
        "reorder_quantity_days": 14,
        "service_level": 0.95,
    }
    hist = np.array([80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0])  # std > 0
    sigma_hist = float(np.std(hist, ddof=1))

    forecast = np.full(28, 100.0, dtype=float)
    z90 = 1.2815515655446004
    q10 = forecast - z90 * sigma_hist
    q90 = forecast + z90 * sigma_hist

    point = policy_point_forecast(forecast, hist, cfg)
    quant = policy_quantile_forecast(forecast, q10, q90, cfg)
    quant_hist = policy_quantile_forecast_sigma(forecast, q10, q90, cfg, sigma_d=sigma_hist)

    # Các hàm policy trả giá trị đã làm tròn 4 chữ số thập phân.
    assert quant["sigma_lead_time"] == pytest.approx(sigma_hist * np.sqrt(7), abs=1e-4)
    assert quant["rop"] == pytest.approx(point["rop"], abs=2e-4)
    assert quant["order_qty"] == pytest.approx(point["order_qty"], abs=1e-4)
    # Biến thể (c): μ từ q0.5 + σ lịch sử cũng phải bằng point.
    assert quant_hist["rop"] == pytest.approx(point["rop"], abs=2e-4)
    assert quant_hist["order_qty"] == pytest.approx(point["order_qty"], abs=1e-4)


# =========================================================
# (iii) Monotone rearrangement
# =========================================================
def test_monotone_rearrangement_keeps_original_and_counts_rows():
    pred = pd.DataFrame(
        {
            "forecast": [5.0, 2.0, 3.0],
            "q_0.1": [10.0, 1.0, 1.0],
            "q_0.5": [5.0, 2.0, 3.0],
            "q_0.9": [6.0, 3.0, 2.0],
            "q_0.95": [7.0, 4.0, 4.0],
        }
    )
    out, stats = rearrange_quantiles(pred, keep_original=True)

    # Bản gốc còn nguyên
    assert out["q_0.1_raw"].tolist() == [10.0, 1.0, 1.0]
    assert out["q_0.5_raw"].tolist() == [5.0, 2.0, 3.0]
    # Dòng 0 và dòng 2 bị đảo; dòng 1 vốn đã đúng.
    assert out[["q_0.1", "q_0.5", "q_0.9", "q_0.95"]].values.tolist() == [
        [5.0, 6.0, 7.0, 10.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
    assert stats["n_rows_total"] == 3
    assert stats["n_rows_rearranged"] == 2
    assert stats["n_q_0.1_gt_q_0.5_before"] == 1
    assert stats["n_q_0.5_gt_q_0.9_before"] == 1
    assert stats["n_q_0.1_gt_q_0.5_after"] == 0
    assert stats["n_q_0.5_gt_q_0.9_after"] == 0
    # forecast là median nên đồng bộ theo q_0.5 đã sắp xếp
    assert out["forecast"].tolist() == [6.0, 2.0, 2.0]
    assert stats["forecast_changed_rows"] == 2


# =========================================================
# (iv) Sigma sai số chỉ dùng fold trước origin
# =========================================================
def test_error_sigma_uses_only_prior_folds_and_fold0_fallback():
    n_days = 120
    daily = pd.DataFrame(
        {
            "item_id": ["A"] * n_days,
            "d_num": np.arange(1, n_days + 1),
            "sales": np.arange(n_days, dtype=float),
        }
    )
    rows = []
    errors_by_fold = {0: [1, 2, 3, 4, 5], 1: [2, 2, 2, 2, 2], 2: [0, 0, 0, 0, 0]}
    for fold_k, errs in errors_by_fold.items():
        for i, e in enumerate(errs):
            rows.append(
                {
                    "fold": fold_k,
                    "item_id": "A",
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fold_k * 5 + i),
                    "actual": 100.0 + e,
                    "forecast": 100.0,
                }
            )
    pred = pd.DataFrame(rows)
    bt = {"initial_origin": 100, "step_size": 5}

    sigma = compute_error_sigmas(
        pred, daily, bt, model_name="Test", baseline_window_days=91
    ).set_index("fold")

    # fold 0: chưa có fold trước → fallback σ91 của lịch sử
    hist_at_fold0 = daily[daily["d_num"] <= 100]["sales"].to_numpy(dtype=float)[-91:]
    assert bool(sigma.loc[0, "fallback_used"]) is True
    assert sigma.loc[0, "n_error_obs"] == 0
    assert sigma.loc[0, "sigma_error"] == pytest.approx(
        float(np.std(hist_at_fold0, ddof=1)), abs=1e-6
    )

    # fold 1: chỉ dùng sai số fold 0
    assert bool(sigma.loc[1, "fallback_used"]) is False
    assert sigma.loc[1, "n_error_obs"] == 5
    assert sigma.loc[1, "sigma_error"] == pytest.approx(
        float(np.std(errors_by_fold[0], ddof=1)), abs=1e-6
    )

    # fold 2: dùng fold 0 + fold 1, không dùng fold 2 (fold hiện tại)
    combined = errors_by_fold[0] + errors_by_fold[1]
    assert sigma.loc[2, "n_error_obs"] == 10
    assert sigma.loc[2, "sigma_error"] == pytest.approx(
        float(np.std(combined, ddof=1)), abs=1e-6
    )


# =========================================================
# (v) Bolt không có q0.95 thật
# =========================================================
def test_quantile_availability_marks_bolt_unavailable():
    bolt_path = Path("results/predictions/chronos_bolt_predictions.parquet")
    hist_path = Path("results/predictions/chronos2_history_predictions.parquet")
    if not bolt_path.exists() or not hist_path.exists():
        pytest.skip("Thiếu file dự báo Chronos để kiểm tra phân vị.")

    bolt = pd.read_parquet(bolt_path)
    history = pd.read_parquet(hist_path)

    bolt_avail = quantile_availability("Chronos_Bolt", bolt)
    hist_avail = quantile_availability("Chronos2_History", history)

    assert bolt_avail["has_q_0.95"] is True
    assert bolt_avail["share_q90_equal_q95"] == pytest.approx(1.0)
    assert bolt_avail["quantile_policy_available"] is False

    assert hist_avail["quantile_policy_available"] is True
