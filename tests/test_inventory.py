"""Unit tests cho mô-đun tồn kho Phase 5 — chạy trong .venv chính.

Kiểm tra:
  (a) costs.py: holding, ordering, shortage, fill_rate, stockout_rate.
  (b) policies.py: ROP, SS, bốn chính sách, cửa sổ baseline, tồn kho ban đầu.
  (c) simulator.py: dòng tồn kho đúng, không âm, fill rate hợp lệ, tồn kho bảo toàn.
  (d) run_simulation.py: fill rate gộp và cột summary.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.inventory.costs import (
    calc_fill_rate,
    calc_holding_cost,
    calc_ordering_cost,
    calc_shortage_cost,
    calc_stockout_rate,
    calc_total_cost,
)
from src.inventory.policies import (
    build_oracle_paths,
    calc_initial_inventory,
    calc_order_qty,
    calc_rop,
    calc_safety_stock_normal,
    policy_baseline,
    policy_point_forecast,
    policy_quantile_forecast,
    z_from_service_level,
)
from src.inventory.run_simulation import _summarize_results
from src.inventory.simulator import run_simulation_for_item, simulate_inventory


# =========================================================
# (a) Tests cho costs.py
# =========================================================


class TestHoldingCost:
    def test_basic(self):
        inv = [10.0, 5.0, 0.0, 3.0]
        assert calc_holding_cost(inv, h := 0.05) == pytest.approx((10 + 5 + 0 + 3) * h)

    def test_negative_inventory_ignored(self):
        # lost-sales: không có tồn kho âm, nhưng hàm phải robust
        inv = [5.0, -2.0, 3.0]
        result = calc_holding_cost(inv, 0.01)
        assert result == pytest.approx((5 + 0 + 3) * 0.01)

    def test_zero(self):
        assert calc_holding_cost([0, 0, 0], 0.1) == 0.0


class TestOrderingCost:
    def test_basic(self):
        assert calc_ordering_cost(3, 5.0) == pytest.approx(15.0)

    def test_zero_orders(self):
        assert calc_ordering_cost(0, 999.0) == 0.0

    def test_negative_orders_raises(self):
        with pytest.raises(ValueError):
            calc_ordering_cost(-1, 5.0)


class TestShortageCost:
    def test_basic(self):
        assert calc_shortage_cost([2.0, 0.0, 1.0], 5.0) == pytest.approx(15.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            calc_shortage_cost([-1.0], 5.0)


class TestFillRate:
    def test_perfect(self):
        assert calc_fill_rate([10, 5, 3], [10, 5, 3]) == pytest.approx(1.0)

    def test_partial(self):
        result = calc_fill_rate([8, 5, 3], [10, 5, 3])
        assert result == pytest.approx(16 / 18)

    def test_zero_demand(self):
        assert calc_fill_rate([0, 0], [0, 0]) == 1.0

    def test_capped_at_one(self):
        # sold không được > demanded
        assert calc_fill_rate([10, 10], [8, 8]) <= 1.0


class TestStockoutRate:
    def test_basic(self):
        assert calc_stockout_rate(3, 10) == pytest.approx(0.3)

    def test_no_stockout(self):
        assert calc_stockout_rate(0, 10) == 0.0

    def test_zero_days_raises(self):
        with pytest.raises(ValueError):
            calc_stockout_rate(0, 0)


class TestTotalCost:
    def test_sum(self):
        result = calc_total_cost(
            inventory_levels=[10.0, 5.0],
            units_short=[1.0, 0.0],
            num_orders=2,
            holding_cost_per_unit_per_day=0.05,
            ordering_cost=5.0,
            shortage_cost_per_unit=2.0,
        )
        expected_h = (10 + 5) * 0.05
        expected_k = 2 * 5.0
        expected_p = 1.0 * 2.0
        assert result["holding"] == pytest.approx(expected_h, abs=0.001)
        assert result["ordering"] == pytest.approx(expected_k, abs=0.001)
        assert result["shortage"] == pytest.approx(expected_p, abs=0.001)
        assert result["total"] == pytest.approx(expected_h + expected_k + expected_p, abs=0.001)


# =========================================================
# (b) Tests cho policies.py
# =========================================================


CFG = {
    "lead_time_days": 7,
    "reorder_quantity_days": 14,
    "holding_cost_per_unit_per_day": 0.05,
    "ordering_cost": 5.0,
    "shortage_cost_per_unit": 2.0,
    "service_level": 0.95,
    "initial_inventory_days": 7,
    "initial_inventory_lookback_days": 28,
    "baseline_window_days": 91,
}


class TestZFromServiceLevel:
    def test_known_values(self):
        assert z_from_service_level(0.90) == pytest.approx(1.2816, abs=1e-3)
        assert z_from_service_level(0.95) == pytest.approx(1.6449, abs=1e-3)
        assert z_from_service_level(0.99) == pytest.approx(2.3263, abs=1e-3)

    def test_monotonic(self):
        assert z_from_service_level(0.90) < z_from_service_level(0.95) < z_from_service_level(0.99)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            z_from_service_level(0.0)
        with pytest.raises(ValueError):
            z_from_service_level(1.0)


class TestSafetyStockNormal:
    def test_formula(self):
        # SS = z * sigma * sqrt(L)
        ss = calc_safety_stock_normal(sigma_d=2.0, lead_time_days=9, z_score=1.645)
        assert ss == pytest.approx(1.645 * 2.0 * np.sqrt(9))

    def test_non_negative(self):
        ss = calc_safety_stock_normal(sigma_d=0.0, lead_time_days=7, z_score=1.645)
        assert ss >= 0.0


class TestROP:
    def test_formula(self):
        rop = calc_rop(d_bar=5.0, lead_time_days=7, safety_stock=3.0)
        assert rop == pytest.approx(5.0 * 7 + 3.0)

    def test_non_negative(self):
        rop = calc_rop(d_bar=0.0, lead_time_days=7, safety_stock=0.0)
        assert rop >= 0.0


class TestPolicyBaseline:
    def test_returns_valid_keys(self):
        hist = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 2.0, 3.0] * 10)
        result = policy_baseline(hist, CFG)
        assert "rop" in result
        assert "safety_stock" in result
        assert "order_qty" in result
        assert result["rop"] >= 0.0
        assert result["safety_stock"] >= 0.0
        assert result["order_qty"] >= 1.0

    def test_constant_demand(self):
        # Nhu cầu cố định 2.0/ngày, sigma=0
        hist = np.full(100, 2.0)
        result = policy_baseline(hist, CFG)
        assert result["d_bar"] == pytest.approx(2.0)
        assert result["sigma_d"] == pytest.approx(0.0)
        assert result["safety_stock"] == pytest.approx(0.0)
        assert result["rop"] == pytest.approx(2.0 * 7)

    def test_uses_only_recent_window(self):
        # 20 ngày cũ nhu cầu 10, 10 ngày gần nhu cầu 1 → cửa sổ 10 phải bỏ 20 ngày cũ
        hist = np.array([10.0] * 20 + [1.0] * 10)
        result = policy_baseline(hist, CFG, window_days=10)
        assert result["d_bar"] == pytest.approx(1.0)
        assert result["n_history_used"] == 10
        assert result["window_days_used"] == 10

    def test_full_history_variant_uses_all_data(self):
        hist = np.array([10.0] * 20 + [1.0] * 10)
        result = policy_baseline(hist, CFG, window_days=0)
        assert result["d_bar"] == pytest.approx((10.0 * 20 + 1.0 * 10) / 30)
        assert result["n_history_used"] == 30

    def test_default_window_from_config(self):
        hist = np.arange(1.0, 201.0)  # 200 ngày tăng dần
        result = policy_baseline(hist, CFG)
        assert result["n_history_used"] == CFG["baseline_window_days"]
        assert result["d_bar"] == pytest.approx(hist[-91:].mean())


class TestInitialInventory:
    def test_uses_actual_recent_demand(self):
        hist = np.array([100.0] * 100 + [2.0] * 28)
        result = calc_initial_inventory(hist, CFG)
        # Chỉ 28 ngày gần nhất, không bị 100 ngày cũ kéo lên
        assert result == pytest.approx(2.0 * CFG["initial_inventory_days"])

    def test_zero_demand(self):
        assert calc_initial_inventory(np.zeros(50), CFG) == 0.0


class TestOrderQty:
    def test_uses_first_cycle_days(self):
        fc = np.array([1.0] * 14 + [100.0] * 14)
        # Q = tổng 14 ngày đầu, không bị 14 ngày sau kéo lên
        assert calc_order_qty(fc, CFG) == pytest.approx(14.0)

    def test_short_forecast_falls_back_to_mean(self):
        fc = np.array([2.0] * 7)
        assert calc_order_qty(fc, CFG) == pytest.approx(2.0 * 14)


class TestPolicyPointForecast:
    def test_returns_valid_keys(self):
        hist = np.random.RandomState(0).uniform(0, 5, 200)
        forecast = np.random.RandomState(1).uniform(1, 4, 28)
        result = policy_point_forecast(forecast, hist, CFG)
        assert result["rop"] >= 0.0
        assert result["order_qty"] >= 1.0

    def test_higher_forecast_gives_higher_rop(self):
        hist = np.ones(100) * 2.0
        fc_low = np.ones(28) * 2.0
        fc_high = np.ones(28) * 10.0
        result_low = policy_point_forecast(fc_low, hist, CFG)
        result_high = policy_point_forecast(fc_high, hist, CFG)
        # d_bar cao hơn → ROP cao hơn (SS như nhau vì sigma lấy từ hist)
        assert result_high["rop"] > result_low["rop"]


class TestPolicyQuantileForecast:
    @staticmethod
    def _inputs(fc=3.0, q10=1.0, q90=5.0, n=28):
        return np.full(n, fc), np.full(n, q10), np.full(n, q90)

    def test_returns_valid_keys(self):
        fc, q10, q90 = self._inputs()
        result = policy_quantile_forecast(fc, q10, q90, CFG)
        assert "rop" in result
        assert result["rop"] >= 0.0
        assert result["safety_stock"] >= 0.0
        assert result["method"] == "normal_sum_q10_q90"

    def test_rop_is_quantile_of_sum_not_sum_of_quantiles(self):
        fc, q10, q90 = self._inputs(fc=2.0, q10=0.0, q90=4.0)
        result = policy_quantile_forecast(fc, q10, q90, CFG)
        z = z_from_service_level(CFG["service_level"])
        sigma_daily = (4.0 - 0.0) / (2 * 1.2815515655446004)
        expected = 2.0 * 7 + z * np.sqrt(7 * sigma_daily ** 2)
        assert result["rop"] == pytest.approx(expected, abs=0.01)
        # Không được bằng tổng q0.9 (28 đơn vị) — đó là cận trên comonotonic
        assert result["rop"] < 7 * 4.0

    def test_higher_quantile_ninety_gives_higher_rop(self):
        fc, q10, q90_low = self._inputs(q90=4.0)
        _, _, q90_high = self._inputs(q90=8.0)
        low = policy_quantile_forecast(fc, q10, q90_low, CFG)
        high = policy_quantile_forecast(fc, q10, q90_high, CFG)
        assert high["rop"] >= low["rop"]

    def test_higher_service_level_gives_higher_rop(self):
        fc, q10, q90 = self._inputs()
        cfg_90 = dict(CFG, service_level=0.90)
        cfg_99 = dict(CFG, service_level=0.99)
        rop_90 = policy_quantile_forecast(fc, q10, q90, cfg_90)["rop"]
        rop_99 = policy_quantile_forecast(fc, q10, q90, cfg_99)["rop"]
        assert rop_99 >= rop_90

    def test_scaled_quantiles_give_higher_rop(self):
        fc, q10, q90 = self._inputs()
        base = policy_quantile_forecast(fc, q10, q90, CFG)
        scaled = policy_quantile_forecast(fc * 2, q10 * 2, q90 * 2, CFG)
        assert scaled["rop"] >= base["rop"]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            policy_quantile_forecast(np.ones(28), np.ones(27), np.ones(28), CFG)


# =========================================================
# (c) Tests cho simulator.py
# =========================================================


class TestSimulateInventory:
    def test_no_demand(self):
        """Không có nhu cầu → không thiếu hàng, tồn kho không đổi."""
        result = simulate_inventory(
            actual_demand=np.zeros(10),
            rop=5.0, order_qty=10.0, lead_time=2, initial_inventory=20.0,
        )
        assert (result["units_short"] == 0.0).all()
        assert (result["inventory_end"] >= 0.0).all()

    def test_inventory_non_negative_lost_sales(self):
        """Mô hình lost-sales: tồn kho không bao giờ âm."""
        demand = np.full(28, 10.0)  # nhu cầu lớn
        result = simulate_inventory(
            actual_demand=demand,
            rop=0.0, order_qty=5.0, lead_time=7, initial_inventory=3.0,
        )
        assert (result["inventory_end"] >= 0.0).all(), "Tồn kho phải >= 0 với lost-sales model"

    def test_mass_balance(self):
        """Bảo toàn vật chất: initial + received = sold + short + end_inventory."""
        demand = np.array([3.0, 2.0, 5.0, 1.0, 4.0, 2.0, 3.0])
        result = simulate_inventory(
            actual_demand=demand,
            rop=5.0, order_qty=20.0, lead_time=2, initial_inventory=10.0,
        )
        # Tổng vào = tồn kho đầu + tổng nhận hàng
        total_in = 10.0 + np.sum(result["orders_received"])
        # Tổng ra = tổng bán + tồn kho cuối
        total_out = np.sum(result["units_sold"]) + result["inventory_end"][-1]
        assert total_in == pytest.approx(total_out, abs=1e-6), (
            f"Mất cân bằng vật chất: in={total_in:.4f}, out={total_out:.4f}"
        )

    def test_fill_rate_zero_when_no_inventory(self):
        """Tồn kho ban đầu = 0, không đặt hàng → không bán được gì."""
        result = simulate_inventory(
            actual_demand=np.ones(5) * 5.0,
            rop=-1.0,  # không bao giờ đặt hàng (ROP rất thấp)
            order_qty=0.0, lead_time=3, initial_inventory=0.0,
        )
        assert np.sum(result["units_sold"]) == pytest.approx(0.0)

    def test_order_arrives_after_lead_time(self):
        """Đơn hàng đặt ngày t phải đến đúng ngày t + lead_time."""
        result = simulate_inventory(
            actual_demand=np.zeros(10),
            rop=100.0,  # luôn đặt hàng ngay từ đầu
            order_qty=50.0, lead_time=3, initial_inventory=0.0,
        )
        # Nhận hàng phải xảy ra vào ngày 3 (t=3, t=0 đặt)
        assert result["orders_received"][3] > 0.0


class TestOraclePolicy:
    def test_paths_shape_and_non_negative(self):
        demand = np.array([2.0, 1.0, 3.0, 0.0, 2.0, 1.0, 2.0] * 4)
        rop, qty = build_oracle_paths(demand, lead_time_days=7, reorder_quantity_days=14)
        assert len(rop) == len(demand) == len(qty)
        assert (rop >= 0).all()
        assert (qty >= 1.0).all()

    def test_rop_covers_next_lead_time_demand(self):
        demand = np.arange(1.0, 29.0)
        rop, _ = build_oracle_paths(demand, lead_time_days=7, reorder_quantity_days=14)
        assert rop[0] == pytest.approx(demand[1:8].sum())
        assert rop[5] == pytest.approx(demand[6:13].sum())

    def test_oracle_is_lower_bound_than_baseline(self):
        # Nhu cầu biến động mạnh để baseline (ROP cố định) bị thiếu hàng.
        demand = np.array([8, 0, 1, 9, 0, 2, 8, 1, 0, 9, 2, 0, 8, 1, 0, 9,
                           2, 0, 8, 1, 0, 9, 2, 0, 8, 1, 0, 9], dtype=float)
        init_inv = calc_initial_inventory(np.full(28, 3.0), CFG)
        baseline = run_simulation_for_item(
            item_id="TEST", fold=0, actual_demand=demand,
            rop=10.0, order_qty=20.0, lead_time=7, initial_inventory=init_inv,
            holding_cost_per_unit_per_day=0.002032,
            ordering_cost=5.0, shortage_cost_per_unit=2.0,
        )
        rop_path, qty_path = build_oracle_paths(demand, 7, 14)
        oracle = run_simulation_for_item(
            item_id="TEST", fold=0, actual_demand=demand,
            rop=0.0, order_qty=0.0, lead_time=7, initial_inventory=init_inv,
            holding_cost_per_unit_per_day=0.002032,
            ordering_cost=5.0, shortage_cost_per_unit=2.0,
            dynamic_rop=rop_path, dynamic_order_qty=qty_path,
        )
        assert oracle["total_short"] <= baseline["total_short"]
        assert oracle["cost_total"] <= baseline["cost_total"]

    def test_dynamic_path_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            simulate_inventory(
                actual_demand=np.ones(10), rop=1.0, order_qty=1.0,
                lead_time=2, initial_inventory=5.0,
                dynamic_rop=np.ones(9),
            )


class TestRunSimulationForItem:
    def test_output_keys(self):
        demand = np.array([2.0, 1.0, 3.0, 0.0, 2.0, 1.0, 2.0] * 4)
        result = run_simulation_for_item(
            item_id="TEST_ITEM",
            fold=0,
            actual_demand=demand,
            rop=5.0, order_qty=20.0, lead_time=2, initial_inventory=10.0,
            holding_cost_per_unit_per_day=0.05,
            ordering_cost=5.0,
            shortage_cost_per_unit=2.0,
        )
        for key in ["cost_total", "fill_rate", "stockout_rate", "days_stockout",
                    "num_orders"]:
            assert key in result, f"Thiếu key '{key}' trong kết quả mô phỏng"

    def test_stockout_counts_only_days_with_unmet_demand(self):
        # Ngày 0 bán hết 5/5 (không thiếu), các ngày sau tồn kho = 0 nhưng
        # nhu cầu = 0 → KHÔNG được tính là stockout.
        result = run_simulation_for_item(
            item_id="TEST",
            fold=0,
            actual_demand=np.array([5.0] + [0.0] * 9),
            rop=-1.0, order_qty=0.0, lead_time=3, initial_inventory=5.0,
            holding_cost_per_unit_per_day=0.05,
            ordering_cost=5.0,
            shortage_cost_per_unit=2.0,
        )
        assert result["days_stockout"] == 0
        assert result["stockout_rate"] == pytest.approx(0.0)

    def test_stockout_counts_days_with_shortage(self):
        result = run_simulation_for_item(
            item_id="TEST",
            fold=0,
            actual_demand=np.array([5.0, 0.0, 0.0, 0.0]),
            rop=-1.0, order_qty=0.0, lead_time=3, initial_inventory=2.0,
            holding_cost_per_unit_per_day=0.05,
            ordering_cost=5.0,
            shortage_cost_per_unit=2.0,
        )
        assert result["days_stockout"] == 1
        assert result["stockout_rate"] == pytest.approx(0.25)

    def test_fill_rate_in_range(self):
        demand = np.ones(28) * 2.0
        result = run_simulation_for_item(
            item_id="TEST",
            fold=0,
            actual_demand=demand,
            rop=10.0, order_qty=30.0, lead_time=3, initial_inventory=15.0,
            holding_cost_per_unit_per_day=0.05,
            ordering_cost=5.0,
            shortage_cost_per_unit=2.0,
        )
        assert 0.0 <= result["fill_rate"] <= 1.0
        assert 0.0 <= result["stockout_rate"] <= 1.0
        assert result["cost_total"] >= 0.0


# =========================================================
# (d) Tests cho tổng hợp kết quả (run_simulation._summarize_results)
# =========================================================


def _summary_input() -> pd.DataFrame:
    """Dữ liệu giả lập tối thiểu cho 1 baseline và 1 mô hình."""
    base = {
        "scenario": "normal",
        "cost_holding": 10.0,
        "cost_ordering": 5.0,
        "cost_shortage": 2.0,
        "cost_total": 17.0,
        "stockout_rate": 0.1,
        "num_orders": 1,
        "zero_tier": "smooth",
    }
    rows = [
        dict(base, model="Baseline", policy="baseline", item_id="A", fold=0,
             fill_rate=1.0, total_demand=100.0, total_sold=90.0, total_short=10.0),
        dict(base, model="Baseline", policy="baseline", item_id="B", fold=0,
             fill_rate=0.5, total_demand=10.0, total_sold=10.0, total_short=0.0),
        dict(base, model="LightGBM", policy="point_forecast", item_id="A", fold=0,
             fill_rate=1.0, total_demand=100.0, total_sold=80.0, total_short=20.0),
        dict(base, model="LightGBM", policy="point_forecast", item_id="B", fold=0,
             fill_rate=0.2, total_demand=10.0, total_sold=8.0, total_short=2.0),
    ]
    return pd.DataFrame(rows)


class TestSummarizeResults:
    def test_fill_rate_pooled_identity_every_row(self):
        summary = _summarize_results(_summary_input())
        expected = 1.0 - summary["total_short"] / summary["total_demand"]
        assert np.allclose(summary["fill_rate_pooled"], expected, atol=1e-4)

    def test_fill_rate_pooled_uses_unit_weights(self):
        summary = _summarize_results(_summary_input())
        baseline = summary[summary["model"] == "Baseline"].iloc[0]
        # (90 + 10) / (100 + 10) = 0.9091, không phải mean(1.0, 0.5) = 0.75
        assert baseline["fill_rate_pooled"] == pytest.approx(100 / 110, abs=1e-4)
        assert baseline["fill_rate_mean_per_sku_fold"] == pytest.approx(0.75, abs=1e-4)
        assert "fill_rate" not in summary.columns

    def test_summary_keeps_baseline_and_model(self):
        summary = _summarize_results(_summary_input())
        assert set(summary["model"]) == {"Baseline", "LightGBM"}
        assert summary["model"].duplicated().sum() == 0


# =========================================================
# (e) Tests cho ma trận độ nhạy (sensitivity.find_inversions)
# =========================================================


class TestSensitivityInversions:
    @staticmethod
    def _full_df() -> pd.DataFrame:
        # Cấu hình cơ sở: A(1) < B(2) < C(3).
        # Cấu hình K=10: B(1) < A(2) < C(3) → A và B đảo thứ hạng.
        rows = []
        for model, policy, base_rank in [
            ("A", "baseline", 1),
            ("B", "baseline", 2),
            ("C", "baseline", 3),
        ]:
            rows.append(dict(parameter="ordering_cost", value=5.0,
                             is_base_config=True, model=model, policy=policy,
                             rank_cost=base_rank))
        new_ranks = {"A": 2, "B": 1, "C": 3}
        for model, policy, _ in [
            ("A", "baseline", 0), ("B", "baseline", 0), ("C", "baseline", 0),
        ]:
            rows.append(dict(parameter="ordering_cost", value=10.0,
                             is_base_config=False, model=model, policy=policy,
                             rank_cost=new_ranks[model]))
        return pd.DataFrame(rows)

    def test_detects_only_flipped_pair(self):
        from src.inventory.sensitivity import find_inversions

        inv = find_inversions(self._full_df())
        assert len(inv) == 1
        pair = {inv.iloc[0]["policy_a"], inv.iloc[0]["policy_b"]}
        assert pair == {"A | baseline", "B | baseline"}

    def test_base_config_never_reports_inversion(self):
        from src.inventory.sensitivity import find_inversions

        df = self._full_df()
        df = df[df["is_base_config"]]
        assert find_inversions(df).empty
