"""Unit tests cho mô-đun tồn kho Phase 5 — chạy trong .venv chính.

Kiểm tra:
  (a) costs.py: holding, ordering, shortage, fill_rate, stockout_rate.
  (b) policies.py: ROP, SS, ba chính sách.
  (c) simulator.py: dòng tồn kho đúng, không âm, fill rate hợp lệ, tồn kho bảo toàn.
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
    calc_rop,
    calc_safety_stock_normal,
    policy_baseline,
    policy_point_forecast,
    policy_quantile_forecast,
)
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
    "z_score": 1.645,
    "reorder_quantity_days": 14,
    "holding_cost_per_unit_per_day": 0.05,
    "ordering_cost": 5.0,
    "shortage_cost_per_unit": 2.0,
    "service_level": 0.95,
    "initial_inventory_days": 7,
}


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
    def test_returns_valid_keys(self):
        fc = np.ones(28) * 3.0
        q90 = np.ones(28) * 5.0
        result = policy_quantile_forecast(fc, q90, CFG)
        assert "rop" in result
        assert result["rop"] >= 0.0
        assert result["safety_stock"] >= 0.0


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
        for key in ["cost_total", "fill_rate", "stockout_rate", "num_orders"]:
            assert key in result, f"Thiếu key '{key}' trong kết quả mô phỏng"

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
