"""Hàm tính toán chi phí tồn kho (Phase 5).

Gồm 3 thành phần:
  - Chi phí lưu kho (holding cost): h * sum(max(I_t, 0))
  - Chi phí đặt hàng (ordering cost): K * số lần đặt hàng
  - Chi phí thiếu hàng (shortage cost / lost sales): p * sum(max(-I_t, 0))

Tất cả tham số (h, K, p) đọc từ config, KHÔNG hard-code.
"""

from typing import Union
import numpy as np


def calc_holding_cost(
    inventory_levels: Union[np.ndarray, list],
    holding_cost_per_unit_per_day: float,
) -> float:
    """Tính tổng chi phí lưu kho trong toàn bộ chu kỳ mô phỏng.

    Args:
        inventory_levels: Mảng tồn kho cuối ngày (on-hand), có thể âm nếu thiếu hàng.
        holding_cost_per_unit_per_day: Chi phí lưu kho mỗi đơn vị mỗi ngày (h).

    Returns:
        Tổng chi phí lưu kho (chỉ tính phần dương).
    """
    inv = np.asarray(inventory_levels, dtype=float)
    return float(holding_cost_per_unit_per_day * np.sum(np.maximum(inv, 0.0)))


def calc_ordering_cost(
    num_orders: int,
    ordering_cost: float,
) -> float:
    """Tính tổng chi phí đặt hàng.

    Args:
        num_orders: Số lần đặt hàng trong chu kỳ mô phỏng.
        ordering_cost: Chi phí cố định mỗi lần đặt hàng (K).

    Returns:
        Tổng chi phí đặt hàng.
    """
    if num_orders < 0:
        raise ValueError(f"num_orders phải >= 0, nhận: {num_orders}")
    return float(num_orders * ordering_cost)


def calc_shortage_cost(
    units_short: Union[np.ndarray, list],
    shortage_cost_per_unit: float,
) -> float:
    """Tính tổng chi phí thiếu hàng (lost sales model).

    Args:
        units_short: Mảng số đơn vị thiếu hàng mỗi ngày (>= 0).
        shortage_cost_per_unit: Chi phí mỗi đơn vị thiếu hàng (p).

    Returns:
        Tổng chi phí thiếu hàng.
    """
    us = np.asarray(units_short, dtype=float)
    if np.any(us < 0):
        raise ValueError("units_short không được âm — dùng max(0, shortage) trước khi truyền vào.")
    return float(shortage_cost_per_unit * np.sum(us))


def calc_total_cost(
    inventory_levels: Union[np.ndarray, list],
    units_short: Union[np.ndarray, list],
    num_orders: int,
    holding_cost_per_unit_per_day: float,
    ordering_cost: float,
    shortage_cost_per_unit: float,
) -> dict:
    """Tính tổng chi phí tồn kho và từng thành phần.

    Returns:
        dict với các key: holding, ordering, shortage, total.
    """
    h = calc_holding_cost(inventory_levels, holding_cost_per_unit_per_day)
    k = calc_ordering_cost(num_orders, ordering_cost)
    p = calc_shortage_cost(units_short, shortage_cost_per_unit)
    return {
        "holding": round(h, 4),
        "ordering": round(k, 4),
        "shortage": round(p, 4),
        "total": round(h + k + p, 4),
    }


def calc_fill_rate(
    units_sold: Union[np.ndarray, list],
    units_demanded: Union[np.ndarray, list],
) -> float:
    """Tính Fill Rate = tổng nhu cầu đáp ứng được / tổng nhu cầu.

    Fill Rate = 1.0 nghĩa là đáp ứng 100% nhu cầu.
    Trả về 1.0 nếu tổng nhu cầu = 0.

    Args:
        units_sold: Số đơn vị bán được (thực tế đáp ứng) mỗi ngày.
        units_demanded: Số đơn vị nhu cầu thực mỗi ngày.

    Returns:
        Fill rate trong khoảng [0.0, 1.0].
    """
    sold = np.asarray(units_sold, dtype=float)
    demanded = np.asarray(units_demanded, dtype=float)
    total_demand = float(np.sum(demanded))
    if total_demand == 0.0:
        return 1.0
    return float(min(np.sum(sold) / total_demand, 1.0))


def calc_stockout_rate(
    days_with_stockout: int,
    total_days: int,
) -> float:
    """Tính tỷ lệ ngày thiếu hàng = số ngày có thiếu hàng / tổng số ngày.

    Args:
        days_with_stockout: Số ngày tồn kho về 0 hoặc âm trước khi bán.
        total_days: Tổng số ngày trong chu kỳ mô phỏng.

    Returns:
        Tỷ lệ thiếu hàng trong khoảng [0.0, 1.0].
    """
    if total_days <= 0:
        raise ValueError("total_days phải > 0")
    return float(days_with_stockout / total_days)
