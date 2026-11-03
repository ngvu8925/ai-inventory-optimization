"""Mô phỏng tồn kho theo ngày (Continuous Review, Lost Sales) — Phase 5.

Mô hình:
  - Review liên tục (Continuous Review / (s, Q) policy).
  - Khi tồn kho khả dụng <= ROP → đặt order_qty đơn vị, giao sau L ngày.
  - Mô hình Lost Sales: nhu cầu vượt quá tồn kho → mất doanh thu, không backorder.
  - Không cho phép tồn kho âm trong báo cáo.

Đầu vào:
  - actual_demand: chuỗi nhu cầu thực mỗi ngày (array of float).
  - rop: Điểm đặt hàng lại (Reorder Point).
  - order_qty: Lượng đặt mỗi lần.
  - lead_time: Lead time cố định (ngày).
  - initial_inventory: Tồn kho ban đầu trước ngày đầu tiên mô phỏng.

Đầu ra (per ngày):
  - inventory_end: Tồn kho cuối ngày (on-hand, >= 0 với lost-sales model).
  - units_sold: Số đơn vị bán được.
  - units_short: Số đơn vị thiếu (lost sales).
  - orders_placed: Số lượng đặt trong ngày (0 hoặc order_qty).
  - orders_received: Số lượng nhận trong ngày.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def simulate_inventory(
    actual_demand: np.ndarray,
    rop: float,
    order_qty: float,
    lead_time: int,
    initial_inventory: float,
) -> Dict[str, np.ndarray]:
    """Mô phỏng tồn kho theo ngày với chính sách (s=ROP, Q=order_qty) và lost-sales.

    Thứ tự trong ngày:
      1. Nhận hàng (orders đến hạn trong ngày t).
      2. Bán hàng: units_sold = min(inventory, demand).
      3. Tính thiếu hàng: units_short = max(demand - inventory, 0).
      4. Cập nhật tồn kho cuối ngày.
      5. Kiểm tra ROP: nếu inventory_end <= ROP và chưa có đơn đang chờ → đặt.

    Args:
        actual_demand: Chuỗi nhu cầu thực hàng ngày (độ dài T).
        rop: Điểm đặt hàng lại.
        order_qty: Lượng đặt mỗi lần (cố định trong mô phỏng này).
        lead_time: Lead time cố định (ngày).
        initial_inventory: Tồn kho ban đầu (on-hand ngay trước ngày 0).

    Returns:
        Dict với các mảng độ dài T:
          - inventory_end: Tồn kho on-hand cuối ngày (>= 0).
          - units_sold: Đơn vị bán được.
          - units_short: Đơn vị thiếu hàng.
          - orders_placed: Lượng đặt trong ngày (0 hoặc order_qty).
          - orders_received: Lượng nhận trong ngày.
          - inventory_available: Tồn kho đầu ngày sau khi nhận hàng.
    """
    T = len(actual_demand)
    demand = np.asarray(actual_demand, dtype=float)

    inventory_end = np.zeros(T)
    units_sold = np.zeros(T)
    units_short = np.zeros(T)
    orders_placed = np.zeros(T)
    orders_received = np.zeros(T)
    inventory_available = np.zeros(T)

    # Dòng đặt hàng: (arrival_day, quantity)
    pipeline: List[Tuple[int, float]] = []
    on_hand = float(initial_inventory)
    on_order = 0.0  # tổng hàng đang trên đường

    for t in range(T):
        # 1. Nhận hàng đến hạn ngày t
        received = 0.0
        new_pipeline = []
        for (arr_day, qty) in pipeline:
            if arr_day == t:
                received += qty
            else:
                new_pipeline.append((arr_day, qty))
        pipeline = new_pipeline
        on_hand += received
        on_order -= received
        orders_received[t] = received
        inventory_available[t] = on_hand  # tồn kho sau khi nhận, trước khi bán

        # 2. Bán hàng (lost-sales model)
        sold = min(on_hand, demand[t])
        short = max(demand[t] - on_hand, 0.0)
        on_hand -= sold
        units_sold[t] = sold
        units_short[t] = short

        # 3. Tồn kho cuối ngày (>= 0 với lost-sales)
        inventory_end[t] = on_hand

        # 4. Kiểm tra đặt hàng: đặt khi tồn kho khả dụng <= ROP và không có đơn chờ
        # Tồn kho khả dụng = on_hand + on_order (inventory position)
        inventory_position = on_hand + on_order
        if inventory_position <= rop and on_order == 0.0:
            qty_order = max(order_qty, 1.0)
            arrival_day = t + lead_time
            pipeline.append((arrival_day, qty_order))
            on_order += qty_order
            orders_placed[t] = qty_order

    return {
        "inventory_end": inventory_end,
        "units_sold": units_sold,
        "units_short": units_short,
        "orders_placed": orders_placed,
        "orders_received": orders_received,
        "inventory_available": inventory_available,
    }


def run_simulation_for_item(
    item_id: str,
    fold: int,
    actual_demand: np.ndarray,
    rop: float,
    order_qty: float,
    lead_time: int,
    initial_inventory: float,
    holding_cost_per_unit_per_day: float,
    ordering_cost: float,
    shortage_cost_per_unit: float,
) -> Dict[str, float]:
    """Chạy mô phỏng cho 1 SKU trong 1 fold và tính toán tất cả chỉ số.

    Returns:
        Dict tóm tắt kết quả: chi phí, fill_rate, stockout_rate, tổng số đặt hàng, v.v.
    """
    from src.inventory.costs import (
        calc_holding_cost, calc_ordering_cost,
        calc_shortage_cost, calc_fill_rate, calc_stockout_rate,
    )

    result = simulate_inventory(
        actual_demand=actual_demand,
        rop=rop,
        order_qty=order_qty,
        lead_time=lead_time,
        initial_inventory=initial_inventory,
    )

    T = len(actual_demand)
    num_orders = int(np.sum(result["orders_placed"] > 0))

    holding = calc_holding_cost(result["inventory_end"], holding_cost_per_unit_per_day)
    ordering = calc_ordering_cost(num_orders, ordering_cost)
    shortage = calc_shortage_cost(result["units_short"], shortage_cost_per_unit)
    total_cost = holding + ordering + shortage

    fill_rate = calc_fill_rate(result["units_sold"], actual_demand)
    days_stockout = int(np.sum(result["inventory_end"] == 0.0))
    stockout_rate = calc_stockout_rate(days_stockout, T)

    return {
        "item_id": item_id,
        "fold": fold,
        "cost_holding": round(holding, 4),
        "cost_ordering": round(ordering, 4),
        "cost_shortage": round(shortage, 4),
        "cost_total": round(total_cost, 4),
        "fill_rate": round(fill_rate, 4),
        "stockout_rate": round(stockout_rate, 4),
        "num_orders": num_orders,
        "total_demand": round(float(np.sum(actual_demand)), 2),
        "total_sold": round(float(np.sum(result["units_sold"])), 2),
        "total_short": round(float(np.sum(result["units_short"])), 2),
        "avg_inventory": round(float(np.mean(result["inventory_end"])), 4),
        "rop_used": round(rop, 4),
        "order_qty_used": round(order_qty, 4),
        "initial_inventory": round(initial_inventory, 4),
    }
