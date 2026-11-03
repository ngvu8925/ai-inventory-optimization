"""Pipeline chạy mô phỏng tồn kho cho tất cả SKU, fold, và chính sách — Phase 5.

Đọc dự báo từ results/predictions/, áp dụng 3 chính sách tồn kho,
mô phỏng theo ngày, và xuất bảng kết quả vào results/inventory/.

Chạy: python src/inventory/run_simulation.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_processed_daily
from src.forecasting.backtest import load_backtest_config, get_sku_zero_tiers
from src.forecasting.features import _ensure_d_num
from src.inventory.policies import load_assumptions, compute_policy_params
from src.inventory.simulator import run_simulation_for_item


# Danh sách mô hình → file dự báo tương ứng
# Thêm model mới vào đây khi cần, không sửa code khác.
MODEL_FILES = {
    "MovingAverage":          "results/predictions/movingaverage_predictions.parquet",
    "LightGBM":               "results/predictions/lightgbm_predictions.parquet",
    "Chronos2_History":       "results/predictions/chronos2_history_predictions.parquet",
    "Chronos2_Covariates":    "results/predictions/chronos2_covariates_predictions.parquet",
    "Chronos_Bolt":           "results/predictions/chronos_bolt_predictions.parquet",
}

# Danh sách chính sách áp dụng cho từng mô hình
# quantile_forecast chỉ áp dụng cho mô hình có cột q_0.9
POLICIES_POINT = ["baseline", "point_forecast"]
POLICIES_QUANTILE = ["baseline", "point_forecast", "quantile_forecast"]


def load_forecast(model_name: str, pred_path: str) -> pd.DataFrame:
    """Đọc file dự báo Parquet hoặc CSV."""
    path = Path(pred_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dự báo: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def run_all_simulations(
    pred_dir: str = "results/predictions",
    out_dir: str = "results/inventory",
    cfg_path: str = "configs/assumptions.yaml",
    backtest_cfg_path: str = "configs/backtest.yaml",
    daily_data_path: str = "data/processed/sales_daily.parquet",
    stress_scenario: str = "none",
) -> pd.DataFrame:
    """Chạy mô phỏng đầy đủ cho tất cả mô hình × chính sách × SKU × fold.

    Args:
        stress_scenario: 'none', 'demand_spike', hoặc 'lead_time_extended'.

    Returns:
        DataFrame tóm tắt kết quả toàn bộ.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_assumptions(cfg_path)
    backtest_cfg = load_backtest_config(backtest_cfg_path)

    horizon = int(backtest_cfg["horizon"])
    num_folds = int(backtest_cfg["num_folds"])
    initial_origin = int(backtest_cfg["initial_origin"])
    step_size = int(backtest_cfg["step_size"])

    # Áp dụng stress nếu cần
    lead_time = int(cfg["lead_time_days"])
    if stress_scenario == "lead_time_extended":
        lead_time = int(cfg["stress"]["lead_time_days_extended"])
        print(f"[STRESS] Lead time kéo dài: {lead_time} ngày")
    demand_spike = 1.0
    if stress_scenario == "demand_spike":
        demand_spike = float(cfg["stress"]["demand_spike_factor"])
        print(f"[STRESS] Nhu cầu nhân hệ số: {demand_spike}")

    # Tải dữ liệu lịch sử
    daily_df = load_processed_daily(daily_data_path)
    daily_df = _ensure_d_num(daily_df)
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial_origin)
    items = sorted(daily_df["item_id"].unique())

    all_records = []
    scenario_label = stress_scenario if stress_scenario != "none" else "normal"

    for model_name, pred_path in MODEL_FILES.items():
        try:
            forecast_df = load_forecast(model_name, pred_path)
        except FileNotFoundError:
            print(f"  [SKIP] {model_name}: file dự báo không tồn tại.")
            continue

        has_quantile = "q_0.9" in forecast_df.columns
        policies = POLICIES_QUANTILE if has_quantile else POLICIES_POINT

        print(f"\n--- Mô hình: {model_name} | Chính sách: {policies} ---")

        for policy_name in policies:
            for fold_k in range(num_folds):
                origin_day = initial_origin + fold_k * step_size

                # Lịch sử đến origin (toàn bộ, không cắt)
                hist_df = daily_df[daily_df["d_num"] <= origin_day].copy()

                # Dự báo của fold này
                fold_fc = forecast_df[forecast_df["fold"] == fold_k].copy()
                if len(fold_fc) == 0:
                    print(f"    [SKIP] fold {fold_k}: không có dự báo.")
                    continue

                for item_id in items:
                    hist_item = hist_df[hist_df["item_id"] == item_id]["sales"].values.astype(float)
                    if len(hist_item) == 0:
                        continue

                    try:
                        params = compute_policy_params(
                            item_id=item_id,
                            fold=fold_k,
                            history_df=hist_df,
                            forecast_df=fold_fc,
                            policy_name=policy_name,
                            cfg=cfg,
                        )
                    except (ValueError, KeyError) as e:
                        print(f"    [WARN] {model_name}/{policy_name}/fold{fold_k}/{item_id}: {e}")
                        continue

                    # Nhu cầu thực trong fold (áp dụng stress nếu có)
                    actual_fold = (
                        daily_df[
                            (daily_df["d_num"] > origin_day) &
                            (daily_df["d_num"] <= origin_day + horizon) &
                            (daily_df["item_id"] == item_id)
                        ].sort_values("d_num")["sales"].values.astype(float)
                    )
                    if len(actual_fold) == 0:
                        continue
                    actual_fold = actual_fold * demand_spike  # stress hoặc 1.0

                    # Tồn kho ban đầu = d_bar * initial_inventory_days
                    d_bar = params["d_bar"]
                    init_inv = d_bar * float(cfg["initial_inventory_days"])
                    init_inv = max(init_inv, 0.0)

                    sim = run_simulation_for_item(
                        item_id=item_id,
                        fold=fold_k,
                        actual_demand=actual_fold,
                        rop=params["rop"],
                        order_qty=params["order_qty"],
                        lead_time=lead_time,
                        initial_inventory=init_inv,
                        holding_cost_per_unit_per_day=float(cfg["holding_cost_per_unit_per_day"]),
                        ordering_cost=float(cfg["ordering_cost"]),
                        shortage_cost_per_unit=float(cfg["shortage_cost_per_unit"]),
                    )

                    sim["model"] = model_name
                    sim["policy"] = policy_name
                    sim["scenario"] = scenario_label
                    sim["zero_tier"] = zero_tiers.get(item_id, "unknown")
                    all_records.append(sim)

    results_df = pd.DataFrame(all_records)
    if len(results_df) == 0:
        print("CẢNH BÁO: Không có kết quả nào được tạo ra!")
        return results_df

    # Lưu kết quả chi tiết theo từng SKU và fold
    detail_path = Path(out_dir) / f"simulation_detail_{scenario_label}.csv"
    results_df.to_csv(detail_path, index=False)
    print(f"\nĐã lưu {len(results_df):,} dòng chi tiết → {detail_path}")

    # Tóm tắt theo mô hình × chính sách
    summary = _summarize_results(results_df)
    summary_path = Path(out_dir) / f"simulation_summary_{scenario_label}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Đã lưu bảng tóm tắt → {summary_path}")

    # Tóm tắt theo zero_tier
    tier_summary = _summarize_by_tier(results_df)
    tier_path = Path(out_dir) / f"simulation_by_zero_tier_{scenario_label}.csv"
    tier_summary.to_csv(tier_path, index=False)
    print(f"Đã lưu bảng theo zero_tier → {tier_path}")

    return results_df


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Tóm tắt tổng hợp theo mô hình × chính sách × kịch bản."""
    agg = (
        df.groupby(["model", "policy", "scenario"])
        .agg(
            cost_holding=("cost_holding", "sum"),
            cost_ordering=("cost_ordering", "sum"),
            cost_shortage=("cost_shortage", "sum"),
            cost_total=("cost_total", "sum"),
            fill_rate=("fill_rate", "mean"),
            stockout_rate=("stockout_rate", "mean"),
            num_orders=("num_orders", "sum"),
            total_demand=("total_demand", "sum"),
            total_short=("total_short", "sum"),
            count=("item_id", "count"),
        )
        .reset_index()
    )
    # Làm tròn
    for col in ["cost_holding", "cost_ordering", "cost_shortage", "cost_total", "total_demand", "total_short"]:
        agg[col] = agg[col].round(2)
    agg["fill_rate"] = agg["fill_rate"].round(4)
    agg["stockout_rate"] = agg["stockout_rate"].round(4)
    return agg.sort_values(["scenario", "policy", "cost_total"])


def _summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Tóm tắt theo mô hình × chính sách × zero_tier."""
    agg = (
        df.groupby(["model", "policy", "scenario", "zero_tier"])
        .agg(
            cost_total=("cost_total", "sum"),
            fill_rate=("fill_rate", "mean"),
            stockout_rate=("stockout_rate", "mean"),
            count=("item_id", "count"),
        )
        .reset_index()
    )
    agg["cost_total"] = agg["cost_total"].round(2)
    agg["fill_rate"] = agg["fill_rate"].round(4)
    agg["stockout_rate"] = agg["stockout_rate"].round(4)
    return agg.sort_values(["scenario", "zero_tier", "policy", "cost_total"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chạy mô phỏng tồn kho Phase 5")
    parser.add_argument(
        "--stress",
        choices=["none", "demand_spike", "lead_time_extended"],
        default="none",
        help="Kịch bản stress-test (mặc định: none = kịch bản bình thường)",
    )
    args = parser.parse_args()

    print(f"=== BẮT ĐẦU MÔ PHỎNG TỒN KHO | Kịch bản: {args.stress} ===")
    run_all_simulations(stress_scenario=args.stress)
    print("=== HOÀN TẤT ===")
