"""Pipeline chạy mô phỏng tồn kho cho tất cả SKU, fold, và chính sách — Phase 5.

Đọc dự báo từ results/predictions/, áp dụng các chính sách tồn kho,
mô phỏng theo ngày, và xuất bảng kết quả vào results/inventory/.

Cấu trúc nguồn chạy:
  - Baseline / Baseline_full_history: chạy MỘT LẦN, không phụ thuộc mô hình.
  - Mỗi mô hình dự báo (LightGBM, MovingAverage, 3 biến thể Chronos):
      * 3 biến thể point_forecast (tách nguồn sigma):
          - point_forecast                    : σ toàn bộ lịch sử
          - point_forecast_window_sigma       : σ cửa sổ Baseline (91 ngày)
          - point_forecast_error_sigma        : σ từ sai số các fold trước origin
      * 2 biến thể quantile (nếu mô hình thực sự có q0.1/q0.9/q0.95):
          - quantile_forecast                 : σ từ khoảng q0.1–q0.9
          - quantile_forecast_history_sigma   : σ toàn bộ lịch sử
    Chronos-Bolt bị đánh dấu "không khả dụng" cho chính sách quantile vì
    q0.95 của nó chỉ là bản sao của q0.9, không phải phân vị mô hình xuất ra.

Chạy: python src/inventory/run_simulation.py [--stress none|demand_spike|lead_time_extended]
"""

import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data.loader import load_processed_daily
from src.forecasting.backtest import load_backtest_config, get_sku_zero_tiers
from src.forecasting.features import _ensure_d_num
from src.inventory.policies import (
    build_oracle_paths,
    calc_initial_inventory,
    compute_policy_params,
    load_assumptions,
)
from src.inventory.prediction_prep import prepare_model_predictions
from src.inventory.simulator import run_simulation_for_item


# Danh sách mô hình → file dự báo tương ứng.
# Baseline KHÔNG nằm ở đây vì nó độc lập với mô hình (xem BASELINE_SOURCES).
MODEL_FILES = {
    "MovingAverage":          "results/predictions/movingaverage_predictions.parquet",
    "LightGBM":               "results/predictions/lightgbm_predictions.parquet",
    "Chronos2_History":       "results/predictions/chronos2_history_predictions.parquet",
    "Chronos2_Covariates":    "results/predictions/chronos2_covariates_predictions.parquet",
    "Chronos_Bolt":           "results/predictions/chronos_bolt_predictions.parquet",
}

# Baseline chỉ xuất hiện MỘT LẦN cho mỗi biến thể, không lặp theo mô hình.
BASELINE_SOURCES = [
    ("Baseline", "baseline"),
    ("Baseline_full_history", "baseline_full_history"),
]

# Oracle: tham chiếu cận dưới chi phí, KHÔNG phải mô hình dự báo.
# Tên hiển thị ghi rõ để không bị đọc nhầm là một mô hình.
ORACLE_SOURCE = ("Oracle (lower bound, not a model)", "oracle_reference")

# 3 biến thể point: tách phần do ROP (vị trí) khỏi phần do nguồn sigma.
POLICIES_POINT = [
    "point_forecast",                 # σ toàn bộ lịch sử (chính sách cũ)
    "point_forecast_window_sigma",    # (a) σ cửa sổ Baseline 91 ngày
    "point_forecast_error_sigma",     # (b) σ sai số dự báo các fold trước origin
]
# 2 biến thể quantile; chỉ chạy khi mô hình thực sự xuất q0.1/q0.9/q0.95.
POLICIES_QUANTILE_EXTRA = [
    "quantile_forecast",                 # σ từ khoảng q0.1–q0.9 (chính sách cũ)
    "quantile_forecast_history_sigma",   # (c) σ toàn bộ lịch sử
]

# Nhãn mô tả nguồn sigma để ghi kèm kết quả, tránh đọc nhầm các biến thể.
POLICY_META = {
    "baseline": ("window_91", "Baseline: d̄, σ, Q đều từ cửa sổ 91 ngày"),
    "baseline_full_history": ("full_history", "Baseline dùng toàn bộ lịch sử"),
    "point_forecast": ("full_history", "Point forecast, σ toàn bộ lịch sử"),
    "point_forecast_window_sigma": ("window_91", "Point forecast, σ cửa sổ 91 ngày"),
    "point_forecast_error_sigma": ("forecast_error", "Point forecast, σ sai số các fold trước"),
    "quantile_forecast": ("q10_q90", "Quantile forecast, σ từ q0.1–q0.9"),
    "quantile_forecast_history_sigma": ("full_history", "Quantile forecast, σ toàn bộ lịch sử"),
    "oracle_reference": ("perfect_foresight", "Oracle: nhu cầu thực tương lai, không phải mô hình"),
}


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
    overrides: Optional[dict] = None,
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Chạy mô phỏng đầy đủ cho tất cả nguồn × chính sách × SKU × fold.

    Args:
        stress_scenario: 'none', 'demand_spike', hoặc 'lead_time_extended'.
        overrides: Dict ghi đè tham số trong assumptions.yaml (dùng cho ma
            trận độ nhạy). Chỉ ghi đè các key top-level như
            holding_cost_per_unit_per_day, shortage_cost_per_unit,
            ordering_cost, service_level.
        write_outputs: Nếu False, không ghi file CSV (dùng khi chạy ma trận).

    Returns:
        DataFrame tóm tắt kết quả toàn bộ.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg = load_assumptions(cfg_path)
    if overrides:
        cfg = {**cfg, **overrides}
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

    def _run_source(
        display_model: str,
        policy_name: str,
        forecast_df: pd.DataFrame | None,
        error_sigma_df: pd.DataFrame | None = None,
    ) -> None:
        """Chạy 1 nguồn (baseline hoặc model) cho toàn bộ fold × SKU."""
        print(f"\n--- Nguồn: {display_model} | Chính sách: {policy_name} ---")
        sigma_lookup = {}
        if error_sigma_df is not None and len(error_sigma_df) > 0:
            sigma_lookup = {
                (int(r["fold"]), str(r["item_id"])): float(r["sigma_error"])
                for _, r in error_sigma_df.iterrows()
            }

        for fold_k in range(num_folds):
            origin_day = initial_origin + fold_k * step_size

            # Lịch sử đến hết origin — dùng chung cho mọi chính sách
            hist_df = daily_df[daily_df["d_num"] <= origin_day].copy()

            fold_fc = None
            if forecast_df is not None:
                fold_fc = forecast_df[forecast_df["fold"] == fold_k].copy()
                if len(fold_fc) == 0:
                    print(f"    [SKIP] fold {fold_k}: không có dự báo.")
                    continue

            for item_id in items:
                hist_item = hist_df[hist_df["item_id"] == item_id]["sales"].values.astype(float)
                if len(hist_item) == 0:
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

                # Tồn kho ban đầu giống nhau cho MỌI chính sách:
                # trung bình nhu cầu THỰC trước origin × initial_inventory_days.
                init_inv = calc_initial_inventory(hist_item, cfg)

                if policy_name == "oracle_reference":
                    # Perfect foresight: dùng nhu cầu thực tương lai của fold.
                    # Chỉ là cận dưới chi phí, không phải mô hình dự báo.
                    rop_path, qty_path = build_oracle_paths(
                        actual_fold,
                        lead_time_days=lead_time,
                        reorder_quantity_days=int(cfg["reorder_quantity_days"]),
                    )
                    rop_value, qty_value = 0.0, 0.0
                    dynamic_rop, dynamic_qty = rop_path, qty_path
                else:
                    sigma_error = sigma_lookup.get((fold_k, item_id))
                    try:
                        params = compute_policy_params(
                            item_id=item_id,
                            fold=fold_k,
                            history_df=hist_df,
                            forecast_df=fold_fc,
                            policy_name=policy_name,
                            cfg=cfg,
                            sigma_error=sigma_error,
                        )
                    except (ValueError, KeyError) as e:
                        print(f"    [WARN] {display_model}/{policy_name}/fold{fold_k}/{item_id}: {e}")
                        continue
                    rop_value, qty_value = params["rop"], params["order_qty"]
                    dynamic_rop, dynamic_qty = None, None

                sim = run_simulation_for_item(
                    item_id=item_id,
                    fold=fold_k,
                    actual_demand=actual_fold,
                    rop=rop_value,
                    order_qty=qty_value,
                    lead_time=lead_time,
                    initial_inventory=init_inv,
                    holding_cost_per_unit_per_day=float(cfg["holding_cost_per_unit_per_day"]),
                    ordering_cost=float(cfg["ordering_cost"]),
                    shortage_cost_per_unit=float(cfg["shortage_cost_per_unit"]),
                    dynamic_rop=dynamic_rop,
                    dynamic_order_qty=dynamic_qty,
                )

                sim["model"] = display_model
                sim["policy"] = policy_name
                sim["scenario"] = scenario_label
                sim["zero_tier"] = zero_tiers.get(item_id, "unknown")
                all_records.append(sim)

    # 1) Baseline: chạy một lần, độc lập với mô hình dự báo
    for display_name, policy_name in BASELINE_SOURCES:
        _run_source(display_name, policy_name, forecast_df=None)

    # 2) Các mô hình dự báo
    availability_rows = []
    rearrangement_rows = []
    error_sigma_frames = []
    baseline_window = int(cfg.get("baseline_window_days", 91))

    for model_name, pred_path in MODEL_FILES.items():
        try:
            raw_forecast_df = load_forecast(model_name, pred_path)
        except FileNotFoundError:
            print(f"  [SKIP] {model_name}: file dự báo không tồn tại.")
            continue

        forecast_df, availability, error_sigma_df, rearr_stats = prepare_model_predictions(
            model_name=model_name,
            pred_df=raw_forecast_df,
            daily_df=daily_df,
            bt_cfg=backtest_cfg,
            baseline_window_days=baseline_window,
        )
        availability_rows.append(availability)
        rearrangement_rows.append({"model": model_name, **rearr_stats})
        error_sigma_frames.append(error_sigma_df)

        policies = list(POLICIES_POINT)
        if availability["quantile_policy_available"]:
            policies += POLICIES_QUANTILE_EXTRA
        else:
            print(
                f"  [KHÔNG KHẢ DỤNG] {model_name}: bỏ chính sách quantile — "
                f"{availability['reason']}."
            )

        for policy_name in policies:
            _run_source(model_name, policy_name, forecast_df, error_sigma_df)

    # 3) Oracle: tham chiếu cận dưới chi phí (perfect foresight), không phải mô hình
    _run_source(ORACLE_SOURCE[0], ORACLE_SOURCE[1], forecast_df=None)

    results_df = pd.DataFrame(all_records)
    if len(results_df) == 0:
        print("CẢNH BÁO: Không có kết quả nào được tạo ra!")
        return results_df

    summary = _summarize_results(results_df)
    if not write_outputs:
        return results_df

    # Metadata của bước chuẩn bị dự báo: khả dụng phân vị, sắp xếp lại,
    # sigma sai số, và định nghĩa nguồn sigma của từng chính sách.
    if availability_rows:
        pd.DataFrame(availability_rows).to_csv(
            Path(out_dir) / "quantile_availability.csv", index=False
        )
    if rearrangement_rows:
        pd.DataFrame(rearrangement_rows).to_csv(
            Path(out_dir) / "quantile_rearrangement.csv", index=False
        )
    if error_sigma_frames:
        pd.concat(error_sigma_frames, ignore_index=True).to_csv(
            Path(out_dir) / "forecast_error_sigma.csv", index=False
        )
    pd.DataFrame(
        [
            {"policy": policy, "sigma_source": meta[0], "description": meta[1]}
            for policy, meta in POLICY_META.items()
        ]
    ).to_csv(Path(out_dir) / "policy_metadata.csv", index=False)

    # Lưu kết quả chi tiết theo từng SKU và fold
    detail_path = Path(out_dir) / f"simulation_detail_{scenario_label}.csv"
    results_df.to_csv(detail_path, index=False)
    print(f"\nĐã lưu {len(results_df):,} dòng chi tiết → {detail_path}")

    # Tóm tắt theo nguồn × chính sách
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
    """Tóm tắt tổng hợp theo nguồn × chính sách × kịch bản.

    Chỉ số chính:
      - fill_rate_pooled = 1 - sum(total_short) / sum(total_demand)
        (đúng định nghĩa tổng đơn vị đáp ứng / tổng nhu cầu)
    Chỉ số phụ:
      - fill_rate_mean_per_sku_fold = trung bình cộng fill rate của từng
        SKU×fold (giữ lại để đối chiếu, KHÔNG dùng làm chỉ số chính).
    """
    agg = (
        df.groupby(["model", "policy", "scenario"])
        .agg(
            cost_holding=("cost_holding", "sum"),
            cost_ordering=("cost_ordering", "sum"),
            cost_shortage=("cost_shortage", "sum"),
            cost_total=("cost_total", "sum"),
            fill_rate_mean_per_sku_fold=("fill_rate", "mean"),
            stockout_rate=("stockout_rate", "mean"),
            num_orders=("num_orders", "sum"),
            total_demand=("total_demand", "sum"),
            total_sold=("total_sold", "sum"),
            total_short=("total_short", "sum"),
            count=("item_id", "count"),
        )
        .reset_index()
    )

    # Fill rate gộp: tổng đơn vị bán được / tổng nhu cầu
    demand = agg["total_demand"].to_numpy(dtype=float)
    short = agg["total_short"].to_numpy(dtype=float)
    agg["fill_rate_pooled"] = np.where(demand > 0, 1.0 - short / demand, 1.0)

    for col in ["cost_holding", "cost_ordering", "cost_shortage", "cost_total",
                "total_demand", "total_sold", "total_short"]:
        agg[col] = agg[col].round(2)
    agg["fill_rate_pooled"] = agg["fill_rate_pooled"].round(4)
    agg["fill_rate_mean_per_sku_fold"] = agg["fill_rate_mean_per_sku_fold"].round(4)
    agg["stockout_rate"] = agg["stockout_rate"].round(4)

    cols = [
        "model", "policy", "scenario",
        "fill_rate_pooled", "fill_rate_mean_per_sku_fold", "stockout_rate",
        "cost_holding", "cost_ordering", "cost_shortage", "cost_total",
        "num_orders", "total_demand", "total_sold", "total_short", "count",
    ]
    return agg[cols].sort_values(["scenario", "policy", "cost_total"])


def _summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Tóm tắt theo nguồn × chính sách × zero_tier."""
    agg = (
        df.groupby(["model", "policy", "scenario", "zero_tier"])
        .agg(
            cost_total=("cost_total", "sum"),
            fill_rate_mean_per_sku_fold=("fill_rate", "mean"),
            stockout_rate=("stockout_rate", "mean"),
            total_demand=("total_demand", "sum"),
            total_sold=("total_sold", "sum"),
            total_short=("total_short", "sum"),
            count=("item_id", "count"),
        )
        .reset_index()
    )
    demand = agg["total_demand"].to_numpy(dtype=float)
    short = agg["total_short"].to_numpy(dtype=float)
    agg["fill_rate_pooled"] = np.where(demand > 0, 1.0 - short / demand, 1.0)

    for col in ["cost_total", "total_demand", "total_sold", "total_short"]:
        agg[col] = agg[col].round(2)
    agg["fill_rate_pooled"] = agg["fill_rate_pooled"].round(4)
    agg["fill_rate_mean_per_sku_fold"] = agg["fill_rate_mean_per_sku_fold"].round(4)
    agg["stockout_rate"] = agg["stockout_rate"].round(4)

    cols = [
        "model", "policy", "scenario", "zero_tier",
        "fill_rate_pooled", "fill_rate_mean_per_sku_fold", "stockout_rate",
        "cost_total", "total_demand", "total_sold", "total_short", "count",
    ]
    return agg[cols].sort_values(["scenario", "zero_tier", "policy", "cost_total"])


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
