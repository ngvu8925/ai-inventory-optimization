"""Chẩn đoán Phase 5 — không sửa code/chính sách hiện có.

Mục tiêu: giải thích các quan sát bất thường trong kết quả mô phỏng tồn kho,
chỉ đọc dữ liệu và chạy lại các phép kiểm tra độc lập. Không sửa
policies.py / simulator.py / config.

Các phần:
  1. Kiểm tra nhất quán: dựng lại chính sách dự báo với cùng cửa sổ, cùng
     nguồn sigma, cùng tần suất cập nhật như Baseline; so khớp từng dòng.
  2. Độ phủ phân vị hold-out của 3 biến thể Chronos (tổng thể, zero_tier,
     horizon).
  3. Chênh lệch giữa các chính sách theo zero_tier.
  4. Công thức ROP/Q của quantile_forecast và đối chiếu thực nghiệm với
     point_forecast.
  5. Vai trò của shortage_cost trong ma trận độ nhạy.

Chạy:
  $env:PYTHONPATH='.'
  .\\.venv\\Scripts\\python src/inventory/diagnostics.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from statistics import NormalDist

import numpy as np
import pandas as pd

from src.data.loader import load_processed_daily
from src.forecasting.backtest import get_sku_zero_tiers, load_backtest_config
from src.forecasting.features import _ensure_d_num
from src.inventory.policies import (
    calc_initial_inventory,
    load_assumptions,
    policy_baseline,
    z_from_service_level,
)
from src.inventory.simulator import run_simulation_for_item

CHRONOS_FILES = {
    "Chronos2_History": "results/predictions/chronos2_history_predictions.parquet",
    "Chronos2_Covariates": "results/predictions/chronos2_covariates_predictions.parquet",
    "Chronos_Bolt": "results/predictions/chronos_bolt_predictions.parquet",
}
MA_FILE = "results/predictions/movingaverage_predictions.parquet"

OUT_DIR = Path("results/inventory")
Z_QUANTILE_90 = NormalDist().inv_cdf(0.9)


# =========================================================
# Dữ liệu dùng chung
# =========================================================
def load_context():
    """Đọc dữ liệu ngày, config và phân loại zero_tier."""
    daily = _ensure_d_num(load_processed_daily("data/processed/sales_daily.parquet"))
    bt = load_backtest_config("configs/backtest.yaml")
    cfg = load_assumptions("configs/assumptions.yaml")
    tiers = get_sku_zero_tiers(daily, last_train_day=int(bt["initial_origin"]))
    return daily, bt, cfg, tiers


def _simulate_params(
    item_id: str,
    fold: int,
    actual: np.ndarray,
    rop: float,
    qty: float,
    init_inv: float,
    cfg: dict,
) -> dict:
    """Chạy mô phỏng với tham số đã tính, dùng đúng simulator hiện có."""
    return run_simulation_for_item(
        item_id=item_id,
        fold=fold,
        actual_demand=actual,
        rop=rop,
        order_qty=qty,
        lead_time=int(cfg["lead_time_days"]),
        initial_inventory=init_inv,
        holding_cost_per_unit_per_day=float(cfg["holding_cost_per_unit_per_day"]),
        ordering_cost=float(cfg["ordering_cost"]),
        shortage_cost_per_unit=float(cfg["shortage_cost_per_unit"]),
    )


# =========================================================
# 1. Kiểm tra nhất quán Baseline ↔ chính sách dự báo
# =========================================================
def consistency_test(daily: pd.DataFrame, bt: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dựng lại Baseline bằng đường dự báo và so khớp từng SKU×fold.

    Logic: nếu một chính sách dự báo dùng
      - cùng cửa sổ 91 ngày,
      - cùng σ của cửa sổ 91 ngày (ddof=1),
      - cùng Q = d̄ × 14,
      - cùng tần suất cập nhật: chỉ tại origin của fold,
    thì kết quả mô phỏng phải trùng Baseline từng dòng.
    """
    L = int(cfg["lead_time_days"])
    q_days = int(cfg["reorder_quantity_days"])
    z = z_from_service_level(cfg["service_level"])
    horizon = int(bt["horizon"])
    num_folds = int(bt["num_folds"])
    initial_origin = int(bt["initial_origin"])
    step = int(bt["step_size"])
    items = sorted(daily["item_id"].unique())

    # Giá trị MovingAverage thực tế trong file dự báo (để đối chiếu mean 28 ngày)
    ma_pred = pd.read_parquet(MA_FILE)
    ma_const = (
        ma_pred.groupby(["fold", "item_id"])["forecast"].first().to_dict()
    )
    recompute_diff = []
    for (fold_k, item_id), val in ma_const.items():
        origin = initial_origin + fold_k * step
        hist = daily[
            (daily["item_id"] == item_id) & (daily["d_num"] <= origin)
        ]["sales"].values.astype(float)
        if len(hist) > 0:
            recompute_diff.append(abs(float(np.mean(hist[-28:])) - float(val)))

    variant_specs = {
        "baseline": {
            "window": 0,
            "sigma_source": "window_91",
            "q_basis": "window_91",
            "update": "origin_only",
            "desc": "policy_baseline hiện có (91 ngày, σ cửa sổ, Q=d̄×14)",
        },
        "forecast_ma91_sigma91_q91": {
            "window": 91,
            "sigma_source": "window_91",
            "q_basis": "window_91",
            "update": "origin_only",
            "desc": "Dự báo = mean 91 ngày; σ 91 ngày; Q = d̄91×14",
        },
        "forecast_ma91_sigma_history_q91": {
            "window": 91,
            "sigma_source": "full_history",
            "q_basis": "window_91",
            "update": "origin_only",
            "desc": "Dự báo = mean 91 ngày; σ toàn bộ lịch sử; Q = d̄91×14",
        },
        "forecast_ma28_sigma91_q28": {
            "window": 28,
            "sigma_source": "window_91",
            "q_basis": "window_28",
            "update": "origin_only",
            "desc": "Dự báo = mean 28 ngày; σ 91 ngày; Q = d̄28×14",
        },
        "forecast_ma28_sigma28_q28": {
            "window": 28,
            "sigma_source": "window_28",
            "q_basis": "window_28",
            "update": "origin_only",
            "desc": "Dự báo = mean 28 ngày; σ 28 ngày; Q = d̄28×14",
        },
        "forecast_ma28_sigma_history_q28": {
            "window": 28,
            "sigma_source": "full_history",
            "q_basis": "window_28",
            "update": "origin_only",
            "desc": "Đúng chính sách MovingAverage point_forecast đang dùng",
        },
    }

    rows: dict[str, list[dict]] = {name: [] for name in variant_specs}

    for fold_k in range(num_folds):
        origin = initial_origin + fold_k * step
        hist_df = daily[daily["d_num"] <= origin]

        for item_id in items:
            hist = hist_df[hist_df["item_id"] == item_id]["sales"].values.astype(float)
            if len(hist) == 0:
                continue
            actual = daily[
                (daily["item_id"] == item_id)
                & (daily["d_num"] > origin)
                & (daily["d_num"] <= origin + horizon)
            ].sort_values("d_num")["sales"].values.astype(float)
            if len(actual) == 0:
                continue

            init_inv = calc_initial_inventory(hist, cfg)
            base = policy_baseline(hist, cfg)

            win91 = hist[-91:] if len(hist) >= 91 else hist
            win28 = hist[-28:] if len(hist) >= 28 else hist
            mean91 = float(np.mean(win91))
            mean28 = float(np.mean(win28))
            std91 = float(np.std(win91, ddof=1)) if len(win91) > 1 else 0.0
            std28 = float(np.std(win28, ddof=1)) if len(win28) > 1 else 0.0
            std_hist = float(np.std(hist, ddof=1)) if len(hist) > 1 else 0.0

            params = {
                "baseline": (base["rop"], base["order_qty"]),
                "forecast_ma91_sigma91_q91": (
                    round(mean91 * L + z * std91 * np.sqrt(L), 4),
                    round(max(mean91 * q_days, 1.0), 4),
                ),
                "forecast_ma91_sigma_history_q91": (
                    round(mean91 * L + z * std_hist * np.sqrt(L), 4),
                    round(max(mean91 * q_days, 1.0), 4),
                ),
                "forecast_ma28_sigma91_q28": (
                    round(mean28 * L + z * std91 * np.sqrt(L), 4),
                    round(max(mean28 * q_days, 1.0), 4),
                ),
                "forecast_ma28_sigma28_q28": (
                    round(mean28 * L + z * std28 * np.sqrt(L), 4),
                    round(max(mean28 * q_days, 1.0), 4),
                ),
                "forecast_ma28_sigma_history_q28": (
                    round(mean28 * L + z * std_hist * np.sqrt(L), 4),
                    round(max(mean28 * q_days, 1.0), 4),
                ),
            }

            for name, (rop, qty) in params.items():
                sim = _simulate_params(item_id, fold_k, actual, rop, qty, init_inv, cfg)
                sim["variant"] = name
                rows[name].append(sim)

    # Tổng hợp từng biến thể
    summary_rows = []
    detail_by_variant = {}
    for name, recs in rows.items():
        df = pd.DataFrame(recs)
        detail_by_variant[name] = df
        spec = variant_specs[name]
        summary_rows.append(
            {
                "variant": name,
                "forecast_window_days": spec["window"],
                "sigma_source": spec["sigma_source"],
                "q_basis": spec["q_basis"],
                "update_frequency": spec["update"],
                "description": spec["desc"],
                "total_cost": round(df["cost_total"].sum(), 2),
                "fill_rate_pooled": round(
                    1.0 - df["total_short"].sum() / df["total_demand"].sum(), 4
                ),
                "total_short": round(df["total_short"].sum(), 2),
                "num_orders": int(df["num_orders"].sum()),
                "mean_avg_inventory": round(df["avg_inventory"].mean(), 4),
                "mean_rop": round(df["rop_used"].mean(), 4),
                "mean_order_qty": round(df["order_qty_used"].mean(), 4),
            }
        )

    summary = pd.DataFrame(summary_rows)

    base_df = detail_by_variant["baseline"].set_index(["fold", "item_id"]).sort_index()
    for i, row in summary.iterrows():
        name = row["variant"]
        df = detail_by_variant[name].set_index(["fold", "item_id"]).sort_index()
        cost_diff = (df["cost_total"] - base_df["cost_total"]).abs()
        short_diff = (df["total_short"] - base_df["total_short"]).abs()
        summary.loc[i, "delta_cost_vs_baseline"] = round(
            row["total_cost"] - summary.loc[0, "total_cost"], 2
        )
        summary.loc[i, "delta_fill_vs_baseline"] = round(
            row["fill_rate_pooled"] - summary.loc[0, "fill_rate_pooled"], 4
        )
        summary.loc[i, "delta_short_vs_baseline"] = round(
            row["total_short"] - summary.loc[0, "total_short"], 2
        )
        summary.loc[i, "rows_identical_to_baseline"] = int(
            ((cost_diff < 1e-9) & (short_diff < 1e-9)).sum()
        )
        summary.loc[i, "n_rows"] = int(len(df))
        summary.loc[i, "max_abs_row_cost_diff"] = round(float(cost_diff.max()), 6)
        summary.loc[i, "max_abs_row_short_diff"] = round(float(short_diff.max()), 6)

    # Đối chiếu biến thể "đúng MA đang dùng" với dòng MovingAverage trong file detail
    detail = pd.read_csv(OUT_DIR / "simulation_detail_normal.csv")
    actual_ma = detail[detail["model"] == "MovingAverage"]
    ma_total = float(actual_ma["cost_total"].sum())
    our_ma = float(
        summary.loc[
            summary["variant"] == "forecast_ma28_sigma_history_q28", "total_cost"
        ].iloc[0]
    )
    summary["matches_actual_movingaverage"] = summary["variant"].map(
        lambda v: abs(our_ma - ma_total) < 0.01
        if v == "forecast_ma28_sigma_history_q28"
        else pd.NA
    )

    # File mô tả khác biệt chính xác so với Baseline
    diff_rows = []
    for name, spec in variant_specs.items():
        diffs = []
        if spec["window"] not in (0, 91):
            diffs.append(f"cửa sổ dự báo {spec['window']} ngày (Baseline 91)")
        if spec["sigma_source"] == "full_history":
            diffs.append("σ lấy toàn bộ lịch sử (Baseline σ 91 ngày)")
        if spec["sigma_source"] == "window_28":
            diffs.append("σ lấy cửa sổ 28 ngày (Baseline σ 91 ngày)")
        if spec["q_basis"] == "window_28":
            diffs.append("Q dùng d̄ 28 ngày (Baseline d̄ 91 ngày)")
        diff_rows.append(
            {
                "variant": name,
                "window": spec["window"] if spec["window"] else 91,
                "sigma_source": spec["sigma_source"],
                "q_basis": spec["q_basis"],
                "update_frequency": "origin_only (giống Baseline)",
                "differences_vs_baseline": "; ".join(diffs) if diffs else "không có",
                "ma_forecast_recomputed_matches_file": bool(
                    max(recompute_diff) < 1e-9 if recompute_diff else False
                ),
            }
        )
    differences = pd.DataFrame(diff_rows)

    return summary, differences


# =========================================================
# 2. Độ phủ phân vị hold-out của Chronos
# =========================================================
def quantile_coverage(daily: pd.DataFrame, bt: dict, tiers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính tỷ lệ nhu cầu thực <= q0.5 / q0.9 / q0.95."""
    coverage_rows = []
    integrity_rows = []

    for model, path in CHRONOS_FILES.items():
        pred = pd.read_parquet(path).copy()
        pred["date"] = pd.to_datetime(pred["date"])
        pred = pred.sort_values(["fold", "item_id", "date"])
        pred["h"] = pred.groupby(["fold", "item_id"]).cumcount() + 1
        pred["zero_tier"] = pred["item_id"].map(tiers)

        merged = pred.merge(
            daily[["date", "item_id", "sales"]],
            on=["date", "item_id"],
            how="left",
            suffixes=("", "_daily"),
        )
        actual_diff = float((merged["actual"] - merged["sales"]).abs().max())

        def _cov(df: pd.DataFrame, col: str) -> float:
            return float((df["actual"] <= df[col]).mean())

        def _add(group_type: str, group: str, df: pd.DataFrame) -> None:
            coverage_rows.append(
                {
                    "model": model,
                    "group_type": group_type,
                    "group": group,
                    "n": int(len(df)),
                    "coverage_q50": round(_cov(df, "q_0.5"), 4),
                    "coverage_q90": round(_cov(df, "q_0.9"), 4),
                    "coverage_q95": round(_cov(df, "q_0.95"), 4),
                    "mean_actual": round(float(df["actual"].mean()), 4),
                    "mean_q50": round(float(df["q_0.5"].mean()), 4),
                }
            )

        _add("overall", "all", pred)
        for tier in ["smooth", "medium", "intermittent"]:
            _add("zero_tier", tier, pred[pred["zero_tier"] == tier])
        for h in range(1, int(bt["horizon"]) + 1):
            _add("horizon", f"h={h:02d}", pred[pred["h"] == h])

        integrity_rows.append(
            {
                "model": model,
                "n_rows": int(len(pred)),
                "n_q10_gt_q50": int((pred["q_0.1"] > pred["q_0.5"]).sum()),
                "n_q50_gt_q90": int((pred["q_0.5"] > pred["q_0.9"]).sum()),
                "n_q90_gt_q95": int((pred["q_0.9"] > pred["q_0.95"]).sum()),
                "share_q90_equal_q95": round(float((pred["q_0.9"] == pred["q_0.95"]).mean()), 4),
                "coverage_q50": round(_cov(pred, "q_0.5"), 4),
                "coverage_q90": round(_cov(pred, "q_0.9"), 4),
                "coverage_q95": round(_cov(pred, "q_0.95"), 4),
                "mean_actual": round(float(pred["actual"].mean()), 4),
                "mean_forecast": round(float(pred["forecast"].mean()), 4),
                "mean_bias_forecast_minus_actual": round(
                    float(pred["forecast"].mean() - pred["actual"].mean()), 4
                ),
                "max_abs_actual_vs_daily": round(actual_diff, 6),
            }
        )

    return pd.DataFrame(coverage_rows), pd.DataFrame(integrity_rows)


# =========================================================
# 3. Chênh lệch theo zero_tier
# =========================================================
def zero_tier_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc bảng có sẵn, tính chênh lệch fill/chi phí theo nhóm nhu cầu."""
    tier_df = pd.read_csv(OUT_DIR / "simulation_by_zero_tier_normal.csv")
    tier_df["cost_per_demand"] = tier_df["cost_total"] / tier_df["total_demand"]
    tier_df["short_per_demand"] = tier_df["total_short"] / tier_df["total_demand"]

    policies = tier_df[tier_df["policy"] != "oracle_reference"].copy()
    total_short_all = float(policies["total_short"].sum())
    total_demand_all = float(policies["total_demand"].sum())

    spread_rows = []
    for tier, grp in policies.groupby("zero_tier"):
        best_fill = grp.loc[grp["fill_rate_pooled"].idxmax()]
        worst_fill = grp.loc[grp["fill_rate_pooled"].idxmin()]
        spread_rows.append(
            {
                "zero_tier": tier,
                "n_policies": int(len(grp)),
                "demand_share": round(float(grp["total_demand"].iloc[0]) / total_demand_all, 4),
                "short_share": round(float(grp["total_short"].sum()) / total_short_all, 4),
                "fill_min": round(float(grp["fill_rate_pooled"].min()), 4),
                "fill_max": round(float(grp["fill_rate_pooled"].max()), 4),
                "fill_range": round(
                    float(grp["fill_rate_pooled"].max() - grp["fill_rate_pooled"].min()), 4
                ),
                "fill_best_policy": f"{best_fill['model']} | {best_fill['policy']}",
                "fill_worst_policy": f"{worst_fill['model']} | {worst_fill['policy']}",
                "cost_per_demand_min": round(float(grp["cost_per_demand"].min()), 4),
                "cost_per_demand_max": round(float(grp["cost_per_demand"].max()), 4),
                "cost_per_demand_range": round(
                    float(grp["cost_per_demand"].max() - grp["cost_per_demand"].min()), 4
                ),
            }
        )

    return tier_df.sort_values(["zero_tier", "policy", "cost_total"]), pd.DataFrame(spread_rows)


# =========================================================
# 4. Chính sách quantile: công thức và đối chiếu thực nghiệm
# =========================================================
def quantile_policy_diagnostics(
    daily: pd.DataFrame, bt: dict, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tính các thành phần ROP/Q của point vs quantile và tổng hợp thực nghiệm."""
    L = int(cfg["lead_time_days"])
    q_days = int(cfg["reorder_quantity_days"])
    z = z_from_service_level(cfg["service_level"])
    horizon = int(bt["horizon"])
    num_folds = int(bt["num_folds"])
    initial_origin = int(bt["initial_origin"])
    step = int(bt["step_size"])

    component_rows = []
    for model, path in CHRONOS_FILES.items():
        pred = pd.read_parquet(path).copy()
        pred["date"] = pd.to_datetime(pred["date"])
        pred = pred.sort_values(["fold", "item_id", "date"])

        per_fold_item = []
        for fold_k in range(num_folds):
            origin = initial_origin + fold_k * step
            hist_df = daily[daily["d_num"] <= origin]
            fold_pred = pred[pred["fold"] == fold_k]
            for item_id, grp in fold_pred.groupby("item_id"):
                hist = hist_df[hist_df["item_id"] == item_id]["sales"].values.astype(float)
                if len(hist) < 2:
                    continue
                grp = grp.sort_values("date")
                fc = grp["forecast"].values.astype(float)
                q10 = grp["q_0.1"].values.astype(float)
                q90 = grp["q_0.9"].values.astype(float)
                q50 = grp["q_0.5"].values.astype(float)
                std_hist = float(np.std(hist, ddof=1))
                sigma_lead = float(
                    np.sqrt(
                        np.sum(
                            np.square(
                                np.maximum(q90[:L] - q10[:L], 0.0) / (2 * Z_QUANTILE_90)
                            )
                        )
                    )
                )
                ss_point = z * std_hist * np.sqrt(L)
                ss_quant = z * sigma_lead
                rop_point = float(np.sum(fc[:L]) + ss_point)
                rop_quant = float(np.sum(q50[:L]) + ss_quant)
                q_point = float(max(np.sum(fc[:q_days]) if len(fc) >= q_days else np.mean(fc) * q_days, 1.0))
                q_quant = float(max(np.sum(q50[:q_days]) if len(q50) >= q_days else np.mean(q50) * q_days, 1.0))
                per_fold_item.append(
                    {
                        "sum_forecast_L": float(np.sum(fc[:L])),
                        "sum_q50_L": float(np.sum(q50[:L])),
                        "sigma_hist_sqrtL": std_hist * np.sqrt(L),
                        "sigma_lead": sigma_lead,
                        "ss_point": ss_point,
                        "ss_quant": ss_quant,
                        "rop_point": rop_point,
                        "rop_quant": rop_quant,
                        "q_point": q_point,
                        "q_quant": q_quant,
                    }
                )
        df = pd.DataFrame(per_fold_item)
        component_rows.append(
            {
                "model": model,
                "n_sku_fold": int(len(df)),
                "mean_sum_forecast_lead": round(float(df["sum_forecast_L"].mean()), 4),
                "mean_sum_q50_lead": round(float(df["sum_q50_L"].mean()), 4),
                "mean_sigma_hist_sqrtL": round(float(df["sigma_hist_sqrtL"].mean()), 4),
                "mean_sigma_lead_from_q10_q90": round(float(df["sigma_lead"].mean()), 4),
                "z_service": round(float(z), 4),
                "mean_ss_point": round(float(df["ss_point"].mean()), 4),
                "mean_ss_quantile": round(float(df["ss_quant"].mean()), 4),
                "mean_rop_point_formula": round(float(df["rop_point"].mean()), 4),
                "mean_rop_quantile_formula": round(float(df["rop_quant"].mean()), 4),
                "mean_rop_point_minus_quantile": round(
                    float((df["rop_point"] - df["rop_quant"]).mean()), 4
                ),
                "mean_q_point": round(float(df["q_point"].mean()), 4),
                "mean_q_quantile": round(float(df["q_quant"].mean()), 4),
                "mean_q_point_minus_quantile": round(
                    float((df["q_point"] - df["q_quant"]).mean()), 4
                ),
            }
        )

    # Thực nghiệm từ detail: số đơn, tồn kho trung bình, ROP/Q trung bình, fill, chi phí
    detail = pd.read_csv(OUT_DIR / "simulation_detail_normal.csv")
    detail = detail[detail["model"].isin(CHRONOS_FILES.keys())]
    policy_rows = []
    for (model, policy), grp in detail.groupby(["model", "policy"]):
        policy_rows.append(
            {
                "model": model,
                "policy": policy,
                "num_orders": int(grp["num_orders"].sum()),
                "mean_avg_inventory": round(float(grp["avg_inventory"].mean()), 4),
                "mean_rop_used": round(float(grp["rop_used"].mean()), 4),
                "mean_order_qty_used": round(float(grp["order_qty_used"].mean()), 4),
                "fill_rate_pooled": round(
                    1.0 - grp["total_short"].sum() / grp["total_demand"].sum(), 4
                ),
                "total_short": round(float(grp["total_short"].sum()), 2),
                "cost_total": round(float(grp["cost_total"].sum()), 2),
            }
        )

    return pd.DataFrame(component_rows), pd.DataFrame(policy_rows)


# =========================================================
# 5. Vai trò của shortage_cost trong ma trận độ nhạy
# =========================================================
def shortage_cost_diagnostics() -> pd.DataFrame:
    """Đối chiếu fill/total_short/đơn giá thiếu hàng giữa các mức p."""
    sens = pd.read_csv(OUT_DIR / "sensitivity_matrix.csv")
    sub = sens[sens["parameter"] == "shortage_cost_per_unit"].copy()
    sub["implied_shortage_unit_cost"] = np.where(
        sub["total_short"] > 0, sub["cost_shortage"] / sub["total_short"], np.nan
    )
    sub["cost_identity_error"] = (
        sub["cost_total"]
        - (sub["cost_holding"] + sub["cost_ordering"] + sub["cost_shortage"])
    ).abs()

    rows = []
    for (model, policy), grp in sub.groupby(["model", "policy"]):
        grp = grp.sort_values("value")
        g = {float(r["value"]): r for _, r in grp.iterrows()}
        p_vals = sorted(g)
        fills = [float(g[p]["fill_rate_pooled"]) for p in p_vals]
        shorts = [float(g[p]["total_short"]) for p in p_vals]
        rows.append(
            {
                "model": model,
                "policy": policy,
                "p_1.0_fill": round(fills[0], 4),
                "p_2.0_fill": round(fills[1], 4),
                "p_4.0_fill": round(fills[2], 4),
                "max_fill_diff_across_p": round(max(fills) - min(fills), 6),
                "p_1.0_total_short": round(shorts[0], 2),
                "p_2.0_total_short": round(shorts[1], 2),
                "p_4.0_total_short": round(shorts[2], 2),
                "max_short_diff_across_p": round(max(shorts) - min(shorts), 6),
                "implied_p_at_p1": round(float(g[1.0]["implied_shortage_unit_cost"]), 6),
                "implied_p_at_p2": round(float(g[2.0]["implied_shortage_unit_cost"]), 6),
                "implied_p_at_p4": round(float(g[4.0]["implied_shortage_unit_cost"]), 6),
                "max_cost_identity_error": round(float(grp["cost_identity_error"].max()), 6),
                "cost_delta_p4_minus_p1": round(
                    float(g[4.0]["cost_total"] - g[1.0]["cost_total"]), 2
                ),
                "expected_delta_3x_short": round(3.0 * float(g[2.0]["total_short"]), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy", "model"])


# =========================================================
# main
# =========================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, bt, cfg, tiers = load_context()

    print("[1/5] Kiểm tra nhất quán Baseline ↔ dự báo...")
    consistency, differences = consistency_test(daily, bt, cfg)
    consistency.to_csv(OUT_DIR / "diagnostics_consistency.csv", index=False)
    differences.to_csv(OUT_DIR / "diagnostics_consistency_differences.csv", index=False)

    print("[2/5] Độ phủ phân vị Chronos...")
    coverage, integrity = quantile_coverage(daily, bt, tiers)
    coverage.to_csv(OUT_DIR / "diagnostics_quantile_coverage.csv", index=False)
    integrity.to_csv(OUT_DIR / "diagnostics_quantile_integrity.csv", index=False)

    print("[3/5] Chênh lệch theo zero_tier...")
    tier_long, tier_spread = zero_tier_diagnostics()
    tier_long.to_csv(OUT_DIR / "diagnostics_zero_tier_normal.csv", index=False)
    tier_spread.to_csv(OUT_DIR / "diagnostics_zero_tier_spread.csv", index=False)

    print("[4/5] Công thức quantile và đối chiếu point/quantile...")
    components, policy_summary = quantile_policy_diagnostics(daily, bt, cfg)
    components.to_csv(OUT_DIR / "diagnostics_quantile_components.csv", index=False)
    policy_summary.to_csv(OUT_DIR / "diagnostics_quantile_policy.csv", index=False)

    print("[5/5] Vai trò shortage_cost...")
    shortage = shortage_cost_diagnostics()
    shortage.to_csv(OUT_DIR / "diagnostics_shortage_cost.csv", index=False)

    print("\nĐã ghi:")
    for name in [
        "diagnostics_consistency.csv",
        "diagnostics_consistency_differences.csv",
        "diagnostics_quantile_coverage.csv",
        "diagnostics_quantile_integrity.csv",
        "diagnostics_zero_tier_normal.csv",
        "diagnostics_zero_tier_spread.csv",
        "diagnostics_quantile_components.csv",
        "diagnostics_quantile_policy.csv",
        "diagnostics_shortage_cost.csv",
    ]:
        print(f"  {OUT_DIR / name}")


if __name__ == "__main__":
    main()
