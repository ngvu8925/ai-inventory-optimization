"""Ma trận độ nhạy một tham số cho mô phỏng tồn kho — Phase 5 (sau chẩn đoán).

Phân biệt HAI loại độ nhạy:

1. Độ nhạy KẾ TOÁN CHI PHÍ (không đổi hành vi):
     - holding_cost_per_unit_per_day : A / B / C
     - shortage_cost_per_unit        : 1,0 / 2,0 / 4,0
     - ordering_cost                 : 2,5 / 5,0 / 10,0
   Ba tham số này chỉ thay trọng số quy đổi chi phí. ROP, Q, fill rate,
   tổng thiếu hàng và số đơn KHÔNG đổi. Vì vậy không cần chạy lại mô phỏng:
   tính lại trực tiếp từ `simulation_detail_normal.csv` bằng các đại lượng
   đã lưu (đơn vị-ngày tồn kho, số đơn, tổng thiếu).

2. Độ nhạy HÀNH VI:
     - service_level : 0,90 / 0,95 / 0,99
   service_level đổi z, do đó đổi SS, ROP và hành vi đặt hàng. Phải chạy lại
   mô phỏng cho đúng 3 cấu hình này.

Đầu ra:
  - results/inventory/sensitivity_matrix.csv    : mọi dòng kết quả + thứ hạng
  - results/inventory/sensitivity_rankings.csv  : bảng xếp hạng gọn
  - results/inventory/sensitivity_inversions.csv: các cặp đảo hạng so với cơ sở
  - results/inventory/sensitivity_matrix.md     : bảng markdown

Chạy:
  $env:PYTHONPATH='.'
  .\\.venv\\Scripts\\python src/inventory/sensitivity.py
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.inventory.run_simulation import _summarize_results, run_all_simulations

# Cấu hình cơ sở = mức đang dùng trong assumptions.yaml (h mức B).
BASE_CFG = {
    "holding_cost_per_unit_per_day": 0.002032,
    "shortage_cost_per_unit": 2.0,
    "ordering_cost": 5.0,
    "service_level": 0.95,
}

# Khối kế toán: tính lại từ detail, không chạy lại mô phỏng.
ACCOUNTING_GRID = {
    "holding_cost_per_unit_per_day": [0.001355, 0.002032, 0.002709],
    "shortage_cost_per_unit": [1.0, 2.0, 4.0],
    "ordering_cost": [2.5, 5.0, 10.0],
}

# Khối hành vi: phải chạy lại mô phỏng.
BEHAVIOR_GRID = {
    "service_level": [0.90, 0.95, 0.99],
}

PARAM_LABELS = {
    "holding_cost_per_unit_per_day": "holding_cost_per_unit_per_day (h)",
    "shortage_cost_per_unit": "shortage_cost_per_unit (p)",
    "ordering_cost": "ordering_cost (K)",
    "service_level": "service_level",
}

SENSITIVITY_TYPE = {
    "holding_cost_per_unit_per_day": "cost accounting (no behavior change)",
    "shortage_cost_per_unit": "cost accounting (no behavior change)",
    "ordering_cost": "cost accounting (no behavior change)",
    "service_level": "behavior change (ROP/SS/Q)",
}

ORACLE_POLICY = "oracle_reference"


def _policy_label(row: pd.Series) -> str:
    """Tên hiển thị cho 1 dòng chính sách."""
    return f"{row['model']} | {row['policy']}"


def _run_config(overrides: dict) -> pd.DataFrame:
    """Chạy mô phỏng normal với bộ ghi đè hành vi, ẩn log của pipeline."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        detail = run_all_simulations(
            stress_scenario="none",
            overrides=overrides,
            write_outputs=False,
        )
    return _summarize_results(detail)


def _rank(df: pd.DataFrame) -> pd.Series:
    """Xếp hạng theo cost_total tăng dần (1 = rẻ nhất)."""
    return df["cost_total"].rank(method="min", ascending=True).astype(int)


def _aggregate_base_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp các đại lượng KHÔNG phụ thuộc h, K, p từ detail gốc.

    holding cost gốc = h_base × Σ inventory_end. Vì vậy:
        Σ inventory_end = cost_holding_gốc / h_base
    Dùng lại đại lượng này để tính holding ở mức h khác mà không cần mô phỏng.
    """
    agg = (
        detail.groupby(["model", "policy"], as_index=False)
        .agg(
            sum_inventory_units=("cost_holding", "sum"),
            num_orders=("num_orders", "sum"),
            total_short=("total_short", "sum"),
            total_demand=("total_demand", "sum"),
            total_sold=("total_sold", "sum"),
            fill_rate_mean_per_sku_fold=("fill_rate", "mean"),
            stockout_rate=("stockout_rate", "mean"),
            count=("item_id", "count"),
        )
    )
    agg["sum_inventory_units"] = agg["sum_inventory_units"] / BASE_CFG[
        "holding_cost_per_unit_per_day"
    ]
    return agg


def _accounting_summary(agg: pd.DataFrame, param: str, value: float) -> pd.DataFrame:
    """Tính lại chi phí cho một mức h/K/p từ các đại lượng đã lưu."""
    cfg = dict(BASE_CFG)
    cfg[param] = float(value)

    holding = agg["sum_inventory_units"] * cfg["holding_cost_per_unit_per_day"]
    ordering = agg["num_orders"] * cfg["ordering_cost"]
    shortage = agg["total_short"] * cfg["shortage_cost_per_unit"]
    total = holding + ordering + shortage

    out = agg.copy()
    out["scenario"] = "normal"
    out["cost_holding"] = holding.round(2)
    out["cost_ordering"] = ordering.round(2)
    out["cost_shortage"] = shortage.round(2)
    out["cost_total"] = total.round(2)
    out["fill_rate_pooled"] = np.where(
        out["total_demand"] > 0, 1.0 - out["total_short"] / out["total_demand"], 1.0
    )
    out["fill_rate_pooled"] = out["fill_rate_pooled"].round(4)
    out["fill_rate_mean_per_sku_fold"] = out["fill_rate_mean_per_sku_fold"].round(4)
    out["stockout_rate"] = out["stockout_rate"].round(4)
    for col in ["total_demand", "total_sold", "total_short"]:
        out[col] = out[col].round(2)
    out["sum_inventory_units"] = out["sum_inventory_units"].round(4)
    return out


def _verify_accounting_reconstruction(
    agg: pd.DataFrame, base_detail: pd.DataFrame
) -> dict:
    """Đối chiếu chi phí dựng lại ở mức cơ sở với chi phí mô phỏng gốc."""
    recon = _accounting_summary(
        agg,
        "holding_cost_per_unit_per_day",
        BASE_CFG["holding_cost_per_unit_per_day"],
    )
    original = (
        base_detail.groupby(["model", "policy"], as_index=False)["cost_total"]
        .sum()
        .rename(columns={"cost_total": "cost_total_original"})
    )
    merged = recon.merge(original, on=["model", "policy"], how="inner")
    diff = (merged["cost_total"] - merged["cost_total_original"]).abs()
    return {
        "n_policies": int(len(merged)),
        "max_abs_cost_diff": round(float(diff.max()) if len(diff) else np.nan, 6),
        "sum_abs_cost_diff": round(float(diff.sum()) if len(diff) else np.nan, 4),
    }


def _add_block_columns(
    summary: pd.DataFrame,
    param: str,
    value: float,
) -> pd.DataFrame:
    """Gắn nhãn tham số, loại độ nhạy, giá trị và hạng cho 1 cấu hình."""
    out = summary.copy()
    out["parameter"] = param
    out["parameter_label"] = PARAM_LABELS[param]
    out["sensitivity_type"] = SENSITIVITY_TYPE[param]
    out["value"] = float(value)
    out["base_value"] = BASE_CFG[param]
    out["is_base_config"] = abs(float(value) - float(BASE_CFG[param])) < 1e-12
    out["rank_cost"] = _rank(out)
    out["run_seconds"] = 0.0
    return out


def build_sensitivity_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Xây ma trận: kế toán từ detail có sẵn + chạy lại 3 mức service_level."""
    out_dir = Path("results/inventory")
    detail_path = out_dir / "simulation_detail_normal.csv"
    if not detail_path.exists():
        raise FileNotFoundError(
            f"Thiếu {detail_path}. Hãy chạy run_simulation.py cho kịch bản normal trước."
        )

    base_detail = pd.read_csv(detail_path)
    agg = _aggregate_base_detail(base_detail)
    verification = _verify_accounting_reconstruction(agg, base_detail)
    print(
        "[KIỂM TRA] Dựng lại chi phí ở cấu hình cơ sở: "
        f"{verification['n_policies']} chính sách, "
        f"max |diff| = {verification['max_abs_cost_diff']}."
    )

    rows = []

    # (1) Khối kế toán: không chạy lại mô phỏng.
    for param, values in ACCOUNTING_GRID.items():
        for value in values:
            summary = _accounting_summary(agg, param, value)
            rows.append(_add_block_columns(summary, param, value))
            print(
                f"[kế toán] {param} = {value} — rẻ nhất: "
                f"{summary.loc[summary['cost_total'].idxmin(), 'model']}"
            )

    # (2) Khối hành vi: chạy lại đúng 3 mức service_level.
    total_runs = sum(len(v) for v in BEHAVIOR_GRID.values())
    run_i = 0
    t_start = time.perf_counter()
    for param, values in BEHAVIOR_GRID.items():
        for value in values:
            run_i += 1
            t0 = time.perf_counter()
            summary = _run_config({param: value})
            elapsed = time.perf_counter() - t0
            block = _add_block_columns(summary, param, value)
            block["run_seconds"] = round(elapsed, 2)
            rows.append(block)
            print(
                f"[hành vi {run_i}/{total_runs}] {param} = {value} ({elapsed:.1f}s) "
                f"— rẻ nhất: {summary.loc[summary['cost_total'].idxmin(), 'model']}"
            )
    print(f"Tổng thời gian chạy lại hành vi: {time.perf_counter() - t_start:.1f}s.")

    full = pd.concat(rows, ignore_index=True)
    rankings = full[
        [
            "parameter", "parameter_label", "sensitivity_type", "value", "base_value",
            "is_base_config", "model", "policy", "rank_cost", "cost_total",
            "fill_rate_pooled", "total_short", "num_orders", "run_seconds",
        ]
    ].sort_values(["parameter", "value", "rank_cost"]).reset_index(drop=True)
    return full, rankings


def find_inversions(full: pd.DataFrame) -> pd.DataFrame:
    """Tìm các cặp chính sách bị đảo thứ hạng so với cấu hình cơ sở."""
    base = full[full["is_base_config"]].drop_duplicates(
        subset=["model", "policy"]
    ).set_index(["model", "policy"])["rank_cost"]

    records = []
    for (param, value), grp in full.groupby(["parameter", "value"]):
        if grp["is_base_config"].iloc[0]:
            continue
        cur = grp.set_index(["model", "policy"])["rank_cost"]
        common = base.index.intersection(cur.index)
        for i, a in enumerate(common):
            for b in common[i + 1:]:
                base_diff = base[a] - base[b]
                cur_diff = cur[a] - cur[b]
                if (base_diff > 0) != (cur_diff > 0):
                    records.append(
                        {
                            "parameter": param,
                            "sensitivity_type": SENSITIVITY_TYPE[param],
                            "value": value,
                            "policy_a": f"{a[0]} | {a[1]}",
                            "policy_b": f"{b[0]} | {b[1]}",
                            "base_rank_a": int(base[a]),
                            "base_rank_b": int(base[b]),
                            "new_rank_a": int(cur[a]),
                            "new_rank_b": int(cur[b]),
                            "involves_oracle": bool(
                                a[1] == ORACLE_POLICY or b[1] == ORACLE_POLICY
                            ),
                        }
                    )
    return pd.DataFrame(records)


def _write_markdown(rankings: pd.DataFrame, inversions: pd.DataFrame, path: Path) -> None:
    """Xuất bảng markdown gọn để chép vào báo cáo."""
    lines = [
        "# Ma trận độ nhạy (kịch bản normal)",
        "",
        "Mỗi khối chỉ đổi MỘT tham số; các tham số còn lại ở mức cơ sở.",
        "`Oracle` là cận dưới chi phí (perfect foresight), **không phải mô hình**.",
        "",
        "Hai loại khối:",
        "",
        "- **cost accounting (no behavior change)**: h, K, p chỉ đổi trọng số",
        "  chi phí. ROP/Q/fill/tổng thiếu/số đơn không đổi; tính lại từ",
        "  `simulation_detail_normal.csv`, không chạy lại mô phỏng. Thứ hạng",
        "  chi phí vẫn có thể đổi vì trọng số chi phí thay đổi.",
        "- **behavior change (ROP/SS/Q)**: service_level đổi z nên đổi SS, ROP",
        "  và hành vi đặt hàng; phải chạy lại mô phỏng.",
        "",
    ]
    for param in list(ACCOUNTING_GRID) + list(BEHAVIOR_GRID):
        lines.append(f"## {PARAM_LABELS[param]}")
        lines.append("")
        lines.append(f"Loại: **{SENSITIVITY_TYPE[param]}**.")
        lines.append("")
        lines.append("| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |")
        lines.append("|---:|---:|---|---|---:|---:|---:|")
        sub = rankings[rankings["parameter"] == param].sort_values(["value", "rank_cost"])
        for value, grp in sub.groupby("value"):
            for _, r in grp.iterrows():
                lines.append(
                    f"| {value} | {r['rank_cost']} | {r['model']} | {r['policy']} "
                    f"| {r['cost_total']:,.2f} | {r['fill_rate_pooled']:.4f} "
                    f"| {r['total_short']:,.2f} |"
                )
        lines.append("")

    lines.append("## Các cặp bị đảo thứ hạng so với cấu hình cơ sở")
    lines.append("")
    if inversions.empty:
        lines.append("Không có cặp nào đảo thứ hạng.")
    else:
        lines.append(
            "| Tham số | Loại | Giá trị | Cặp A | Cặp B | Hạng cơ sở (A/B) | Hạng mới (A/B) | Liên quan Oracle |"
        )
        lines.append("|---|---|---:|---|---|---|---|---|")
        for _, r in inversions.iterrows():
            lines.append(
                f"| {r['parameter']} | {r['sensitivity_type']} | {r['value']} "
                f"| {r['policy_a']} | {r['policy_b']} "
                f"| {r['base_rank_a']}/{r['base_rank_b']} | {r['new_rank_a']}/{r['new_rank_b']} "
                f"| {'có' if r['involves_oracle'] else 'không'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path("results/inventory")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== MA TRẬN ĐỘ NHẠY — kịch bản normal, mỗi lần đổi MỘT tham số ===")
    print(
        "Khối kế toán (h, K, p): tính lại từ detail có sẵn. "
        "Khối hành vi (service_level): chạy lại 3 cấu hình."
    )
    full, rankings = build_sensitivity_results()

    full.to_csv(out_dir / "sensitivity_matrix.csv", index=False)
    rankings.to_csv(out_dir / "sensitivity_rankings.csv", index=False)

    inversions = find_inversions(full)
    inversions.to_csv(out_dir / "sensitivity_inversions.csv", index=False)
    _write_markdown(rankings, inversions, out_dir / "sensitivity_matrix.md")

    print(f"Đã lưu → {out_dir / 'sensitivity_matrix.csv'}")
    print(f"Đã lưu → {out_dir / 'sensitivity_rankings.csv'}")
    print(f"Đã lưu → {out_dir / 'sensitivity_inversions.csv'}")
    print(f"Đã lưu → {out_dir / 'sensitivity_matrix.md'}")


if __name__ == "__main__":
    main()
