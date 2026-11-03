"""Ma trận độ nhạy một tham số cho mô phỏng tồn kho — Phase 5.

Mỗi lần chỉ đổi MỘT tham số quanh cấu hình cơ sở, chạy trên kịch bản `normal`:

  - holding_cost_per_unit_per_day : A / B / C  (0,001355 / 0,002032 / 0,002709)
  - shortage_cost_per_unit        : 1,0 / 2,0 / 4,0
  - ordering_cost                 : 2,5 / 5,0 / 10,0
  - service_level                 : 0,90 / 0,95 / 0,99

Các tham số còn lại giữ nguyên như `configs/assumptions.yaml` (mức B).

Đầu ra:
  - results/inventory/sensitivity_matrix.csv   : mọi dòng kết quả + thứ hạng
  - results/inventory/sensitivity_rankings.csv : bảng xếp hạng gọn theo cấu hình
  - results/inventory/sensitivity_inversions.csv: các cặp bị đảo thứ hạng so
    với cấu hình cơ sở
  - results/inventory/sensitivity_matrix.md    : bảng markdown để chép báo cáo

Chạy:
  $env:PYTHONPATH='.'
  .\\.venv\\Scripts\\python src/inventory/sensitivity.py
"""

import contextlib
import io
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd

from src.inventory.run_simulation import run_all_simulations

# Cấu hình cơ sở = mức đang dùng trong assumptions.yaml (h mức B).
BASE_CFG = {
    "holding_cost_per_unit_per_day": 0.002032,
    "shortage_cost_per_unit": 2.0,
    "ordering_cost": 5.0,
    "service_level": 0.95,
}

# Lưới một tham số. Mỗi list phải chứa giá trị cơ sở để so sánh trực tiếp.
GRID = {
    "holding_cost_per_unit_per_day": [0.001355, 0.002032, 0.002709],
    "shortage_cost_per_unit": [1.0, 2.0, 4.0],
    "ordering_cost": [2.5, 5.0, 10.0],
    "service_level": [0.90, 0.95, 0.99],
}

# Nhãn dễ đọc cho báo cáo
PARAM_LABELS = {
    "holding_cost_per_unit_per_day": "holding_cost_per_unit_per_day (h)",
    "shortage_cost_per_unit": "shortage_cost_per_unit (p)",
    "ordering_cost": "ordering_cost (K)",
    "service_level": "service_level",
}

ORACLE_POLICY = "oracle_reference"


def _policy_label(row: pd.Series) -> str:
    """Tên hiển thị cho 1 dòng chính sách."""
    return f"{row['model']} | {row['policy']}"


def _run_config(overrides: dict) -> pd.DataFrame:
    """Chạy mô phỏng normal với bộ ghi đè, ẩn log của pipeline."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        detail = run_all_simulations(
            stress_scenario="none",
            overrides=overrides,
            write_outputs=False,
        )
    from src.inventory.run_simulation import _summarize_results

    return _summarize_results(detail)


def _rank(df: pd.DataFrame) -> pd.Series:
    """Xếp hạng theo cost_total tăng dần (1 = rẻ nhất)."""
    return df["cost_total"].rank(method="min", ascending=True).astype(int)


def build_sensitivity_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy toàn bộ ma trận và trả về (kết quả đầy đủ, bảng xếp hạng)."""
    rows = []
    total_runs = sum(len(v) for v in GRID.values())
    run_i = 0
    t_start = time.perf_counter()

    for param, values in GRID.items():
        for value in values:
            run_i += 1
            overrides = {param: value}
            t0 = time.perf_counter()
            summary = _run_config(overrides)
            elapsed = time.perf_counter() - t0

            summary = summary.copy()
            summary["parameter"] = param
            summary["parameter_label"] = PARAM_LABELS[param]
            summary["value"] = value
            summary["base_value"] = BASE_CFG[param]
            # Mỗi lần chỉ đổi 1 tham số nên cấu hình là cơ sở khi value trùng mức cơ sở.
            summary["is_base_config"] = abs(float(value) - float(BASE_CFG[param])) < 1e-12
            summary["rank_cost"] = _rank(summary)
            summary["run_seconds"] = round(elapsed, 2)
            rows.append(summary)

            print(
                f"[{run_i:02d}/{total_runs}] {param} = {value} "
                f"({elapsed:.1f}s) — rẻ nhất: "
                f"{summary.loc[summary['cost_total'].idxmin(), 'model']}"
            )

    full = pd.concat(rows, ignore_index=True)
    # Bảng xếp hạng gọn: mỗi cấu hình 11 dòng, sắp theo hạng.
    rankings = full[
        [
            "parameter", "parameter_label", "value", "base_value", "is_base_config",
            "model", "policy", "rank_cost", "cost_total",
            "fill_rate_pooled", "total_short",
        ]
    ].sort_values(["parameter", "value", "rank_cost"]).reset_index(drop=True)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nTổng thời gian: {total_elapsed:.1f}s cho {total_runs} cấu hình.")
    return full, rankings


def find_inversions(full: pd.DataFrame) -> pd.DataFrame:
    """Tìm các cặp chính sách bị đảo thứ hạng so với cấu hình cơ sở.

    Cấu hình cơ sở = dòng có is_base_config=True (h=B, p=2, K=5, SL=0,95).
    Chỉ so sánh trong cùng một tham số (mỗi lần đổi một tham số).
    """
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
    ]
    for param in GRID:
        lines.append(f"## {PARAM_LABELS[param]}")
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
        lines.append("| Tham số | Giá trị | Cặp A | Cặp B | Hạng cơ sở (A/B) | Hạng mới (A/B) | Liên quan Oracle |")
        lines.append("|---|---:|---|---|---|---|---|")
        for _, r in inversions.iterrows():
            lines.append(
                f"| {r['parameter']} | {r['value']} | {r['policy_a']} | {r['policy_b']} "
                f"| {r['base_rank_a']}/{r['base_rank_b']} | {r['new_rank_a']}/{r['new_rank_b']} "
                f"| {'có' if r['involves_oracle'] else 'không'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = Path("results/inventory")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== MA TRẬN ĐỘ NHẠY — kịch bản normal, mỗi lần đổi MỘT tham số ===")
    print(
        f"{sum(len(v) for v in GRID.values())} cấu hình × "
        f"{len(GRID)} tham số; mỗi cấu hình chạy toàn bộ chính sách × 13 fold × 60 SKU."
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
