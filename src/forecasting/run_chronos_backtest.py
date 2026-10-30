"""Chay backtest cho ChronosModel zero-shot va cap nhat bang so sanh 5 mo hinh.

Tich hop truc tiep vao khung backtest 13 fold cua Phase 3a va 3b.
Cung 13 fold hold-out, cung 60 SKU, cung metrics, cung format bang.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
os.environ.setdefault("MPLCONFIGDIR", __import__("tempfile").gettempdir())

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_processed_daily
from src.forecasting.backtest import (
    load_backtest_config,
    get_sku_zero_tiers,
    evaluate_predictions,
)
from src.forecasting.features import _ensure_d_num
from src.forecasting.tsfm_model import ChronosModel


def run_single_chronos_fold(
    daily_df: pd.DataFrame,
    fold: int,
    origin_day: int,
    model: ChronosModel,
    horizon: int = 28,
) -> pd.DataFrame:
    """Chay 1 fold Chronos: fit -> predict -> merge actual."""
    daily_df = _ensure_d_num(daily_df)
    history_df = daily_df[daily_df["d_num"] <= origin_day].copy()
    test_df = daily_df[
        (daily_df["d_num"] > origin_day) & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()

    # Fit (zero-shot chi cap nhat context)
    model.fit(history_df)

    # Predict
    pred_df = model.predict(
        horizon=horizon,
        future_df=test_df[["date", "item_id"]].drop_duplicates(),
    )

    # Merge voi actual
    merged = test_df[["date", "item_id", "cat_id", "dept_id", "sales"]].merge(
        pred_df, on=["date", "item_id"], how="inner"
    )
    merged = merged.rename(columns={"sales": "actual"})
    merged["fold"] = fold
    merged["origin_day"] = origin_day
    merged["model"] = "Chronos"
    return merged


def run_chronos_backtest(
    daily_data_path: Union[str, Path] = "data/processed/sales_daily.parquet",
    backtest_config_path: Union[str, Path] = "configs/backtest.yaml",
    pred_dir: Union[str, Path] = "results/predictions",
    report_dir: Union[str, Path] = "results/forecasting",
    model_id: str = "amazon/chronos-t5-tiny",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chay toan bo backtest Chronos tren 13 fold va xuat ket qua."""
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    backtest_cfg = load_backtest_config(backtest_config_path)
    horizon = backtest_cfg["horizon"]
    num_folds = backtest_cfg["num_folds"]
    initial_origin = backtest_cfg["initial_origin"]
    step_size = backtest_cfg["step_size"]

    daily_df = load_processed_daily(daily_data_path)
    daily_df = _ensure_d_num(daily_df)
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial_origin)

    print(f"=== BACKTEST CHRONOS ZERO-SHOT: {num_folds} folds x 60 SKUs ===")
    print(f"Model ID: {model_id} (Pretrained Hugging Face, khong fine-tune)")

    # Khoi tao pipeline 1 lan de dung lai cho ca 13 fold
    model = ChronosModel(model_id=model_id, device="cpu", num_samples=20, context_length=512)

    all_fold_preds = []
    t_total = time.time()

    for k in range(num_folds):
        origin_day = initial_origin + k * step_size
        t_fold = time.time()
        fold_preds = run_single_chronos_fold(
            daily_df=daily_df,
            fold=k,
            origin_day=origin_day,
            model=model,
            horizon=horizon,
        )
        elapsed = time.time() - t_fold
        print(f"  Fold {k:2d} (origin d_{origin_day}): {len(fold_preds)} dong | {elapsed:.1f}s")
        all_fold_preds.append(fold_preds)

    total_time = time.time() - t_total
    print(f"Tong thoi gian backtest Chronos: {total_time:.1f}s")

    all_preds_df = pd.concat(all_fold_preds, ignore_index=True)

    # Luu file du bao
    pred_file = Path(pred_dir) / "chronos_predictions.parquet"
    all_preds_df.to_parquet(pred_file, index=False)
    print(f"Da luu {len(all_preds_df):,} dong du bao vao {pred_file}")

    # Danh gia
    overall_df, by_fold_df, by_cat_df, by_tier_df = evaluate_predictions(
        preds_df=all_preds_df,
        daily_df=daily_df,
        zero_tiers=zero_tiers,
    )

    overall_df.to_csv(Path(report_dir) / "chronos_overall.csv", index=False)
    by_fold_df.to_csv(Path(report_dir) / "chronos_by_fold.csv", index=False)
    by_cat_df.to_csv(Path(report_dir) / "chronos_by_category.csv", index=False)
    by_tier_df.to_csv(Path(report_dir) / "chronos_by_zero_tier.csv", index=False)

    # Ve bieu do mau 3 SKU (fold 0)
    _plot_sample_skus(all_preds_df, daily_df, zero_tiers, Path(report_dir))

    # Cap nhat bang so sanh gop 5 mo hinh
    update_5_model_comparisons(report_dir)

    return overall_df, by_fold_df, by_cat_df, by_tier_df


def _plot_sample_skus(
    all_preds_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    zero_tiers: Dict[str, str],
    report_dir: Path,
    fold_to_plot: int = 0,
) -> None:
    """Ve bieu do actual vs predicted cho 3 SKU mau - fold 0."""
    tier_order = ["smooth", "medium", "intermittent"]
    sample_skus = {}
    for tier in tier_order:
        items_in_tier = [k for k, v in zero_tiers.items() if v == tier]
        if items_in_tier:
            sample_skus[tier] = sorted(items_in_tier)[0]

    fold_preds = all_preds_df[all_preds_df["fold"] == fold_to_plot]
    origin = fold_preds["origin_day"].iloc[0]
    history_df = daily_df[
        (daily_df["d_num"] >= max(1, origin - 89)) & (daily_df["d_num"] <= origin)
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, tier in zip(axes, tier_order):
        if tier not in sample_skus:
            continue
        sku = sample_skus[tier]

        hist_sku = history_df[history_df["item_id"] == sku].sort_values("date")
        pred_sku = fold_preds[fold_preds["item_id"] == sku].sort_values("date")

        ax.plot(hist_sku["date"], hist_sku["sales"], color="gray", alpha=0.7, label="Lich su (90d)")
        ax.plot(pred_sku["date"], pred_sku["actual"], color="navy", linewidth=2, label="Actual")
        ax.plot(pred_sku["date"], pred_sku["forecast"], color="darkorange",
                linewidth=2, linestyle="--", label="Chronos (Zero-shot)")
        ax.axvline(x=pd.to_datetime(hist_sku["date"].max()),
                   color="black", linestyle=":", alpha=0.5)
        ax.set_title(f"Tier: {tier}\n{sku}")
        ax.set_xlabel("Ngay")
        ax.set_ylabel("Sales")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Chronos Zero-shot: Actual vs Forecast (Fold {fold_to_plot})", y=1.02)
    plt.tight_layout()
    fig.savefig(report_dir / "chronos_sample_skus.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Da luu bieu do 3 SKU mau vao {report_dir}/chronos_sample_skus.png")


def update_5_model_comparisons(report_dir: Union[str, Path] = "results/forecasting") -> None:
    """Gop tat ca ket qua cua 5 mo hinh (3 Baseline + LightGBM + Chronos)."""
    rdir = Path(report_dir)

    # 1. By Fold
    fold_dfs = []
    for prefix in ["baseline", "lightgbm", "chronos"]:
        f = rdir / f"{prefix}_by_fold.csv"
        if f.exists():
            fold_dfs.append(pd.read_csv(f))
    comp_fold = pd.concat(fold_dfs, ignore_index=True)
    comp_fold = comp_fold.sort_values(by=["fold", "WAPE"]).reset_index(drop=True)
    comp_fold.to_csv(rdir / "model_comparison_by_fold.csv", index=False)

    # Tinh so fold thang MovingAverage
    ma_fold = comp_fold[comp_fold["model"] == "MovingAverage"].set_index("fold")
    win_stats = {}
    for model_name, grp in comp_fold.groupby("model"):
        m_grp = grp.set_index("fold")
        wape_wins = 0
        mase_wins = 0
        for f in range(13):
            if f in m_grp.index and f in ma_fold.index:
                if m_grp.loc[f, "WAPE"] < ma_fold.loc[f, "WAPE"]:
                    wape_wins += 1
                if m_grp.loc[f, "MASE"] < ma_fold.loc[f, "MASE"]:
                    mase_wins += 1
        win_stats[model_name] = {
            "wins_vs_ma_wape": f"{wape_wins}/13",
            "wins_vs_ma_mase": f"{mase_wins}/13",
        }

    # 2. Overall
    over_dfs = []
    for prefix in ["baseline", "lightgbm", "chronos"]:
        f = rdir / f"{prefix}_overall.csv"
        if f.exists():
            over_dfs.append(pd.read_csv(f))
    comp_over = pd.concat(over_dfs, ignore_index=True)
    comp_over["wins_vs_ma_wape"] = comp_over["model"].map(lambda m: win_stats.get(m, {}).get("wins_vs_ma_wape", "-"))
    comp_over["wins_vs_ma_mase"] = comp_over["model"].map(lambda m: win_stats.get(m, {}).get("wins_vs_ma_mase", "-"))
    comp_over = comp_over.sort_values(by="WAPE").reset_index(drop=True)
    comp_over.to_csv(rdir / "model_comparison_overall.csv", index=False)

    # 3. By Category
    cat_dfs = []
    for prefix in ["baseline", "lightgbm", "chronos"]:
        f = rdir / f"{prefix}_by_category.csv"
        if f.exists():
            cat_dfs.append(pd.read_csv(f))
    comp_cat = pd.concat(cat_dfs, ignore_index=True)
    comp_cat = comp_cat.sort_values(by=["cat_id", "WAPE"]).reset_index(drop=True)
    comp_cat.to_csv(rdir / "model_comparison_by_category.csv", index=False)

    # 4. By Zero Tier
    tier_dfs = []
    for prefix in ["baseline", "lightgbm", "chronos"]:
        f = rdir / f"{prefix}_by_zero_tier.csv"
        if f.exists():
            tier_dfs.append(pd.read_csv(f))
    comp_tier = pd.concat(tier_dfs, ignore_index=True)
    comp_tier = comp_tier.sort_values(by=["zero_tier", "WAPE"]).reset_index(drop=True)
    comp_tier.to_csv(rdir / "model_comparison_by_zero_tier.csv", index=False)
    print("Da cap nhat thanh cong ca 4 bang tong hop 5 mo hinh vao results/forecasting/")


if __name__ == "__main__":
    run_chronos_backtest()
