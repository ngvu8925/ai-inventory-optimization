"""Chay backtest cho LightGBMModel va xuat bang so sanh 4 mo hinh.

Tich hop vao khung backtest 13 fold cua Phase 3a.
Cung 13 fold, cung metrics, cung format bang.
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
from src.forecasting.ml_model import LightGBMModel
from src.forecasting.features import FEATURE_COLS, _ensure_d_num


def load_lgbm_config(config_path: Union[str, Path] = "configs/lightgbm.yaml") -> dict:
    """Doc cau hinh LightGBM da tune."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {path}. Chay src/forecasting/tune_lgbm.py truoc."
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Xoa cac key noi bo (bat dau bang _)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def _get_train_origins_for_fold(
    origin_day: int,
    step: int = 7,
    min_hist: int = 56,
) -> List[int]:
    """Lay danh sach cac origin huan luyen cho fold k.

    Cac origin cach nhau 7 ngay, bat dau tu min_hist, ket thuc o origin_day.
    Tat ca cac origin nay <= origin_day (khong ro ri).
    """
    origins = list(range(min_hist, origin_day + 1, step))
    if not origins or origins[-1] != origin_day:
        origins.append(origin_day)
    return origins


def run_single_lgbm_fold(
    daily_df: pd.DataFrame,
    fold: int,
    origin_day: int,
    lgbm_params: Dict,
    horizon: int = 28,
) -> Tuple[pd.DataFrame, LightGBMModel]:
    """Chay 1 fold LightGBM: train -> predict -> merge actual.

    Dung d_num (int) de filter nhanh thay vi string set.
    """
    daily_df = _ensure_d_num(daily_df)
    history_df = daily_df[daily_df["d_num"] <= origin_day]
    test_df = daily_df[
        (daily_df["d_num"] > origin_day) & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()

    # future_context_df: chua gia + lich cua giai doan du bao
    future_context_df = test_df[[
        "date", "item_id", "sell_price", "wday", "month",
        "event_name_1", "snap_CA",
    ]].copy()

    # Khoi tao model voi cac origin huan luyen (toi da den origin_day - horizon)
    train_origins = _get_train_origins_for_fold(origin_day - horizon, step=7, min_hist=56)

    model = LightGBMModel(
        lgbm_params=lgbm_params,
        train_origins=train_origins,
        horizon=horizon,
        name="LightGBM",
    )
    model.fit(history_df)

    pred_df = model.predict(
        horizon=horizon,
        future_df=test_df[["date", "item_id"]].drop_duplicates(),
        future_context_df=future_context_df,
    )

    # Merge voi actual
    merged = test_df[["date", "item_id", "cat_id", "dept_id", "sales"]].merge(
        pred_df, on=["date", "item_id"], how="inner"
    )
    merged = merged.rename(columns={"sales": "actual"})
    merged["fold"] = fold
    merged["origin_day"] = origin_day
    merged["model"] = "LightGBM"
    return merged, model


def run_lgbm_backtest(
    daily_data_path: Union[str, Path] = "data/processed/sales_daily.parquet",
    lgbm_config_path: Union[str, Path] = "configs/lightgbm.yaml",
    backtest_config_path: Union[str, Path] = "configs/backtest.yaml",
    pred_dir: Union[str, Path] = "results/predictions",
    report_dir: Union[str, Path] = "results/forecasting",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chay toan bo backtest LightGBM tren 13 fold va xuat ket qua."""
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    lgbm_params = load_lgbm_config(lgbm_config_path)
    backtest_cfg = load_backtest_config(backtest_config_path)
    horizon = backtest_cfg["horizon"]
    num_folds = backtest_cfg["num_folds"]
    initial_origin = backtest_cfg["initial_origin"]
    step_size = backtest_cfg["step_size"]

    daily_df = load_processed_daily(daily_data_path)
    # Them d_num 1 lan de filter nhanh trong toan bo backtest
    daily_df = _ensure_d_num(daily_df)
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial_origin)

    print(f"=== BACKTEST LIGHTGBM: {num_folds} folds x 60 SKUs ===")
    print(f"Cau hinh LightGBM: {lgbm_params}")

    all_fold_preds = []
    last_fold_model = None  # luu model fold cuoi de lay feature importance

    t_total = time.time()
    for k in range(num_folds):
        origin_day = initial_origin + k * step_size
        t_fold = time.time()
        fold_preds, fold_model = run_single_lgbm_fold(
            daily_df=daily_df,
            fold=k,
            origin_day=origin_day,
            lgbm_params=lgbm_params,
            horizon=horizon,
        )
        elapsed = time.time() - t_fold
        wape_fold = fold_preds["actual"].sum()
        print(f"  Fold {k:2d} (origin d_{origin_day}): {len(fold_preds)} dong | {elapsed:.1f}s")
        all_fold_preds.append(fold_preds)
        last_fold_model = fold_model

    total_time = time.time() - t_total
    print(f"Tong thoi gian backtest: {total_time:.1f}s")

    all_preds_df = pd.concat(all_fold_preds, ignore_index=True)

    # Kiem tra WAPE tong the ngay de phat hien bat thuong
    from src.forecasting.metrics import calc_wape
    overall_wape = calc_wape(all_preds_df["actual"].values, all_preds_df["forecast"].values)
    print(f"\nWAPE tong the: {overall_wape:.4f}")
    if overall_wape < 0.3:
        raise ValueError(
            f"WAPE qua thap ({overall_wape:.4f}) - nghi ngo ro ri du lieu! Dung lai."
        )

    # Luu du bao
    pred_file = Path(pred_dir) / "lightgbm_predictions.parquet"
    all_preds_df.to_parquet(pred_file, index=False)
    print(f"Da luu {len(all_preds_df):,} dong du bao vao {pred_file}")

    # Danh gia
    overall_df, by_fold_df, by_cat_df, by_tier_df = evaluate_predictions(
        preds_df=all_preds_df,
        daily_df=daily_df,
        zero_tiers=zero_tiers,
    )

    overall_df.to_csv(Path(report_dir) / "lightgbm_overall.csv", index=False)
    by_fold_df.to_csv(Path(report_dir) / "lightgbm_by_fold.csv", index=False)
    by_cat_df.to_csv(Path(report_dir) / "lightgbm_by_category.csv", index=False)
    by_tier_df.to_csv(Path(report_dir) / "lightgbm_by_zero_tier.csv", index=False)

    # Feature importance
    if last_fold_model is not None:
        fi_df = last_fold_model.get_feature_importance()
        fi_df.to_csv(Path(report_dir) / "lgbm_feature_importance.csv", index=False)
        _plot_feature_importance(fi_df, Path(report_dir))
        _plot_sample_skus(all_preds_df, daily_df, zero_tiers, Path(report_dir))

    return overall_df, by_fold_df, by_cat_df, by_tier_df


def _plot_feature_importance(fi_df: pd.DataFrame, report_dir: Path) -> None:
    """Ve bieu do top 15 feature quan trong nhat (gain)."""
    top_n = min(15, len(fi_df))
    top_fi = fi_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_fi["feature"][::-1], top_fi["importance_gain"][::-1], color="steelblue")
    ax.set_title("LightGBM Feature Importance (Gain) - Top 15")
    ax.set_xlabel("Importance (Gain)")
    plt.tight_layout()
    fig.savefig(report_dir / "lgbm_feature_importance.png", dpi=160)
    plt.close(fig)
    print(f"Da luu bieu do feature importance vao {report_dir}/lgbm_feature_importance.png")


def _plot_sample_skus(
    all_preds_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    zero_tiers: Dict[str, str],
    report_dir: Path,
    fold_to_plot: int = 0,
) -> None:
    """Ve bieu do actual vs predicted cho 3 SKU mau (smooth, medium, intermittent) - fold 0."""
    tier_order = ["smooth", "medium", "intermittent"]
    sample_skus = {}
    for tier in tier_order:
        items_in_tier = [k for k, v in zero_tiers.items() if v == tier]
        if items_in_tier:
            sample_skus[tier] = sorted(items_in_tier)[0]

    fold_preds = all_preds_df[all_preds_df["fold"] == fold_to_plot]
    origin = fold_preds["origin_day"].iloc[0]
    # Dung d_num neu co, fallback sang string set
    if "d_num" in daily_df.columns:
        history_df = daily_df[
            (daily_df["d_num"] >= max(1, origin - 89)) & (daily_df["d_num"] <= origin)
        ].copy()
    else:
        history_d_set = set(f"d_{i}" for i in range(max(1, origin - 89), origin + 1))
        history_df = daily_df[daily_df["d"].isin(history_d_set)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, tier in zip(axes, tier_order):
        if tier not in sample_skus:
            continue
        sku = sample_skus[tier]

        hist_sku = history_df[history_df["item_id"] == sku].sort_values("date")
        pred_sku = fold_preds[fold_preds["item_id"] == sku].sort_values("date")

        ax.plot(hist_sku["date"], hist_sku["sales"], color="gray", alpha=0.7, label="Lich su (90d)")
        ax.plot(pred_sku["date"], pred_sku["actual"], color="navy", linewidth=2, label="Actual")
        ax.plot(pred_sku["date"], pred_sku["forecast"], color="crimson",
                linewidth=2, linestyle="--", label="LightGBM")
        ax.axvline(x=pd.to_datetime(hist_sku["date"].max()),
                   color="black", linestyle=":", alpha=0.5)
        ax.set_title(f"Tier: {tier}\n{sku}")
        ax.set_xlabel("Ngay")
        ax.set_ylabel("Sales")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"LightGBM: Actual vs Forecast (Fold {fold_to_plot})", y=1.02)
    plt.tight_layout()
    fig.savefig(report_dir / "lgbm_sample_skus.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Da luu bieu do 3 SKU mau vao {report_dir}/lgbm_sample_skus.png")


def build_comparison_table(report_dir: Union[str, Path] = "results/forecasting") -> pd.DataFrame:
    """Gop 4 bang overall thanh 1 bang so sanh 4 mo hinh."""
    report_dir = Path(report_dir)
    dfs = []
    for fname in ["baseline_overall.csv", "lightgbm_overall.csv"]:
        fpath = report_dir / fname
        if fpath.exists():
            dfs.append(pd.read_csv(fpath))
    if not dfs:
        raise FileNotFoundError("Chua co file ket qua. Chay backtest truoc.")
    comp = pd.concat(dfs, ignore_index=True)
    comp = comp.sort_values("WAPE").reset_index(drop=True)
    comp.to_csv(report_dir / "comparison_all_models_overall.csv", index=False)
    return comp


if __name__ == "__main__":
    overall, by_fold, by_cat, by_tier = run_lgbm_backtest()
    print("\n--- BANG KET QUA TONG THE LIGHTGBM ---")
    print(overall.to_string(index=False))

    try:
        comp = build_comparison_table()
        print("\n--- BANG SO SANH 4 MO HINH (OVERALL) ---")
        print(comp.to_string(index=False))
    except FileNotFoundError as e:
        print(str(e))
