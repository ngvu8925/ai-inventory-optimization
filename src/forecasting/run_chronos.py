"""Script chay backtest 13 fold cho 3 bien the Chronos Zero-shot trong .venv-tsfm.

3 bien the:
1. 'chronos2_history': Chronos-2 chi dung lich su (univariate)
2. 'chronos2_covariates': Chronos-2 + bien ngoai sinh biet truoc (oracle 28d: gia, SNAP, su kien, wday, month)
3. 'chronos_bolt': Chronos-Bolt (amazon/chronos-bolt-mini) chi dung lich su

Ket qua luu tai results/predictions/ duoi dang parquet voi day du phan vi:
- forecast (trung vi / median 0.5)
- q_0.1, q_0.5, q_0.9, q_0.95
"""

import argparse
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.loader import load_processed_daily
from src.forecasting.backtest import (
    load_backtest_config,
    get_sku_zero_tiers,
    evaluate_predictions,
)
from src.forecasting.features import _ensure_d_num
from src.forecasting.chronos_model import ChronosModel


VARIANTS = {
    "chronos2_history": {
        "name": "Chronos2_History",
        "file_prefix": "chronos2_history",
        "model_id": "amazon/chronos-2",
        "desc": "Chronos-2 chi dung lich su target (Univariate)",
    },
    "chronos2_covariates": {
        "name": "Chronos2_Covariates",
        "file_prefix": "chronos2_covariates",
        "model_id": "amazon/chronos-2",
        "desc": "Chronos-2 dung lich su + bien ngoai sinh biet truoc 28d",
    },
    "chronos_bolt": {
        "name": "Chronos_Bolt",
        "file_prefix": "chronos_bolt",
        "model_id": "amazon/chronos-bolt-mini",
        "desc": "Chronos-Bolt (mini) patch-based chi dung lich su",
    },
}


def run_single_chronos_fold(
    daily_df: pd.DataFrame,
    fold: int,
    origin_day: int,
    model: ChronosModel,
    horizon: int = 28,
) -> pd.DataFrame:
    """Chay 1 fold cho ChronosModel: fit -> predict -> merge actual."""
    daily_df = _ensure_d_num(daily_df)
    history_df = daily_df[daily_df["d_num"] <= origin_day].copy()
    test_df = daily_df[
        (daily_df["d_num"] > origin_day) & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()

    # Chuan bi has_event neu chua co
    if "has_event" not in test_df.columns:
        test_df["has_event"] = (
            test_df["event_name_1"].notna() & (test_df["event_name_1"] != "")
        ).astype(int)

    future_context_df = test_df[[
        "date", "item_id", "sell_price", "snap_CA", "has_event", "wday", "month"
    ]].copy()

    # Fit (zero-shot luu context lich su)
    model.fit(history_df)

    # Predict
    pred_df = model.predict(
        horizon=horizon,
        future_df=test_df[["date", "item_id"]].drop_duplicates(),
        future_context_df=future_context_df if model.variant == "chronos2_covariates" else None,
    )

    # Merge voi actual
    cols_to_merge = ["date", "item_id", "cat_id", "dept_id", "sales"]
    merged = test_df[cols_to_merge].merge(pred_df, on=["date", "item_id"], how="inner")
    merged = merged.rename(columns={"sales": "actual"})
    merged["fold"] = fold
    merged["origin_day"] = origin_day
    merged["model"] = model.name

    return merged


def run_variant_backtest(
    variant_key: str,
    daily_df: pd.DataFrame,
    zero_tiers: Dict[str, str],
    initial_origin: int = 1577,
    num_folds: int = 13,
    step_size: int = 28,
    horizon: int = 28,
    pred_dir: Union[str, Path] = "results/predictions",
    report_dir: Union[str, Path] = "results/forecasting",
    folds_to_run: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, float]:
    """Chay backtest cho 1 bien the Chronos qua cac fold chi dinh."""
    info = VARIANTS[variant_key]
    model_name = info["name"]
    file_prefix = info["file_prefix"]
    model_id = info["model_id"]

    print(f"\n=======================================================")
    print(f"BAT DAU BACKTEST: {model_name}")
    print(f"Mo ta: {info['desc']}")
    print(f"Model ID: {model_id} | Context max: 1024d")
    print(f"=======================================================")

    model = ChronosModel(
        variant=variant_key,
        model_id=model_id,
        context_length=1024,
        device="cpu",
        seed=42,
        name=model_name,
    )

    if folds_to_run is None:
        folds_to_run = list(range(num_folds))

    all_fold_preds = []
    t_start = time.time()

    for k in folds_to_run:
        origin_day = initial_origin + k * step_size
        t0 = time.time()
        fold_preds = run_single_chronos_fold(
            daily_df=daily_df,
            fold=k,
            origin_day=origin_day,
            model=model,
            horizon=horizon,
        )
        elapsed = time.time() - t0
        print(f"  Fold {k:2d} (origin d_{origin_day}): {len(fold_preds)} dong | Thoi gian: {elapsed:.1f}s")
        all_fold_preds.append(fold_preds)

    total_time = time.time() - t_start
    print(f"Tong thoi gian chay {len(folds_to_run)} fold ({model_name}): {total_time:.1f}s")

    all_preds_df = pd.concat(all_fold_preds, ignore_index=True)

    # Neu chay du 13 fold thi luu ket qua
    if len(folds_to_run) == num_folds:
        Path(pred_dir).mkdir(parents=True, exist_ok=True)
        Path(report_dir).mkdir(parents=True, exist_ok=True)

        parquet_file = Path(pred_dir) / f"{file_prefix}_predictions.parquet"
        all_preds_df.to_parquet(parquet_file, index=False)
        print(f"Da luu {len(all_preds_df):,} dong du bao vao {parquet_file}")

        # Tinh toan metrics
        overall_df, by_fold_df, by_cat_df, by_tier_df = evaluate_predictions(
            preds_df=all_preds_df,
            daily_df=daily_df,
            zero_tiers=zero_tiers,
        )

        overall_df.to_csv(Path(report_dir) / f"{file_prefix}_overall.csv", index=False)
        by_fold_df.to_csv(Path(report_dir) / f"{file_prefix}_by_fold.csv", index=False)
        by_cat_df.to_csv(Path(report_dir) / f"{file_prefix}_by_category.csv", index=False)
        by_tier_df.to_csv(Path(report_dir) / f"{file_prefix}_by_zero_tier.csv", index=False)
        print(f"Da luu cac file metrics cua {model_name} vao {report_dir}/")

    return all_preds_df, total_time


def main():
    parser = argparse.ArgumentParser(description="Chay backtest Chronos trong .venv-tsfm")
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["all"], default="all",
                        help="Bien the can chay (chronos2_history, chronos2_covariates, chronos_bolt, hoac all)")
    parser.add_argument("--test-fold", type=int, default=None,
                        help="Chay thu 1 fold (vi du: --test-fold 0) de do thoi gian truoc")
    args = parser.parse_args()

    backtest_cfg = load_backtest_config("configs/backtest.yaml")
    horizon = backtest_cfg["horizon"]
    num_folds = backtest_cfg["num_folds"]
    initial_origin = backtest_cfg["initial_origin"]
    step_size = backtest_cfg["step_size"]

    daily_df = load_processed_daily("data/processed/sales_daily.parquet")
    daily_df = _ensure_d_num(daily_df)
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial_origin)

    variants_to_run = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    if args.test_fold is not None:
        print(f"=== CHAY THU 1 FOLD (Fold {args.test_fold}) DE DO THOI GIAN ===")
        timing_summary = {}
        for var_key in variants_to_run:
            _, elapsed = run_variant_backtest(
                variant_key=var_key,
                daily_df=daily_df,
                zero_tiers=zero_tiers,
                initial_origin=initial_origin,
                num_folds=num_folds,
                step_size=step_size,
                horizon=horizon,
                folds_to_run=[args.test_fold],
            )
            timing_summary[var_key] = elapsed
        print("\n=== KET QUA DO THOI GIAN 1 FOLD ===")
        for k, v in timing_summary.items():
            print(f"  {k}: {v:.1f}s / fold -> Uoc tinh 13 fold: {v * 13 / 60:.1f} phut")
        return

    # Chay toan bo 13 fold cho cac bien the
    print(f"=== CHAY DU 13 FOLDS CHO CAC BIEN THE: {variants_to_run} ===")
    total_timings = {}
    for var_key in variants_to_run:
        _, elapsed = run_variant_backtest(
            variant_key=var_key,
            daily_df=daily_df,
            zero_tiers=zero_tiers,
            initial_origin=initial_origin,
            num_folds=num_folds,
            step_size=step_size,
            horizon=horizon,
            folds_to_run=list(range(num_folds)),
        )
        total_timings[var_key] = elapsed

    print("\n=== TONG KET THOI GIAN CHAY 13 FOLDS ===")
    for k, v in total_timings.items():
        print(f"  {k}: {v:.1f}s ({v / 60:.2f} phut)")


if __name__ == "__main__":
    main()
