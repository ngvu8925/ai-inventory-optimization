"""Run selected Chronos variants on the established rolling-origin folds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["PIP_CACHE_DIR"] = r"D:\pip_cache"
os.environ["TEMP"] = r"D:\pip_cache"
os.environ["TMP"] = r"D:\pip_cache"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_processed_daily
from src.forecasting.backtest import evaluate_predictions, get_sku_zero_tiers
from src.forecasting.chronos_model import ChronosModel, MODEL_NAMES
from src.forecasting.metrics import calc_bias, calc_pinball_loss

VARIANTS = ("bolt_history", "chronos2_history", "chronos2_exogenous")
FILE_PREFIX = {
    "bolt_history": "chronos_bolt_small_history",
    "chronos2_history": "chronos2_history",
    "chronos2_exogenous": "chronos2_exogenous",
}
RAM_LIMIT_GB = 4.0


def available_ram_gb() -> float:
    """Return system available RAM in GiB."""
    import psutil

    return psutil.virtual_memory().available / (1024**3)


def require_ram(variant: str) -> float:
    """Check the user's RAM cutoff immediately before loading/inference."""
    free_gb = available_ram_gb()
    print(f"RAM available before {variant}: {free_gb:.2f} GB")
    if free_gb < RAM_LIMIT_GB:
        raise RuntimeError(
            f"Available RAM below {RAM_LIMIT_GB:.1f} GB before {variant}; stopping."
        )
    return free_gb


def load_config(path: str = "configs/backtest.yaml") -> dict:
    """Read the fixed rolling-origin backtest configuration."""
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _future_context(test_df: pd.DataFrame) -> pd.DataFrame:
    """Select known-at-origin covariates without including future sales."""
    columns = [
        "date", "item_id", "sell_price", "wday", "month",
        "event_name_1", "snap_CA",
    ]
    return test_df[columns].copy()


def run_one_fold(
    daily_df: pd.DataFrame,
    variant: str,
    fold: int,
    origin_day: int,
    horizon: int,
    seed: int,
    model_id: Optional[str] = None,
) -> tuple[pd.DataFrame, float]:
    """Forecast one fold using history through origin only."""
    from src.forecasting.features import _ensure_d_num

    daily_df = _ensure_d_num(daily_df)
    history = daily_df[daily_df["d_num"] <= origin_day].copy()
    test_df = daily_df[
        (daily_df["d_num"] > origin_day)
        & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()
    future = test_df[["date", "item_id"]].drop_duplicates()
    context = _future_context(test_df) if variant == "chronos2_exogenous" else None

    model = ChronosModel(variant=variant, seed=seed, model_id=model_id)
    model.fit(history)
    require_ram(MODEL_NAMES[variant])
    started = time.perf_counter()
    forecast_df = model.predict(
        horizon=horizon,
        future_df=future,
        future_context_df=context,
    )
    elapsed = time.perf_counter() - started

    merged = test_df[
        ["date", "item_id", "cat_id", "dept_id", "sales"]
    ].merge(forecast_df, on=["date", "item_id"], how="inner", validate="one_to_one")
    merged = merged.rename(columns={"sales": "actual"})
    merged["fold"] = fold
    merged["origin_day"] = origin_day
    merged["model"] = MODEL_NAMES[variant]
    merged["fold_elapsed_seconds"] = elapsed
    if len(merged) != 60 * horizon:
        raise RuntimeError(f"Expected {60*horizon} rows, received {len(merged)}")
    if merged["forecast"].isna().any() or (merged["forecast"] < 0).any():
        raise RuntimeError("Forecast contains NaN or negative values")
    return merged, elapsed


def pinball_table(predictions: pd.DataFrame, by_fold: bool = False) -> pd.DataFrame:
    """Calculate quantile losses separately from point-forecast comparisons."""
    group_cols = ["model", "fold"] if by_fold else ["model"]
    rows = []
    for keys, group in predictions.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["Count"] = len(group)
        for q in (0.1, 0.5, 0.9, 0.95):
            column = f"forecast_q{int(q * 100):02d}"
            if column in group:
                row[f"pinball_q{q:.2f}"] = calc_pinball_loss(
                    group["actual"].to_numpy(), group[column].to_numpy(), q
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_full_results(
    predictions: pd.DataFrame,
    daily_df: pd.DataFrame,
    prefix: str,
    pred_dir: Path,
    report_dir: Path,
) -> None:
    """Write normal backtest tables plus separate quantile losses."""
    config = load_config()
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=config["initial_origin"])
    pred_path = pred_dir / f"{prefix}_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)
    overall, by_fold, by_cat, by_tier = evaluate_predictions(
        preds_df=predictions,
        daily_df=daily_df,
        zero_tiers=zero_tiers,
    )
    overall.to_csv(report_dir / f"{prefix}_overall.csv", index=False)
    by_fold.to_csv(report_dir / f"{prefix}_by_fold.csv", index=False)
    by_cat.to_csv(report_dir / f"{prefix}_by_category.csv", index=False)
    by_tier.to_csv(report_dir / f"{prefix}_by_zero_tier.csv", index=False)
    pinball_table(predictions).to_csv(
        report_dir / f"{prefix}_pinball_overall.csv", index=False
    )
    pinball_table(predictions, by_fold=True).to_csv(
        report_dir / f"{prefix}_pinball_by_fold.csv", index=False
    )

    required = [
        "baseline_by_fold.csv",
        "lightgbm_by_fold.csv",
        *(f"{FILE_PREFIX[v]}_by_fold.csv" for v in VARIANTS),
    ]
    if all((report_dir / filename).exists() for filename in required):
        # Comparison uses only complete 13-fold point metrics, never the old T5 file.
        from src.forecasting.run_chronos_backtest import update_model_comparisons

        update_model_comparisons(report_dir)


def run_variant(
    variant: str,
    fold: Optional[int] = 0,
    all_folds: bool = False,
    model_id: Optional[str] = None,
) -> pd.DataFrame:
    """Run a pilot fold by default; full holdout requires --all-folds."""
    config = load_config()
    daily_df = load_processed_daily()
    initial = int(config["initial_origin"])
    step = int(config["step_size"])
    horizon = int(config["horizon"])
    num_folds = int(config["num_folds"])
    folds = range(num_folds) if all_folds else [int(fold)]
    if any(k < 0 or k >= num_folds for k in folds):
        raise ValueError(f"Fold must be in 0..{num_folds - 1}")

    parts = []
    total_started = time.perf_counter()
    for k in folds:
        origin = initial + k * step
        print(f"=== {MODEL_NAMES[variant]} | fold {k} | origin d_{origin} ===")
        fold_df, elapsed = run_one_fold(
            daily_df, variant, k, origin, horizon, int(config["seed"]), model_id
        )
        print(f"fold_rows={len(fold_df)} inference_seconds={elapsed:.2f}")
        parts.append(fold_df)
    predictions = pd.concat(parts, ignore_index=True)
    total_elapsed = time.perf_counter() - total_started

    pred_dir = Path("results/predictions")
    report_dir = Path("results/forecasting")
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prefix = FILE_PREFIX[variant]
    if all_folds:
        _write_full_results(predictions, daily_df, prefix, pred_dir, report_dir)
        file_path = pred_dir / f"{prefix}_predictions.parquet"
    else:
        suffix = "pilot" if fold == 0 else f"fold_{fold:02d}"
        file_path = pred_dir / f"{prefix}_{suffix}.parquet"
        predictions.to_parquet(file_path, index=False)

    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial)
    diagnostics = predictions.copy()
    diagnostics["zero_tier"] = diagnostics["item_id"].map(zero_tiers)
    bias = calc_bias(diagnostics["actual"], diagnostics["forecast"])
    zero_rates = diagnostics.groupby("zero_tier")["forecast"].apply(
        lambda values: float((values == 0).mean())
    )
    summary = {
        "model": MODEL_NAMES[variant],
        "variant": variant,
        "folds": list(folds),
        "origins": [initial + k * step for k in folds],
        "row_count": len(predictions),
        "elapsed_seconds": total_elapsed,
        "bias": bias,
        "zero_forecast_rate_by_zero_tier": zero_rates.to_dict(),
        "all_forecasts_zero": bool((predictions["forecast"] == 0).all()),
        "prediction_file": str(file_path),
        "torch": None,
        "chronos_forecasting": None,
    }
    try:
        import torch
        import chronos

        summary["torch"] = torch.__version__
        summary["chronos_forecasting"] = chronos.__version__
    except ImportError:
        pass
    summary_path = report_dir / f"{prefix}_{'full' if all_folds else 'pilot'}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["all_forecasts_zero"]:
        raise RuntimeError("All point forecasts are zero; stop and investigate.")
    if all_folds:
        print(f"Wrote point metrics, pinball metrics, and predictions for {variant}.")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--all-folds", action="store_true",
        help="Run all 13 folds; opt-in only after explicit user approval.",
    )
    parser.add_argument(
        "--update-comparisons", action="store_true",
        help="Update tables in the main venv from completed CSVs; no Chronos import.",
    )
    parser.add_argument("--model-id", help="Optional test checkpoint override.")
    args = parser.parse_args()
    if args.update_comparisons:
        from src.forecasting.run_chronos_backtest import update_model_comparisons

        update_model_comparisons(Path("results/forecasting"))
        return
    if not args.variant:
        parser.error("--variant is required unless --update-comparisons is used")
    run_variant(args.variant, fold=args.fold, all_folds=args.all_folds, model_id=args.model_id)


if __name__ == "__main__":
    main()
