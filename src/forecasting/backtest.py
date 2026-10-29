"""Khung thuc thi backtest rolling-origin expanding window tren tap hold-out.

Quy tac bat buoc:
1. 13 fold tren hold-out: origin o_k = 1577 + 28 * k, k = 0..12.
2. Fold k chi duoc fit() tren d_1..d_{o_k}, va predict() tren d_{o_k+1}..d_{o_k+28}.
3. Tinh toan MAE, RMSE, WAPE, Bias va MASE (scale tu train history den origin).
4. Luu ket qua du bao vao results/predictions/ va bao cao vao results/forecasting/.
"""

import sys
from pathlib import Path

# Dam bao import duoc src
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_processed_daily
from src.forecasting.base import ForecastModel
from src.forecasting.baselines import MovingAverageModel, NaiveModel, SeasonalNaiveModel
from src.forecasting.metrics import calc_bias, calc_mae, calc_mase, calc_rmse, calc_wape


def load_backtest_config(config_path: Union[str, Path] = "configs/backtest.yaml") -> dict:
    """Doc file cau hinh backtest."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_sku_zero_tiers(daily_df: pd.DataFrame, last_train_day: int = 1577) -> Dict[str, str]:
    """Phan loai 60 SKU thanh 3 nhom dua tren ty le ngay ban bang 0 trong d_1..d_1577.
    
    - smooth: zero_ratio < 0.33
    - medium: 0.33 <= zero_ratio <= 0.66
    - intermittent: zero_ratio > 0.66
    """
    train_d_cols = set(f"d_{i}" for i in range(1, last_train_day + 1))
    train_data = daily_df[daily_df["d"].isin(train_d_cols)]
    
    zero_ratios = train_data.groupby("item_id")["sales"].apply(lambda s: float((s == 0).mean()))
    
    tiers = {}
    for item, zr in zero_ratios.items():
        if zr < 0.33:
            tiers[item] = "smooth"
        elif zr <= 0.66:
            tiers[item] = "medium"
        else:
            tiers[item] = "intermittent"
    return tiers


def run_single_fold(
    model: ForecastModel,
    daily_df: pd.DataFrame,
    fold: int,
    origin_day: int,
    horizon: int = 28,
) -> pd.DataFrame:
    """Chay 1 fold cu the cho mot mo hinh.
    
    origin_day: Chi so ngay origin (vi du: 1577, 1605, ...)
    """
    # 1. Du lieu huan luyen: d_1 den d_{origin_day}
    train_d_set = set(f"d_{i}" for i in range(1, origin_day + 1))
    history_df = daily_df[daily_df["d"].isin(train_d_set)].copy()
    
    # 2. Du lieu kiem thu: d_{origin_day + 1} den d_{origin_day + horizon}
    test_d_set = set(f"d_{i}" for i in range(origin_day + 1, origin_day + horizon + 1))
    test_df = daily_df[daily_df["d"].isin(test_d_set)].copy()
    
    # 3. Fit mo hinh chi tren history
    model.fit(history_df)
    
    # 4. Predict tren horizon
    future_features = test_df[["date", "item_id"]].drop_duplicates().copy()
    pred_df = model.predict(horizon=horizon, future_df=future_features)
    
    # 5. Ghep voi actual sales va metadata
    merged = test_df[["date", "item_id", "cat_id", "dept_id", "sales"]].merge(
        pred_df,
        on=["date", "item_id"],
        how="inner",
    )
    merged = merged.rename(columns={"sales": "actual"})
    merged["fold"] = fold
    merged["origin_day"] = origin_day
    merged["model"] = model.name
    return merged


def evaluate_predictions(
    preds_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    zero_tiers: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tinh toan cac bang ket qua danh gia: overall, by_fold, by_category, by_zero_tier."""
    
    # Tinh truoc in-sample naive scale cho tung SKU va tung origin_day
    # de dung cho chi so MASE
    scales: Dict[Tuple[int, str], float] = {}
    origins = preds_df["origin_day"].unique()
    
    for orig in origins:
        train_d_set = set(f"d_{i}" for i in range(1, orig + 1))
        hist_orig = daily_df[daily_df["d"].isin(train_d_set)]
        for item, group in hist_orig.groupby("item_id"):
            s = group.sort_values(by="date")["sales"].values.astype(float)
            diffs = np.abs(np.diff(s))
            scale_val = float(np.mean(diffs))
            if scale_val == 0.0:
                raise ValueError(f"SKU {item} tai origin {orig} co scale = 0!")
            scales[(orig, item)] = scale_val

    # Gan them zero_tier
    eval_df = preds_df.copy()
    eval_df["zero_tier"] = eval_df["item_id"].map(zero_tiers)
    
    # Helper tinh metrics cho 1 nhom du doan
    def compute_group_metrics_dict(group: pd.DataFrame) -> dict:
        yt = group["actual"].values
        yp = group["forecast"].values
        
        mae = calc_mae(yt, yp)
        rmse = calc_rmse(yt, yp)
        wape = calc_wape(yt, yp)
        bias = calc_bias(yt, yp)
        
        # MASE: tinh theo tung (origin, item) roi lay trung binh
        mase_list = []
        for (orig, item), sub_grp in group.groupby(["origin_day", "item_id"]):
            sub_mae = calc_mae(sub_grp["actual"].values, sub_grp["forecast"].values)
            sc = scales[(orig, item)]
            mase_list.append(sub_mae / sc)
        mase = float(np.mean(mase_list))
        
        return {
            "WAPE": round(wape, 4),
            "MASE": round(mase, 4),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Bias": round(bias, 4),
            "Count": int(len(group)),
        }

    # 1. Overall theo Model
    overall_list = []
    for model_name, m_group in eval_df.groupby("model"):
        met = compute_group_metrics_dict(m_group)
        met["model"] = model_name
        overall_list.append(met)
    overall_df = pd.DataFrame(overall_list)[["model", "WAPE", "MASE", "MAE", "RMSE", "Bias", "Count"]]

    # 2. Theo Fold
    by_fold_list = []
    for (model_name, fold_id), f_group in eval_df.groupby(["model", "fold"]):
        met = compute_group_metrics_dict(f_group)
        met["model"] = model_name
        met["fold"] = fold_id
        by_fold_list.append(met)
    by_fold_df = pd.DataFrame(by_fold_list)[["model", "fold", "WAPE", "MASE", "MAE", "RMSE", "Bias", "Count"]]

    # 3. Theo Category
    by_cat_list = []
    for (model_name, cat), c_group in eval_df.groupby(["model", "cat_id"]):
        met = compute_group_metrics_dict(c_group)
        met["model"] = model_name
        met["cat_id"] = cat
        by_cat_list.append(met)
    by_cat_df = pd.DataFrame(by_cat_list)[["model", "cat_id", "WAPE", "MASE", "MAE", "RMSE", "Bias", "Count"]]

    # 4. Theo Zero Demand Tier
    by_tier_list = []
    for (model_name, tier), t_group in eval_df.groupby(["model", "zero_tier"]):
        met = compute_group_metrics_dict(t_group)
        met["model"] = model_name
        met["zero_tier"] = tier
        by_tier_list.append(met)
    by_tier_df = pd.DataFrame(by_tier_list)[["model", "zero_tier", "WAPE", "MASE", "MAE", "RMSE", "Bias", "Count"]]

    return overall_df, by_fold_df, by_cat_df, by_tier_df


def run_all_baselines(
    daily_data_path: Union[str, Path] = "data/processed/sales_daily.parquet",
    config_path: Union[str, Path] = "configs/backtest.yaml",
    pred_dir: Union[str, Path] = "results/predictions",
    report_dir: Union[str, Path] = "results/forecasting",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chay toan bo quy trinh backtest cho 3 mo hinh baseline va xuat ket qua."""
    config = load_backtest_config(config_path)
    horizon = config["horizon"]
    num_folds = config["num_folds"]
    initial_origin = config["initial_origin"]
    step_size = config["step_size"]

    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    daily_df = load_processed_daily(daily_data_path)
    zero_tiers = get_sku_zero_tiers(daily_df, last_train_day=initial_origin)

    models: List[ForecastModel] = [
        NaiveModel(name="Naive"),
        SeasonalNaiveModel(name="SeasonalNaive", season_length=7),
        MovingAverageModel(name="MovingAverage", window=28),
    ]

    all_preds_list = []

    print(f"=== BAT DAU BACKTEST: {num_folds} folds x {len(models)} models x 60 SKUs ===")
    for model in models:
        print(f"Dang chay mo hinh: {model.name}...")
        model_fold_preds = []
        for k in range(num_folds):
            origin_day = initial_origin + k * step_size
            fold_pred = run_single_fold(
                model=model,
                daily_df=daily_df,
                fold=k,
                origin_day=origin_day,
                horizon=horizon,
            )
            model_fold_preds.append(fold_pred)

        model_all_preds = pd.concat(model_fold_preds, ignore_index=True)
        # Luu du bao cua model vao results/predictions/
        pred_file = Path(pred_dir) / f"{model.name.lower()}_predictions.parquet"
        model_all_preds.to_parquet(pred_file, index=False)
        print(f"  -> Da luu {len(model_all_preds):,} dong du bao vao {pred_file}")
        all_preds_list.append(model_all_preds)

    total_preds_df = pd.concat(all_preds_list, ignore_index=True)

    print("\nDang tinh toan cac bang chi so danh gia...")
    overall_df, by_fold_df, by_cat_df, by_tier_df = evaluate_predictions(
        preds_df=total_preds_df,
        daily_df=daily_df,
        zero_tiers=zero_tiers,
    )

    # Luu cac bang ket qua vao results/forecasting/
    overall_df.to_csv(Path(report_dir) / "baseline_overall.csv", index=False)
    by_fold_df.to_csv(Path(report_dir) / "baseline_by_fold.csv", index=False)
    by_cat_df.to_csv(Path(report_dir) / "baseline_by_category.csv", index=False)
    by_tier_df.to_csv(Path(report_dir) / "baseline_by_zero_tier.csv", index=False)

    print(f"Da xuat toan bo bao cao vao {report_dir}/")
    return overall_df, by_fold_df, by_cat_df, by_tier_df


if __name__ == "__main__":
    overall, by_fold, by_cat, by_tier = run_all_baselines()
    print("\n--- BANG KET QUA TONG THE (OVERALL) ---")
    print(overall.to_string(index=False))
    print("\n--- BANG KET QUA THEO NGANH HANG (BY CATEGORY) ---")
    print(by_cat.to_string(index=False))
    print("\n--- BANG KET QUA THEO NHOM TY LE ZERO (BY TIER) ---")
    print(by_tier.to_string(index=False))
