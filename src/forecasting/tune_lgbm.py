"""Tinh chinh sieu tham so LightGBM TREN DU LIEU TRUOC D_1577.

2 fold noi bo:
  - Fold A: origin = 1521, du bao d_1522..d_1549
  - Fold B: origin = 1549, du bao d_1550..d_1577

Luoi 8 cau hinh (2 x 2 x 2):
  objective: ['poisson', 'tweedie'] (tweedie_variance_power=1.5)
  num_leaves: [15, 31]
  learning_rate: [0.03, 0.05]

Cac tham so khac co dinh: n_estimators=500, min_child_samples=20, seed=42.
Tieu chi chon: WAPE trung binh 2 fold (nho nhat).

TOI UU TOC DO:
- Dung d_num (int) de filter thay vi string set -> nhanh ~10x.
- Moi fold chi dung 20 train origins gan nhat (khoang 20*7=140 ngay) thay vi
  toan bo lich su. Ly do: LightGBM hoc pattern, khong can toan bo 200+ origins;
  cac origin xa qua co the gay noise (data drift). Quy tac: origin range
  [max_origin - 20*7, max_origin], step=7.
  -> Voi fold A (origin=1521): origins tu 1381 den 1521, step=7 (21 origins)
  -> Voi fold B (origin=1549): origins tu 1409 den 1549, step=7 (21 origins)

TUYET DOI KHONG dung bat ky du lieu sau d_1577.
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from itertools import product
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml

from src.data.loader import load_processed_daily
from src.forecasting.features import build_train_dataset, build_features_for_origin, FEATURE_COLS, _ensure_d_num
from src.forecasting.metrics import calc_wape
import lightgbm as lgb


# 2 fold noi bo (chi dung d_1..d_1577)
# 1577 - 28 = 1549, 1549 - 28 = 1521
# Kiem tra: 1577 mod 7 == 1549 mod 7 == 1521 mod 7 == 6 -> cung thu
TUNE_FOLDS = [
    {"origin": 1521, "val_start": 1522, "val_end": 1549},
    {"origin": 1549, "val_start": 1550, "val_end": 1577},
]

# So origins toi da de build tap train (tinh tu origin tro ve)
# 20 origins x 7 ngay = 140 ngay lich su. Du de model hoc seasonality tuan.
MAX_TRAIN_ORIGINS = 20

PARAM_GRID = []
for obj, nl, lr in product(
    ["poisson", "tweedie"],
    [15, 31],
    [0.03, 0.05],
):
    cfg: Dict = {
        "objective": obj,
        "num_leaves": nl,
        "learning_rate": lr,
        "min_child_samples": 20,
        "n_estimators": 500,
        "seed": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if obj == "tweedie":
        cfg["tweedie_variance_power"] = 1.5  # muc do gian doan trung binh
    PARAM_GRID.append(cfg)

assert len(PARAM_GRID) == 8, f"Luoi phai co 8 cau hinh, thuc te {len(PARAM_GRID)}"


def _make_train_origins(origin: int, step: int = 7, max_origins: int = MAX_TRAIN_ORIGINS, min_hist: int = 56) -> List[int]:
    """Tao danh sach train origins: cach nhau step ngay, bat dau tu origin tro ve.

    Chi lay toi da max_origins origins gan nhat de giam thoi gian build features.
    Dam bao moi origin co it nhat min_hist ngay lich su.
    """
    origins = list(range(min_hist, origin + 1, step))
    if not origins or origins[-1] != origin:
        origins.append(origin)
    # Chi lay max_origins origins gan nhat
    return origins[-max_origins:]


def _run_tune_fold(
    daily_df: pd.DataFrame,
    fold_cfg: Dict,
    lgbm_params: Dict,
    horizon: int = 28,
) -> float:
    """Chay 1 fold noi bo, tra ve WAPE tren tap val.

    Dung d_num (int) de filter nhanh thay vi string set.
    """
    origin = fold_cfg["origin"]
    val_start = fold_cfg["val_start"]
    val_end = fold_cfg["val_end"]

    # Lay cac train origins (20 origins gan nhat)
    train_origins = _make_train_origins(origin)

    # Build tap train (chi dung d_num <= origin)
    X_train, y_train = build_train_dataset(
        daily_df=daily_df,
        train_origins=[o for o in train_origins if o < origin],  # strict < tranh overlap
        horizon=horizon,
    )
    valid_mask = ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = np.maximum(y_train[valid_mask], 0.0)

    # Build val set: feature tai origin, target = val period
    history_df = daily_df[daily_df["d_num"] <= origin]
    val_df = daily_df[(daily_df["d_num"] >= val_start) & (daily_df["d_num"] <= val_end)].copy()

    X_val_list = []
    y_val_list = []
    for item_id, item_val in val_df.groupby("item_id"):
        item_val = item_val.sort_values("date").reset_index(drop=True)
        item_hist = history_df[history_df["item_id"] == item_id]
        X_v = build_features_for_origin(
            history_df=item_hist,
            future_context_df=item_val[["date", "item_id", "sell_price",
                                         "wday", "month", "event_name_1", "snap_CA", "sales"]],
            origin_day=origin,
            horizon=horizon,
        )
        item_val_s = item_val.sort_values("date").reset_index(drop=True)
        item_val_s["horizon_h"] = range(1, len(item_val_s) + 1)
        y_v = X_v[["item_id", "horizon_h"]].merge(
            item_val_s[["item_id", "horizon_h", "sales"]],
            on=["item_id", "horizon_h"], how="left"
        )["sales"].values.astype(float)
        X_val_list.append(X_v)
        y_val_list.append(y_v)

    X_val_df = pd.concat(X_val_list, ignore_index=True)
    y_val_arr = np.concatenate(y_val_list)
    valid_val_mask = ~np.isnan(y_val_arr)
    X_val_df = X_val_df[valid_val_mask]
    y_val_arr = np.maximum(y_val_arr[valid_val_mask], 0.0)

    if len(X_train) == 0 or len(X_val_df) == 0:
        print("  [CANH BAO] Tap train hoac val trong, bo qua fold nay.")
        return 999.0

    # Train LightGBM
    params = dict(lgbm_params)
    num_boost_round = int(params.pop("n_estimators", 500))

    lgb_train = lgb.Dataset(X_train[FEATURE_COLS], label=y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val_df[FEATURE_COLS], label=y_val_arr,
                          reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = np.maximum(model.predict(X_val_df[FEATURE_COLS]), 0.0)
    wape = calc_wape(y_val_arr, preds)
    return wape


def run_tuning(
    daily_df: pd.DataFrame,
    output_config_path: str = "configs/lightgbm.yaml",
    horizon: int = 28,
) -> Dict:
    """Chay luoi tim kiem va luu tham so tot nhat vao configs/lightgbm.yaml.

    Toi uu: them d_num truoc de filter nhanh.
    """
    # Them d_num 1 lan duy nhat
    daily_df = _ensure_d_num(daily_df)

    # Loc chi giai doan phe duyet: d_1..d_1577 (TUYET DOI)
    last_allowed = 1577
    tune_df = daily_df[daily_df["d_num"] <= last_allowed].copy()

    print(f"=== BAT DAU TUNING: {len(PARAM_GRID)} cau hinh x {len(TUNE_FOLDS)} fold ===")
    print(f"Du lieu cho phep: d_1..d_{last_allowed}. TUYET DOI khong dung sau d_{last_allowed}.")
    print(f"So train origins moi fold: toi da {MAX_TRAIN_ORIGINS} (step=7 ngay, lay gan nhat)")
    for fold in TUNE_FOLDS:
        origs = _make_train_origins(fold["origin"])
        print(f"  Fold origin={fold['origin']}: {len(origs)} train origins [{origs[0]}..{origs[-1]}]")

    results = []
    for i, params in enumerate(PARAM_GRID):
        fold_wapes = []
        t0 = time.time()
        for fold_cfg in TUNE_FOLDS:
            wape = _run_tune_fold(tune_df, fold_cfg, params, horizon)
            fold_wapes.append(wape)
        elapsed = time.time() - t0
        mean_wape = float(np.mean(fold_wapes))
        obj_label = params["objective"]
        if obj_label == "tweedie":
            obj_label += f"(p={params.get('tweedie_variance_power', '?')})"
        print(
            f"  [{i+1:2d}/{len(PARAM_GRID)}] obj={obj_label}, "
            f"num_leaves={params['num_leaves']}, lr={params['learning_rate']:.2f} "
            f"-> WAPE fold_A={fold_wapes[0]:.4f}, fold_B={fold_wapes[1]:.4f}, "
            f"mean={mean_wape:.4f} ({elapsed:.1f}s)"
        )
        results.append({"params": params, "mean_wape": mean_wape, "fold_wapes": fold_wapes})

    # Chon cau hinh tot nhat
    best = min(results, key=lambda r: r["mean_wape"])
    best_params = best["params"].copy()
    best_wape = best["mean_wape"]

    print(f"\n=== CAU HINH TOT NHAT (WAPE trung binh 2 fold = {best_wape:.4f}) ===")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Luu vao configs/lightgbm.yaml
    save_params = dict(best_params)
    save_params["_tuning_mean_wape"] = round(best_wape, 6)
    save_params["_tuning_fold_A_wape"] = round(best["fold_wapes"][0], 6)
    save_params["_tuning_fold_B_wape"] = round(best["fold_wapes"][1], 6)
    save_params["_note"] = (
        "Tuned tren d_1..d_1577 voi 2 fold noi bo (origin 1521 va 1549). "
        "TUYET DOI khong dung cac fold hold-out de chon tham so. "
        f"Moi fold dung {MAX_TRAIN_ORIGINS} train origins gan nhat (step=7)."
    )

    with open(output_config_path, "w", encoding="utf-8") as f:
        yaml.dump(save_params, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\nDa luu cau hinh vao {output_config_path}")
    return best_params


if __name__ == "__main__":
    daily = load_processed_daily()
    best = run_tuning(daily)
