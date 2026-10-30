"""Kiem thu them cho Phase 3b: LightGBM feature engineering va leak test.

Tests:
(a) Feature tai origin t khong thay doi du du lieu sau t bi nhan 1000.
(b) Rolling feature chi dung du lieu <= t (kiem tra gia tri cu the).
(c) Moi fold LightGBM dung 60 x 28 = 1680 dong, forecast >= 0.
(d) Leak test: sua du lieu sau origin -> du bao fold do khong doi.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_processed_daily
from src.forecasting.features import build_features_for_origin, FEATURE_COLS


@pytest.fixture(scope="module")
def daily_data():
    path = Path("data/processed/sales_daily.parquet")
    if not path.exists():
        pytest.skip("Chua co sales_daily.parquet")
    return load_processed_daily(path)


@pytest.fixture(scope="module")
def small_history(daily_data):
    """Lay 90 ngay dau + context 28 ngay tiep theo de test nhanh."""
    origin = 90
    train_d = set(f"d_{i}" for i in range(1, origin + 1))
    ctx_d = set(f"d_{i}" for i in range(origin + 1, origin + 29))
    hist = daily_data[daily_data["d"].isin(train_d)].copy()
    ctx = daily_data[daily_data["d"].isin(ctx_d)][
        ["date", "item_id", "sell_price", "wday", "month", "event_name_1", "snap_CA"]
    ].copy()
    return hist, ctx, origin


def test_feature_not_affected_by_future_data(daily_data):
    """(a) Feature tai origin t phai giong nhau du du lieu sau t bi sua."""
    origin = 100
    train_d = set(f"d_{i}" for i in range(1, origin + 1))
    ctx_d = set(f"d_{i}" for i in range(origin + 1, origin + 29))

    hist_clean = daily_data[daily_data["d"].isin(train_d)].copy()
    ctx = daily_data[daily_data["d"].isin(ctx_d)][
        ["date", "item_id", "sell_price", "wday", "month", "event_name_1", "snap_CA"]
    ].copy()

    # Tao ban sao du lieu sau origin bi sua (nhan 1000)
    corrupted = daily_data.copy()
    future_mask = corrupted["d"].apply(lambda d: int(d.split("_")[1]) > origin)
    corrupted.loc[future_mask, "sales"] = corrupted.loc[future_mask, "sales"] * 1000 + 888
    hist_corrupted = corrupted[corrupted["d"].isin(train_d)].copy()

    # Build feature tren ca hai
    feat_clean = build_features_for_origin(hist_clean, ctx, origin_day=origin)
    feat_corrupted = build_features_for_origin(hist_corrupted, ctx, origin_day=origin)

    # Feature phai giong nhau hoan toan (du lieu sau origin khong anh huong)
    pd.testing.assert_frame_equal(
        feat_clean[FEATURE_COLS].reset_index(drop=True),
        feat_corrupted[FEATURE_COLS].reset_index(drop=True),
        check_exact=False,
        atol=1e-8,
        obj="Feature bi anh huong boi du lieu sau origin (RO RI)"
    )


def test_rolling_features_use_only_history_up_to_t(small_history):
    """(b) Rolling feature chi dung du lieu <= t, khong co thong tin tuong lai."""
    hist, ctx, origin = small_history

    feat = build_features_for_origin(hist, ctx, origin_day=origin)

    # Tat ca SKU phai co feature
    assert len(feat) == 60 * 28, f"Ky vong {60*28} hang, thuc te {len(feat)}"

    # Kiem tra roll_mean_7 cua moi SKU khop voi tinh tay
    for item_id, grp in hist.groupby("item_id"):
        grp_sorted = grp.sort_values("date")
        sales_arr = grp_sorted["sales"].values.astype(float)
        expected_mean_7 = float(np.mean(sales_arr[-7:])) if len(sales_arr) >= 7 else float(np.mean(sales_arr))

        item_feat = feat[feat["item_id"] == item_id]["roll_mean_7"].values
        # Tat ca 28 hang cua SKU nay phai co cung roll_mean_7 (tinh 1 lan tai origin)
        assert np.allclose(item_feat, expected_mean_7, atol=1e-6), (
            f"SKU {item_id}: roll_mean_7 = {item_feat[0]:.4f}, expected = {expected_mean_7:.4f}"
        )


def test_lgbm_fold_row_count_and_non_negative(daily_data):
    """(c) Moi fold LightGBM: dung 60 x 28 = 1680 dong, forecast >= 0."""
    lgbm_config_path = Path("configs/lightgbm.yaml")
    if not lgbm_config_path.exists():
        pytest.skip("Chua co configs/lightgbm.yaml (chay tune_lgbm.py truoc)")

    import yaml
    with open(lgbm_config_path) as f:
        raw = yaml.safe_load(f)
    lgbm_params = {k: v for k, v in raw.items() if not k.startswith("_")}

    from src.forecasting.run_lgbm_backtest import run_single_lgbm_fold

    origin_day = 1577  # fold 0
    fold_preds, _ = run_single_lgbm_fold(
        daily_df=daily_data,
        fold=0,
        origin_day=origin_day,
        lgbm_params=lgbm_params,
        horizon=28,
    )

    assert len(fold_preds) == 60 * 28, f"Ky vong 1680 dong, thuc te {len(fold_preds)}"
    assert fold_preds["item_id"].nunique() == 60
    assert (fold_preds["forecast"] >= 0).all(), "Co gia tri forecast am!"
    assert fold_preds["forecast"].isna().sum() == 0, "Co NaN trong forecast!"


def test_lgbm_leak_resistance(daily_data):
    """(d) Leak test: sua du lieu sau origin -> du bao LightGBM khong doi."""
    lgbm_config_path = Path("configs/lightgbm.yaml")
    if not lgbm_config_path.exists():
        pytest.skip("Chua co configs/lightgbm.yaml (chay tune_lgbm.py truoc)")

    import yaml
    with open(lgbm_config_path) as f:
        raw = yaml.safe_load(f)
    lgbm_params = {k: v for k, v in raw.items() if not k.startswith("_")}

    from src.forecasting.run_lgbm_backtest import run_single_lgbm_fold

    origin_day = 1577

    # Chay tren du lieu sach
    clean_preds, _ = run_single_lgbm_fold(
        daily_df=daily_data,
        fold=0,
        origin_day=origin_day,
        lgbm_params=lgbm_params,
        horizon=28,
    )

    # Tao du lieu bi pha (sau origin bi sua)
    corrupted = daily_data.copy()
    future_mask = corrupted["d"].apply(lambda d: int(d.split("_")[1]) > origin_day)
    corrupted.loc[future_mask, "sales"] = corrupted.loc[future_mask, "sales"] * 1000 + 888

    corrupted_preds, _ = run_single_lgbm_fold(
        daily_df=corrupted,
        fold=0,
        origin_day=origin_day,
        lgbm_params=lgbm_params,
        horizon=28,
    )

    pd.testing.assert_series_equal(
        clean_preds["forecast"].reset_index(drop=True),
        corrupted_preds["forecast"].reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
        obj="LightGBM: Du bao thay doi khi sua du lieu sau origin (RO RI!)"
    )
