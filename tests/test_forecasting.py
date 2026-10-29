"""Kiem thu cho module du bao va khung backtest (Phase 3a).

Kiem tra:
1. Metrics so voi gia tri tinh tay tren vi du nho (MAE, RMSE, WAPE, Bias, MASE).
2. Kiem tra ngoai le cua MASE khi scale = 0 va WAPE khi actual sum = 0.
3. Kiem tra kich thuoc du bao cua moi fold: dung 60 SKU x 28 ngay = 1.680 dong.
4. TEST CHONG RO RI (LEAK TEST): sua toan bo du lieu sau origin (nhan 1000)
   thi du bao cua fold do tuyet doi KHONG doi.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_processed_daily
from src.forecasting.base import ForecastModel
from src.forecasting.baselines import MovingAverageModel, NaiveModel, SeasonalNaiveModel
from src.forecasting.backtest import load_backtest_config, run_single_fold
from src.forecasting.metrics import (
    calc_bias,
    calc_mae,
    calc_mase,
    calc_metrics_dict,
    calc_rmse,
    calc_wape,
)


def test_metrics_hand_calculated():
    """Kiem tra cac ham metrics tren tap so lieu nho tinh tay."""
    y_true = np.array([2, 0, 4, 1, 3], dtype=float)
    y_pred = np.array([3, 1, 2, 1, 0], dtype=float)
    # Sai so: [1, 1, -2, 0, -3] -> tri tuyet doi: [1, 1, 2, 0, 3] (tong = 7)
    
    # 1. MAE = 7 / 5 = 1.4
    assert pytest.approx(calc_mae(y_true, y_pred), 1e-6) == 1.4
    
    # 2. RMSE = sqrt((1 + 1 + 4 + 0 + 9) / 5) = sqrt(3) ~= 1.7320508
    assert pytest.approx(calc_rmse(y_true, y_pred), 1e-6) == np.sqrt(3.0)
    
    # 3. WAPE = 7 / (2 + 0 + 4 + 1 + 3) = 7 / 10 = 0.70
    assert pytest.approx(calc_wape(y_true, y_pred), 1e-6) == 0.70
    
    # 4. Bias = mean([1, 1, -2, 0, -3]) = -3 / 5 = -0.60
    assert pytest.approx(calc_bias(y_true, y_pred), 1e-6) == -0.60
    
    # 5. MASE: voi y_train = [1, 3, 2, 5]
    # diffs = [2, 1, 3] -> scale = (2 + 1 + 3) / 3 = 2.0
    # MASE = MAE / scale = 1.4 / 2.0 = 0.70
    y_train = np.array([1, 3, 2, 5], dtype=float)
    assert pytest.approx(calc_mase(y_true, y_pred, y_train), 1e-6) == 0.70


def test_metrics_edge_cases():
    """Kiem tra cac truong hop bien: scale = 0 va tong actual = 0."""
    # Scale = 0 (chuoi train la hang so) -> phai bao loi theo dung quy dinh
    y_train_const = np.array([5, 5, 5, 5], dtype=float)
    with pytest.raises(ValueError, match="MASE scale bang 0"):
        calc_mase([2, 3], [2, 4], y_train_const)
        
    # Tong actual = 0 va co sai so -> bao loi WAPE
    with pytest.raises(ValueError, match="WAPE khong xac dinh"):
        calc_wape([0, 0, 0], [1, 0, 0])
        
    # Tong actual = 0 va du bao dung 0 -> WAPE = 0.0
    assert calc_wape([0, 0, 0], [0, 0, 0]) == 0.0


@pytest.fixture(scope="module")
def daily_data():
    """Doc du lieu daily da tien xu ly."""
    daily_path = Path("data/processed/sales_daily.parquet")
    if not daily_path.exists():
        pytest.skip("Chua co file sales_daily.parquet")
    return load_processed_daily(daily_path)


def test_each_fold_row_count(daily_data):
    """Kiem tra moi fold co dung 60 SKU x 28 ngay = 1.680 dong du bao."""
    config = load_backtest_config("configs/backtest.yaml")
    horizon = config["horizon"]
    origin_day = config["initial_origin"]  # fold 0: 1577
    
    models = [
        NaiveModel(name="Naive"),
        SeasonalNaiveModel(name="SeasonalNaive", season_length=7),
        MovingAverageModel(name="MovingAverage", window=28),
    ]
    
    for model in models:
        fold_res = run_single_fold(
            model=model,
            daily_df=daily_data,
            fold=0,
            origin_day=origin_day,
            horizon=horizon,
        )
        assert len(fold_res) == 60 * 28, (
            f"Model {model.name}: ky vong {60 * 28} dong, thuc te {len(fold_res)}"
        )
        assert fold_res["item_id"].nunique() == 60
        assert fold_res["forecast"].isna().sum() == 0
        assert (fold_res["forecast"] >= 0).all()


def test_leak_resistance_data_after_origin_modified(daily_data):
    """TEST CHONG RO RI: Sua moi du lieu sau origin thi du bao KHONG doi."""
    origin_day = 1577
    horizon = 28
    
    # 1. Chay du bao chuan tren du lieu goc cho ca 3 mo hinh
    models = [
        NaiveModel(name="Naive"),
        SeasonalNaiveModel(name="SeasonalNaive", season_length=7),
        MovingAverageModel(name="MovingAverage", window=28),
    ]
    
    # 2. Tao ban sao du lieu va SUA TOAN BO DU LIEU SAU ORIGIN (d_1578 tro di)
    # bang cach nhan sales voi 1000 va cong them 888
    corrupted_data = daily_data.copy()
    future_mask = corrupted_data["d"].apply(lambda d_str: int(d_str.split("_")[1]) > origin_day)
    corrupted_data.loc[future_mask, "sales"] = corrupted_data.loc[future_mask, "sales"] * 1000 + 888
    
    for model in models:
        # Du bao tren du lieu goc
        clean_pred = run_single_fold(
            model=model,
            daily_df=daily_data,
            fold=0,
            origin_day=origin_day,
            horizon=horizon,
        )
        
        # Du bao tren du lieu bi sua sau origin
        corrupted_pred = run_single_fold(
            model=model,
            daily_df=corrupted_data,
            fold=0,
            origin_day=origin_day,
            horizon=horizon,
        )
        
        # Du bao cua mo hinh phai giong nhau 100% tuyet doi
        pd.testing.assert_series_equal(
            clean_pred["forecast"],
            corrupted_pred["forecast"],
            check_exact=True,
            check_names=False,
            obj=f"Du bao cua model {model.name} bi anh huong boi du lieu sau origin (RO RI)",
        )
