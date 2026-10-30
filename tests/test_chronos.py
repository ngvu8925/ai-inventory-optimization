"""Unit tests cho ChronosModel zero-shot (Phase 4).

Kiem tra:
(a) Ke thua dung ForecastModel, co phuong thuc fit() va predict().
(b) predict() tra ve DataFrame dung 60 x 28 = 1680 dong, khong co gia tri am, khong NaN.
(c) Leak test: sua du lieu sau origin thi du bao khong thay doi.
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_processed_daily
from src.forecasting.base import ForecastModel
from src.forecasting.features import _ensure_d_num
from src.forecasting.tsfm_model import ChronosModel


@pytest.fixture(scope="module")
def daily_data():
    """Doc du lieu daily da tien xu ly."""
    return load_processed_daily()


def test_chronos_inheritance():
    """(a) ChronosModel phai ke thua ForecastModel."""
    model = ChronosModel(model_id="amazon/chronos-t5-tiny")
    assert isinstance(model, ForecastModel)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert model.name == "Chronos"


def test_chronos_predict_shape_and_non_negative(daily_data):
    """(b) predict() tren 1 fold: dung 60 x 28 = 1680 dong, forecast >= 0, khong NaN."""
    daily_df = _ensure_d_num(daily_data)
    origin_day = 1577
    horizon = 28

    history_df = daily_df[daily_df["d_num"] <= origin_day].copy()
    test_df = daily_df[
        (daily_df["d_num"] > origin_day) & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()

    model = ChronosModel(model_id="amazon/chronos-t5-tiny", device="cpu", num_samples=10, context_length=128)
    model.fit(history_df)

    pred_df = model.predict(
        horizon=horizon,
        future_df=test_df[["date", "item_id"]].drop_duplicates(),
    )

    # 1. Kiem tra so dong: dung 60 x 28 = 1680
    assert len(pred_df) == 60 * 28, f"Ky vong 1680 dong, thuc te {len(pred_df)}"

    # 2. Cac cot bat buoc
    assert set(["date", "item_id", "forecast"]).issubset(pred_df.columns)

    # 3. Khong co NaN
    assert pred_df["forecast"].notna().all(), "Du bao co gia tri NaN"

    # 4. Khong co gia tri am
    assert (pred_df["forecast"] >= 0.0).all(), "Du bao co gia tri am"

    # 5. Du 60 SKU duy nhat
    assert pred_df["item_id"].nunique() == 60


def test_chronos_leak_resistance(daily_data):
    """(c) Leak test: sua du lieu sau origin -> du bao Chronos khong doi."""
    daily_df = _ensure_d_num(daily_data)
    origin_day = 1577
    horizon = 28

    # Du lieu sach
    history_clean = daily_df[daily_df["d_num"] <= origin_day].copy()
    test_clean = daily_df[
        (daily_df["d_num"] > origin_day) & (daily_df["d_num"] <= origin_day + horizon)
    ].copy()

    model_clean = ChronosModel(model_id="amazon/chronos-t5-tiny", device="cpu", num_samples=10, context_length=128)
    model_clean.fit(history_clean)
    preds_clean = model_clean.predict(horizon=horizon, future_df=test_clean[["date", "item_id"]])

    # Du lieu sau origin bi sua doi (nhan 1000)
    dirty_daily = daily_df.copy()
    dirty_mask = dirty_daily["d_num"] > origin_day
    dirty_daily.loc[dirty_mask, "sales"] = dirty_daily.loc[dirty_mask, "sales"] * 1000

    history_dirty = dirty_daily[dirty_daily["d_num"] <= origin_day].copy()
    test_dirty = dirty_daily[
        (dirty_daily["d_num"] > origin_day) & (dirty_daily["d_num"] <= origin_day + horizon)
    ].copy()

    model_dirty = ChronosModel(model_id="amazon/chronos-t5-tiny", device="cpu", num_samples=10, context_length=128)
    model_dirty.fit(history_dirty)
    preds_dirty = model_dirty.predict(horizon=horizon, future_df=test_dirty[["date", "item_id"]])

    # So sanh du bao 2 lan
    pd.testing.assert_frame_equal(
        preds_clean.sort_values(by=["item_id", "date"]).reset_index(drop=True),
        preds_dirty.sort_values(by=["item_id", "date"]).reset_index(drop=True),
        check_exact=True,
    )
