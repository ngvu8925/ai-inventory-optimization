"""Slow CPU integration checks for Chronos leakage and output shape.

Run explicitly with: pytest -m slow tests/test_chronos_slow.py
The small Bolt-Tiny checkpoint is used to keep this opt-in test shorter.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TEMP"] = r"D:\pip_cache"
os.environ["TMP"] = r"D:\pip_cache"

import pandas as pd
import pytest

pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_processed_daily
from src.forecasting.chronos_model import ChronosModel
from src.forecasting.features import _ensure_d_num


@pytest.fixture(scope="module")
def fold_data():
    daily = _ensure_d_num(load_processed_daily())
    origin = 1577
    horizon = 28
    history = daily[daily["d_num"] <= origin].copy()
    future = daily[
        (daily["d_num"] > origin) & (daily["d_num"] <= origin + horizon)
    ].copy()
    return daily, origin, horizon, history, future


def _predict(history: pd.DataFrame, future: pd.DataFrame, horizon: int) -> pd.DataFrame:
    model = ChronosModel(
        variant="bolt_history",
        model_id="amazon/chronos-bolt-tiny",
        seed=42,
    )
    model.fit(history)
    return model.predict(
        horizon=horizon,
        future_df=future[["date", "item_id"]].drop_duplicates(),
    )


def test_bolt_tiny_fold_has_expected_rows_and_nonnegative_quantiles(fold_data):
    _, _, horizon, history, future = fold_data
    predictions = _predict(history, future, horizon)
    assert len(predictions) == 60 * 28
    assert predictions["item_id"].nunique() == 60
    assert predictions["date"].nunique() == 28
    quantile_cols = [column for column in predictions if column.startswith("forecast_q")]
    assert {"forecast_q10", "forecast_q50", "forecast_q90"}.issubset(quantile_cols)
    assert (predictions[quantile_cols] >= 0).all().all()
    assert predictions[quantile_cols].notna().all().all()


def test_bolt_tiny_fold_is_invariant_to_sales_after_origin(fold_data):
    daily, origin, horizon, history, future = fold_data
    clean = _predict(history, future, horizon)

    changed = daily.copy()
    changed.loc[changed["d_num"] > origin, "sales"] *= 1000
    dirty_history = changed[changed["d_num"] <= origin].copy()
    dirty_future = changed[
        (changed["d_num"] > origin) & (changed["d_num"] <= origin + horizon)
    ].copy()
    dirty = _predict(dirty_history, dirty_future, horizon)

    pd.testing.assert_frame_equal(
        clean.sort_values(["item_id", "date"]).reset_index(drop=True),
        dirty.sort_values(["item_id", "date"]).reset_index(drop=True),
        check_exact=True,
    )
