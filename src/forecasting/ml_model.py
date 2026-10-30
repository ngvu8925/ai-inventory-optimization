"""Mo hinh LightGBM cho du bao nhu cau (Phase 3b).

Chien luoc: Direct multi-step forecasting
- 1 model chung cho tat ca 60 SKU.
- Moi hang train = (SKU, origin t, horizon_step h = 1..28).
- horizon_h la 1 feature rieng biet -> model hoc duoc cach du bao tung buoc khac nhau.
- Muc tieu: Poisson hoac Tweedie (phu hop voi du lieu co nhieu gia tri 0).
- Du bao khong am (clip 0), khong lam tron.

GIA DINH GHI RO: Gia ban va lich su kien cua 28 ngay du bao duoc xem la biet truoc
(oracle features). Baseline khong co nhung feature nay, can neu ro khi so sanh.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.forecasting.base import ForecastModel
from src.forecasting.features import (
    FEATURE_COLS,
    _ensure_d_num,
    build_features_for_origin,
    build_train_dataset,
)


class LightGBMModel(ForecastModel):
    """Mo hinh LightGBM direct multi-step forecasting.

    fit() nhan toan bo lich su den origin va danh sach origin huan luyen.
    predict() xay dung feature cho origin hien tai va goi model.
    """

    def __init__(
        self,
        lgbm_params: Optional[Dict] = None,
        train_origins: Optional[List[int]] = None,
        horizon: int = 28,
        name: str = "LightGBM",
    ):
        """
        lgbm_params: Sieu tham so LightGBM (doc tu configs/lightgbm.yaml).
        train_origins: Danh sach cac origin dung de huan luyen model.
                      Neu None -> chi dung origin hien tai (origin cua fold).
        horizon: Do dai du bao (mac dinh 28).
        """
        super().__init__(name=name)
        self.lgbm_params = lgbm_params or {}
        self.train_origins = train_origins  # se duoc set boi backtest hoac tune
        self.horizon = horizon
        self.model_: Optional[lgb.Booster] = None
        self._history_df: Optional[pd.DataFrame] = None  # luu history de predict

    def fit(self, history_df: pd.DataFrame) -> "LightGBMModel":
        """Huan luyen model tren lich su du lieu.

        history_df: du lieu TU d_1 DEN d_{origin_k} (expanding).
        """
        self._history_df = history_df.copy()
        self._history_df["date"] = pd.to_datetime(self._history_df["date"])

        # Them d_num 1 lan de filter nhanh
        self._history_df = _ensure_d_num(self._history_df)

        # Xac dinh origin cua fold hien tai
        current_origin = int(self._history_df["d_num"].max())

        # Cac origin de lay mau huan luyen phai co du target 28 ngay nam trong history_df
        # Tuc la origin + horizon <= current_origin
        max_train_origin = current_origin - self.horizon

        if self.train_origins is not None:
            valid_origins = [o for o in self.train_origins if o <= max_train_origin]
        else:
            # Mac dinh: lay cac origin step=7 tu d_56 toi max_train_origin
            valid_origins = list(range(56, max_train_origin + 1, 7))

        if len(valid_origins) == 0:
            valid_origins = [max(56, max_train_origin)]

        # Neu co it nhat 3 origin, tach origin cuoi cung lam val de early stopping
        use_val = len(valid_origins) >= 3
        if use_val:
            train_origins_actual = valid_origins[:-1]
            val_origins = valid_origins[-1:]
        else:
            train_origins_actual = valid_origins
            val_origins = []

        # Build tap huan luyen
        X_train, y_train = build_train_dataset(
            daily_df=self._history_df,
            train_origins=train_origins_actual,
            horizon=self.horizon,
        )

        valid_mask = ~np.isnan(y_train)
        X_train = X_train[valid_mask]
        y_train = np.maximum(y_train[valid_mask], 0.0)

        # Xay dung LightGBM Dataset
        lgb_train = lgb.Dataset(
            X_train[FEATURE_COLS],
            label=y_train,
            free_raw_data=False,
        )

        # Tham so mac dinh neu khong co config
        params = {
            "objective": "poisson",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 20,
            "seed": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        params.update(self.lgbm_params)

        num_boost_round = int(params.pop("n_estimators", 300))

        if use_val:
            X_val, y_val = build_train_dataset(
                daily_df=self._history_df,
                train_origins=val_origins,
                horizon=self.horizon,
            )
            valid_mask_val = ~np.isnan(y_val)
            X_val = X_val[valid_mask_val]
            y_val = np.maximum(y_val[valid_mask_val], 0.0)

            lgb_val = lgb.Dataset(
                X_val[FEATURE_COLS],
                label=y_val,
                reference=lgb_train,
                free_raw_data=False,
            )
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            self.model_ = lgb.train(
                params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )
        else:
            self.model_ = lgb.train(
                params,
                lgb_train,
                num_boost_round=num_boost_round,
                callbacks=[lgb.log_evaluation(-1)],
            )

        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Du bao nhu cau cho horizon ngay toi.

        future_context_df: DataFrame voi thong tin gia + lich cua giai doan du bao.
                          Bat buoc phai truyen cho LightGBMModel.
        """
        if self.model_ is None:
            raise RuntimeError("Model chua duoc huan luyen. Goi fit() truoc.")
        if future_context_df is None:
            raise ValueError(
                "LightGBMModel can future_context_df (gia + lich ngay du bao). "
                "Truyen future_context_df khi goi predict()."
            )
        if self._history_df is None:
            raise RuntimeError("history_df chua duoc luu. Goi fit() truoc.")

        # Xac dinh origin hien tai
        d_vals = self._history_df["d_num"] if "d_num" in self._history_df.columns else \
                 self._history_df["d"].str.replace("d_", "", regex=False).astype(int)
        current_origin = int(d_vals.max())

        # Build feature cho origin hien tai
        X_pred = build_features_for_origin(
            history_df=self._history_df,
            future_context_df=future_context_df,
            origin_day=current_origin,
            horizon=horizon,
        )

        # Predict
        raw_preds = self.model_.predict(X_pred[FEATURE_COLS])
        raw_preds = np.maximum(raw_preds, 0.0)  # clip khong am

        X_pred = X_pred.copy()
        X_pred["forecast"] = raw_preds

        # Lay ngay tuong ung tu future_context_df
        ctx_sorted = future_context_df.sort_values(["item_id", "date"]).copy()
        ctx_sorted["horizon_h"] = ctx_sorted.groupby("item_id").cumcount() + 1.0

        result = X_pred[["item_id", "horizon_h", "forecast"]].merge(
            ctx_sorted[["item_id", "horizon_h", "date"]],
            on=["item_id", "horizon_h"],
            how="left",
        )

        result = result[["date", "item_id", "forecast"]].sort_values(
            by=["item_id", "date"]
        ).reset_index(drop=True)

        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """Tra ve feature importance (gain) cua model."""
        if self.model_ is None:
            raise RuntimeError("Model chua duoc huan luyen.")
        imp = self.model_.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance_gain": imp,
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
