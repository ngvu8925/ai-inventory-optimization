"""Lop ChronosModel cho Phase 4: Time-Series Foundation Model (Zero-shot).

Ho tro 3 bien the theo dung quyet dinh nghien cuu:
1. 'chronos2_history': Chronos-2 (amazon/chronos-2) chi dung lich su (univariate).
2. 'chronos2_covariates': Chronos-2 (amazon/chronos-2) dung lich su + bien ngoai sinh
   biet truoc 28 ngay (sell_price, snap_CA, has_event, wday, month).
3. 'chronos_bolt': Chronos-Bolt (amazon/chronos-bolt-mini hoac tiny) chi dung lich su.

Dac ta bat buoc:
- Zero-shot hoan toan, KHONG fine-tune, KHONG tune tren hold-out.
- Cat ngu canh lich su toi da 1024 ngay gan nhat truoc origin.
- Diem du bao (point forecast) = trung vi (quantile 0.5), clip >= 0, khong lam tron.
- Xuat kem cac phan vi [0.1, 0.5, 0.9, 0.95] phuc vu tinh ton kho an toan o Phase 5.
- Co dinh seed dam bao tinh tai lap (Rule 9 AGENTS.md).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Offline cache de tranh goi mang ngoai y muon
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch

from src.forecasting.base import ForecastModel
from src.forecasting.features import _ensure_d_num


COVARIATE_COLS = ["sell_price", "snap_CA", "has_event", "wday", "month"]
QUANTILES = [0.1, 0.5, 0.9, 0.95]


class ChronosModel(ForecastModel):
    """Mo hinh Chronos Zero-shot ho tro 3 bien the."""

    def __init__(
        self,
        variant: str = "chronos2_history",
        model_id: Optional[str] = None,
        context_length: int = 1024,
        device: str = "cpu",
        seed: int = 42,
        name: Optional[str] = None,
    ):
        """
        variant: 'chronos2_history', 'chronos2_covariates', hoac 'chronos_bolt'.
        model_id: Checkpoint Hugging Face (neu None thi tu chon theo variant).
        context_length: So ngay lich su toi da (mac dinh 1024).
        device: 'cpu' hoac 'cuda'.
        seed: Co dinh seed de dam bao tai lap.
        """
        valid_variants = ["chronos2_history", "chronos2_covariates", "chronos_bolt"]
        if variant not in valid_variants:
            raise ValueError(f"variant phai thuoc {valid_variants}, nhan: {variant}")

        self.variant = variant
        self.context_length = context_length
        self.device = device
        self.seed = seed

        if model_id is None:
            if self.variant in ["chronos2_history", "chronos2_covariates"]:
                self.model_id = "amazon/chronos-2"
            else:
                self.model_id = "amazon/chronos-bolt-mini"
        else:
            self.model_id = model_id

        default_name = {
            "chronos2_history": "Chronos2_History",
            "chronos2_covariates": "Chronos2_Covariates",
            "chronos_bolt": "Chronos_Bolt",
        }[variant]
        super().__init__(name=name or default_name)

        self._history_df: Optional[pd.DataFrame] = None
        self._pipeline = None

    def _ensure_pipeline(self):
        """Khoi tao pipeline tuong ung voi variant neu chua load."""
        if self._pipeline is None:
            if self.variant in ["chronos2_history", "chronos2_covariates"]:
                from chronos import Chronos2Pipeline
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=torch.float32,
                    local_files_only=True,
                )
            else:
                from chronos import ChronosBoltPipeline
                self._pipeline = ChronosBoltPipeline.from_pretrained(
                    self.model_id,
                    device_map=self.device,
                    dtype=torch.float32,
                    local_files_only=True,
                )
        return self._pipeline

    def fit(self, history_df: pd.DataFrame) -> "ChronosModel":
        """Luu tru lich su va chuan bi pipeline zero-shot."""
        self._history_df = history_df.copy()
        self._history_df["date"] = pd.to_datetime(self._history_df["date"])
        self._ensure_pipeline()
        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Du bao nhu cau cho horizon ngay tiep theo."""
        if self._history_df is None:
            raise RuntimeError("Model chua co lich su. Goi fit() truoc.")

        pipeline = self._ensure_pipeline()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # Chuan bi lich su: cat toi da context_length ngay gan nhat cho moi SKU
        hist = self._history_df.sort_values(by=["item_id", "date"]).copy()
        if self.context_length is not None:
            hist = hist.groupby("item_id").tail(self.context_length).reset_index(drop=True)

        unique_items = sorted(hist["item_id"].unique())

        if self.variant == "chronos_bolt":
            # Chronos-Bolt pipeline dung predict_quantiles tren list tensors
            context_tensors = []
            for item in unique_items:
                s = hist[hist["item_id"] == item]["sales"].values.astype(np.float32)
                context_tensors.append(torch.from_numpy(s))

            # Bolt ho tro quantiles len den 0.9. Neu yeu cau 0.95, clamp ve 0.9.
            bolt_quantiles = [0.1, 0.5, 0.9]
            with torch.no_grad():
                q_preds, _ = pipeline.predict_quantiles(
                    inputs=context_tensors,
                    prediction_length=horizon,
                    quantile_levels=bolt_quantiles,
                )
            # q_preds: [batch_size, horizon, num_quantiles]
            q_arr = q_preds.cpu().numpy()

            # Xac dinh cac ngay du bao
            if future_df is not None and "date" in future_df.columns:
                future_dates = sorted(pd.to_datetime(future_df["date"]).unique())
            elif future_context_df is not None and "date" in future_context_df.columns:
                future_dates = sorted(pd.to_datetime(future_context_df["date"]).unique())
            else:
                last_d = hist["date"].max()
                future_dates = pd.date_range(start=last_d + pd.Timedelta(days=1), periods=horizon)

            records = []
            for i, item in enumerate(unique_items):
                item_q = q_arr[i]  # [horizon, 3]
                for h_idx in range(horizon):
                    d = future_dates[h_idx]
                    q10 = max(0.0, float(item_q[h_idx, 0]))
                    q50 = max(0.0, float(item_q[h_idx, 1]))
                    q90 = max(0.0, float(item_q[h_idx, 2]))
                    q95 = q90  # Bolt khong ho tro 0.95 truc tiep, gan bang q90 theo quy dinh clamp
                    records.append({
                        "date": d,
                        "item_id": item,
                        "forecast": q50,  # diem du bao = trung vi (0.5)
                        "q_0.1": q10,
                        "q_0.5": q50,
                        "q_0.9": q90,
                        "q_0.95": q95,
                    })
            return pd.DataFrame(records).sort_values(by=["item_id", "date"]).reset_index(drop=True)

        elif self.variant == "chronos2_history":
            # Chronos-2 univariate (chi dung lich su target)
            ctx_df = hist[["item_id", "date", "sales"]].copy()

            if future_df is not None and "date" in future_df.columns:
                fut_df = future_df[["item_id", "date"]].copy()
                fut_df["date"] = pd.to_datetime(fut_df["date"])
            else:
                last_d = hist["date"].max()
                future_dates = pd.date_range(start=last_d + pd.Timedelta(days=1), periods=horizon)
                fut_rows = []
                for item in unique_items:
                    for d in future_dates:
                        fut_rows.append({"item_id": item, "date": d})
                fut_df = pd.DataFrame(fut_rows)

            fut_df = fut_df.sort_values(by=["item_id", "date"]).reset_index(drop=True)

            with torch.no_grad():
                pred_df = pipeline.predict_df(
                    df=ctx_df,
                    future_df=fut_df,
                    prediction_length=horizon,
                    quantile_levels=QUANTILES,
                    id_column="item_id",
                    timestamp_column="date",
                    target="sales",
                    freq="D",
                )

            # Chuan hoa cot tra ve
            pred_df["forecast"] = np.maximum(pred_df["0.5"].values.astype(float), 0.0)
            pred_df["q_0.1"] = np.maximum(pred_df["0.1"].values.astype(float), 0.0)
            pred_df["q_0.5"] = pred_df["forecast"]
            pred_df["q_0.9"] = np.maximum(pred_df["0.9"].values.astype(float), 0.0)
            pred_df["q_0.95"] = np.maximum(pred_df["0.95"].values.astype(float), 0.0)

            res = pred_df[["date", "item_id", "forecast", "q_0.1", "q_0.5", "q_0.9", "q_0.95"]]
            return res.sort_values(by=["item_id", "date"]).reset_index(drop=True)

        elif self.variant == "chronos2_covariates":
            # Chronos-2 voi bien ngoai sinh biet truoc (sell_price, snap_CA, has_event, wday, month)
            if future_context_df is None:
                raise ValueError("Chronos2_Covariates can truyen future_context_df chua gia va lich su kien!")

            # Chuan bi has_event neu chua co
            ctx_cov = hist.copy()
            if "has_event" not in ctx_cov.columns:
                ctx_cov["has_event"] = (
                    ctx_cov["event_name_1"].notna() & (ctx_cov["event_name_1"] != "")
                ).astype(int)

            f_cov = future_context_df.copy()
            f_cov["date"] = pd.to_datetime(f_cov["date"])
            if "has_event" not in f_cov.columns:
                f_cov["has_event"] = (
                    f_cov["event_name_1"].notna() & (f_cov["event_name_1"] != "")
                ).astype(int)

            cols_to_use = ["item_id", "date", "sales"] + COVARIATE_COLS
            # Dam bao cac cot ton tai day du
            for c in COVARIATE_COLS:
                if c not in ctx_cov.columns:
                    raise KeyError(f"Thieu cot covariate {c} trong history_df")
                if c not in f_cov.columns:
                    raise KeyError(f"Thieu cot covariate {c} trong future_context_df")

            ctx_sub = ctx_cov[cols_to_use].sort_values(["item_id", "date"]).reset_index(drop=True)
            fut_sub = f_cov[["item_id", "date"] + COVARIATE_COLS].sort_values(["item_id", "date"]).reset_index(drop=True)

            with torch.no_grad():
                pred_df = pipeline.predict_df(
                    df=ctx_sub,
                    future_df=fut_sub,
                    prediction_length=horizon,
                    quantile_levels=QUANTILES,
                    id_column="item_id",
                    timestamp_column="date",
                    target="sales",
                    freq="D",
                )

            # Chuan hoa cot tra ve
            pred_df["forecast"] = np.maximum(pred_df["0.5"].values.astype(float), 0.0)
            pred_df["q_0.1"] = np.maximum(pred_df["0.1"].values.astype(float), 0.0)
            pred_df["q_0.5"] = pred_df["forecast"]
            pred_df["q_0.9"] = np.maximum(pred_df["0.9"].values.astype(float), 0.0)
            pred_df["q_0.95"] = np.maximum(pred_df["0.95"].values.astype(float), 0.0)

            res = pred_df[["date", "item_id", "forecast", "q_0.1", "q_0.5", "q_0.9", "q_0.95"]]
            return res.sort_values(by=["item_id", "date"]).reset_index(drop=True)
