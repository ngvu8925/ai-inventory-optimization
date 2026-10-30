"""Mo hinh Time-Series Foundation Model (Chronos) cho du bao nhu cau (Phase 4).

Chien luoc: Zero-shot Forecasting
- Su dung mo hinh pretrained Chronos (Amazon) dua tren kien truc T5.
- Zero-shot hoan toan: KHONG fine-tune tren du lieu M5 (tuan thu nghiem ngat luat AGENTS.md).
- Du bao xac suat: lay median (quantile 0.5) cua 20 mau sample.
- Du bao khong am (clip >= 0), khong lam tron.
- Batch inference cho toan bo 60 SKU giup toc do du bao cuc nhanh tren CPU.
"""

import sys
import os
from pathlib import Path

# Thiet lap offline mode de tranh retry timeout tren sandbox khong co internet
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from src.forecasting.base import ForecastModel


class ChronosModel(ForecastModel):
    """Lop bọc mo hinh Chronos zero-shot, ke thua tu ForecastModel."""

    def __init__(
        self,
        model_id: str = "amazon/chronos-t5-tiny",
        device: str = "cpu",
        num_samples: int = 20,
        context_length: Optional[int] = 512,
        seed: int = 42,
        name: str = "Chronos",
    ):
        """
        model_id: Ten checkpoint Hugging Face (mac dinh amazon/chronos-t5-tiny).
        device: 'cpu' hoac 'cuda'.
        num_samples: So luong mau sample de tinh phan vi (mac dinh 20).
        context_length: So ngay lich su toi da dua vao context window (mac dinh 512 ngay).
        seed: Co dinh seed de dam bao tinh tai lap (reproducible) theo Luat 9 AGENTS.md.
        """
        super().__init__(name=name)
        self.model_id = model_id
        self.device = device
        self.num_samples = num_samples
        self.context_length = context_length
        self.seed = seed
        self._history_df: Optional[pd.DataFrame] = None
        self._pipeline: Optional[ChronosPipeline] = None

    def _ensure_pipeline(self) -> ChronosPipeline:
        """Khoi tao hoac lay pipeline da duoc load."""
        if self._pipeline is None:
            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=torch.float32,
            )
        return self._pipeline

    def fit(self, history_df: pd.DataFrame) -> "ChronosModel":
        """Chuan bi du lieu lich su.
        
        Vi la mo hinh zero-shot, buoc fit khong cap nhat trong so mo hinh
        ma chi luu lai lich su ban hang de phuc vu predict().
        """
        self._history_df = history_df.copy()
        self._history_df["date"] = pd.to_datetime(self._history_df["date"])
        # Khoi tao san pipeline neu chua co
        self._ensure_pipeline()
        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Du bao nhu cau cho horizon ngay tiep theo bang Chronos zero-shot.
        
        Du bao dong thoi (batch) cho tat ca 60 SKU.
        """
        if self._history_df is None:
            raise RuntimeError("Model chua co lich su. Goi fit() truoc.")

        pipeline = self._ensure_pipeline()

        # Sap xep va lay danh sach item_id
        hist = self._history_df.sort_values(by=["item_id", "date"]).copy()
        unique_items = sorted(hist["item_id"].unique())

        context_tensors = []
        for item in unique_items:
            series = hist[hist["item_id"] == item]["sales"].values.astype(np.float32)
            if self.context_length is not None and len(series) > self.context_length:
                series = series[-self.context_length:]
            context_tensors.append(torch.from_numpy(series))

        # Batch prediction tren toan bo 60 chuoi
        # Co dinh seed de ket qua hoan toan tai lap (reproducible)
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # forecast shape: (num_items, num_samples, horizon)
        with torch.no_grad():
            forecast_samples = pipeline.predict(
                context_tensors,
                prediction_length=horizon,
                num_samples=self.num_samples,
            )

        # Lay median forecast: quantile 0.5
        # median_forecast shape: (num_items, horizon)
        median_forecast = torch.quantile(forecast_samples, 0.5, dim=1).cpu().numpy()
        # Clip khong am
        median_forecast = np.maximum(median_forecast, 0.0)

        # Xac dinh cac ngay tuong lai
        if future_df is not None and "date" in future_df.columns:
            future_dates = sorted(pd.to_datetime(future_df["date"]).unique())
        elif future_context_df is not None and "date" in future_context_df.columns:
            future_dates = sorted(pd.to_datetime(future_context_df["date"]).unique())
        else:
            last_date = hist["date"].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

        records = []
        for i, item in enumerate(unique_items):
            item_preds = median_forecast[i]
            for h_idx in range(horizon):
                d = future_dates[h_idx]
                records.append({
                    "date": d,
                    "item_id": item,
                    "forecast": float(item_preds[h_idx]),
                })

        pred_df = pd.DataFrame(records).sort_values(by=["item_id", "date"]).reset_index(drop=True)
        return pred_df
