"""Cac mo hinh baseline co so cho du bao chuoi thoi gian.

1. NaiveModel: Lap lai doanh so cua ngay quan sat cuoi cung (t).
2. SeasonalNaiveModel: Lap lai chu ky tuan (lag 7) cua 7 ngay cuoi cung.
3. MovingAverageModel: Lap lai gia tri trung binh cua 28 ngay cuoi cung.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.forecasting.base import ForecastModel


class NaiveModel(ForecastModel):
    """Mo hinh Naive: lay gia tri cua ngay cuoi cung truoc origin."""

    def __init__(self, name: str = "Naive"):
        super().__init__(name=name)
        self.last_values_: Dict[str, float] = {}
        self.last_dates_: Dict[str, pd.Timestamp] = {}

    def fit(self, history_df: pd.DataFrame) -> "NaiveModel":
        sorted_df = history_df.sort_values(by=["item_id", "date"])
        # Lay ban ghi cuoi cung cho tung SKU
        last_records = sorted_df.groupby("item_id").last().reset_index()
        self.last_values_ = dict(zip(last_records["item_id"], last_records["sales"].astype(float)))
        self.last_dates_ = dict(zip(last_records["item_id"], pd.to_datetime(last_records["date"])))
        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # future_context_df khong duoc dung boi Baseline (bo qua)
        records = []
        if future_df is not None:
            items = future_df["item_id"].unique()
            for item in items:
                val = self.last_values_.get(item, 0.0)
                sub_dates = future_df[future_df["item_id"] == item]["date"].unique()
                for d in sub_dates:
                    records.append({"date": d, "item_id": item, "forecast": float(val)})
        else:
            for item, val in self.last_values_.items():
                start_date = self.last_dates_[item]
                for h in range(1, horizon + 1):
                    fc_date = start_date + pd.Timedelta(days=h)
                    records.append({"date": fc_date, "item_id": item, "forecast": float(val)})

        res = pd.DataFrame(records)
        res["forecast"] = np.maximum(res["forecast"], 0.0)
        return res[["date", "item_id", "forecast"]].sort_values(by=["item_id", "date"]).reset_index(drop=True)


class SeasonalNaiveModel(ForecastModel):
    """Mo hinh Seasonal Naive (chu ky tuan, lag 7).
    
    Lay chuoi 7 ngay cuoi cung truoc origin va lap lai theo thu trong tuan.
    """

    def __init__(self, season_length: int = 7, name: str = "SeasonalNaive"):
        super().__init__(name=name)
        self.season_length = season_length
        self.last_seasons_: Dict[str, np.ndarray] = {}
        self.last_dates_: Dict[str, pd.Timestamp] = {}

    def fit(self, history_df: pd.DataFrame) -> "SeasonalNaiveModel":
        sorted_df = history_df.sort_values(by=["item_id", "date"])
        self.last_seasons_ = {}
        self.last_dates_ = {}
        
        for item, group in sorted_df.groupby("item_id"):
            sales_arr = group["sales"].values.astype(float)
            if len(sales_arr) < self.season_length:
                # Neu lich su ngan hon 7 ngay, pad bang gia tri dau
                pad_len = self.season_length - len(sales_arr)
                season = np.pad(sales_arr, (pad_len, 0), mode="edge")
            else:
                season = sales_arr[-self.season_length:]
            self.last_seasons_[item] = season
            self.last_dates_[item] = pd.to_datetime(group["date"].iloc[-1])
            
        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        records = []
        if future_df is not None:
            for item in future_df["item_id"].unique():
                season = self.last_seasons_.get(item, np.zeros(self.season_length))
                sub_dates = sorted(future_df[future_df["item_id"] == item]["date"].unique())
                for i, d in enumerate(sub_dates):
                    fc_val = season[i % self.season_length]
                    records.append({"date": d, "item_id": item, "forecast": float(fc_val)})
        else:
            for item, season in self.last_seasons_.items():
                start_date = self.last_dates_[item]
                for h in range(1, horizon + 1):
                    fc_date = start_date + pd.Timedelta(days=h)
                    fc_val = season[(h - 1) % self.season_length]
                    records.append({"date": fc_date, "item_id": item, "forecast": float(fc_val)})

        res = pd.DataFrame(records)
        res["forecast"] = np.maximum(res["forecast"], 0.0)
        return res[["date", "item_id", "forecast"]].sort_values(by=["item_id", "date"]).reset_index(drop=True)


class MovingAverageModel(ForecastModel):
    """Mo hinh Moving Average: tinh trung binh cua window ngay cuoi cung."""

    def __init__(self, window: int = 28, name: str = "MovingAverage"):
        super().__init__(name=name)
        self.window = window
        self.means_: Dict[str, float] = {}
        self.last_dates_: Dict[str, pd.Timestamp] = {}

    def fit(self, history_df: pd.DataFrame) -> "MovingAverageModel":
        sorted_df = history_df.sort_values(by=["item_id", "date"])
        self.means_ = {}
        self.last_dates_ = {}
        
        for item, group in sorted_df.groupby("item_id"):
            sales_arr = group["sales"].values.astype(float)
            w = min(len(sales_arr), self.window)
            if w > 0:
                mean_val = float(np.mean(sales_arr[-w:]))
            else:
                mean_val = 0.0
            self.means_[item] = mean_val
            self.last_dates_[item] = pd.to_datetime(group["date"].iloc[-1])
            
        return self

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        records = []
        if future_df is not None:
            for item in future_df["item_id"].unique():
                val = self.means_.get(item, 0.0)
                sub_dates = future_df[future_df["item_id"] == item]["date"].unique()
                for d in sub_dates:
                    records.append({"date": d, "item_id": item, "forecast": float(val)})
        else:
            for item, val in self.means_.items():
                start_date = self.last_dates_[item]
                for h in range(1, horizon + 1):
                    fc_date = start_date + pd.Timedelta(days=h)
                    records.append({"date": fc_date, "item_id": item, "forecast": float(val)})

        res = pd.DataFrame(records)
        res["forecast"] = np.maximum(res["forecast"], 0.0)
        return res[["date", "item_id", "forecast"]].sort_values(by=["item_id", "date"]).reset_index(drop=True)
