"""Giao dien co so (Interface) cho moi mo hinh du bao nhu cau trong he thong.

Moi mo hinh (Baseline, LightGBM/XGBoost, Chronos/TimesFM) deu phai ke thua
lop ForecastModel va trien khai fit() cung predict().

Thay doi Phase 3b (tuong thich nguoc):
- predict() co them tham so tuy chon future_context_df chua bien ngoai sinh
  (gia ban, lich su kien, SNAP) cua giai doan du bao.
- Baseline khong dung future_context_df (mac dinh None) -> chay binh thuong.
- LightGBM dung future_context_df de lay gia va lich khi xay dung feature.
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class ForecastModel(ABC):
    """Lop co so truu tuong cho cac mo hinh du bao."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, history_df: pd.DataFrame) -> "ForecastModel":
        """Huan luyen mo hinh hoac luu tru lich su ban hang den thoi diem origin.

        history_df phai chua it nhat cac cot: ['date', 'item_id', 'sales'].
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Du bao nhu cau cho horizon ngay tiep theo.

        Tham so:
            horizon: So ngay can du bao.
            future_df: DataFrame chua ['date', 'item_id'] cua giai doan du bao
                (baseline va LightGBM deu dung).
            future_context_df: DataFrame chua bien ngoai sinh cua giai doan du bao
                (['date', 'item_id', 'sell_price', 'wday', 'month', 'day',
                  'event_name_1', 'event_type_1', 'snap_CA', ...]).
                Baseline bo qua tham so nay (None). LightGBM dung de build feature.

        GIA DINH (ghi ro): gia ban va lich su kien cua 28 ngay toi duoc xem la
            biet truoc tai thoi diem du bao (oracle features). Day la gia dinh
            pho bien trong tai lieu M5 nhung can ghi ro khi bao cao so sanh.

        Tra ve pd.DataFrame voi dung cac cot: ['date', 'item_id', 'forecast'].
        Gia tri forecast khong duoc chua NaN va khong am (sales >= 0).
        """
        pass

    def __repr__(self) -> str:
        return f"ForecastModel(name='{self.name}')"
