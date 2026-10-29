"""Giao dien co so (Interface) cho moi mo hinh du bao nhu cau trong he thong.

Moi mo hinh (Baseline, LightGBM/XGBoost, Chronos/TimesFM) deu phai ke thua
lop ForecastModel va trien khai fit() cung predict().
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
    ) -> pd.DataFrame:
        """Du bao nhu cau cho horizon ngay tiep theo.
        
        Tra ve pd.DataFrame voi dung cac cot: ['date', 'item_id', 'forecast'].
        Gia tri forecast khong duoc chua NaN va khong am (sales >= 0).
        """
        pass

    def __repr__(self) -> str:
        return f"ForecastModel(name='{self.name}')"
