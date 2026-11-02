"""Chronos zero-shot models, isolated from the main forecasting imports."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.forecasting.base import ForecastModel

ChronosVariant = Literal["chronos2_history", "chronos2_exogenous", "bolt_history"]
CONTEXT_LENGTH = 1024
MODEL_IDS = {
    "chronos2_history": "amazon/chronos-2",
    "chronos2_exogenous": "amazon/chronos-2",
    "bolt_history": "amazon/chronos-bolt-small",
}
MODEL_NAMES = {
    "chronos2_history": "Chronos2-History",
    "chronos2_exogenous": "Chronos2-Exogenous",
    "bolt_history": "ChronosBoltSmall-History",
}
EXOGENOUS_COLUMNS = (
    "weekday_h",
    "month_h",
    "dayofmonth_h",
    "has_event_h",
    "snap_CA_h",
    "sell_price_h",
    "price_ratio_h",
)


class ChronosModel(ForecastModel):
    """Zero-shot Chronos wrapper with optional known-future covariates.

    Chronos and torch are imported only when a model is loaded, so importing
    this module does not require the TSFM environment.
    """

    def __init__(
        self,
        variant: ChronosVariant,
        seed: int = 42,
        model_id: Optional[str] = None,
    ) -> None:
        if variant not in MODEL_IDS:
            raise ValueError(f"Unknown Chronos variant: {variant}")
        super().__init__(MODEL_NAMES[variant])
        self.variant = variant
        self.model_id = model_id or MODEL_IDS[variant]
        self.seed = seed
        self._history: Optional[pd.DataFrame] = None
        self._pipeline = None
        self._torch = None
        self._quantile_levels: list[float] = []

    @property
    def uses_exogenous(self) -> bool:
        return self.variant == "chronos2_exogenous"

    def fit(self, history_df: pd.DataFrame) -> "ChronosModel":
        """Store history up to the current origin; no weights are updated."""
        required = {"date", "item_id", "sales"}
        missing = required.difference(history_df.columns)
        if missing:
            raise ValueError(f"Missing history columns: {sorted(missing)}")
        history = history_df.copy()
        history["date"] = pd.to_datetime(history["date"])
        self._history = history.sort_values(["item_id", "date"]).reset_index(drop=True)
        return self

    def _load_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        import torch
        from chronos import Chronos2Pipeline, ChronosBoltPipeline

        self._torch = torch
        torch.manual_seed(self.seed)
        if self.variant.startswith("chronos2"):
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id, device_map="cpu"
            )
        else:
            self._pipeline = ChronosBoltPipeline.from_pretrained(
                self.model_id, device_map="cpu"
            )
        available = [float(q) for q in self._pipeline.quantiles]
        wanted = [0.1, 0.5, 0.9]
        if 0.95 in available:
            wanted.append(0.95)
        if not set(wanted).issubset(available):
            raise RuntimeError(f"Model quantiles {available} do not include {wanted}")
        self._quantile_levels = wanted

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _feature_frame(frame: pd.DataFrame, origin_price: float) -> pd.DataFrame:
        """Build the same target-day exogenous features used by LightGBM."""
        result = pd.DataFrame(index=frame.index)
        dates = pd.to_datetime(frame["date"])
        result["weekday_h"] = pd.to_numeric(frame["wday"], errors="coerce").fillna(0)
        result["month_h"] = pd.to_numeric(frame["month"], errors="coerce").fillna(dates.dt.month)
        result["dayofmonth_h"] = dates.dt.day
        event = frame.get("event_name_1", pd.Series(index=frame.index, dtype=object))
        result["has_event_h"] = (event.notna() & (event.astype(str) != "")).astype(float)
        snap = frame.get("snap_CA", pd.Series(0.0, index=frame.index))
        if not isinstance(snap, pd.Series):
            snap = pd.Series(snap, index=frame.index)
        result["snap_CA_h"] = pd.to_numeric(snap, errors="coerce")
        result["snap_CA_h"] = result["snap_CA_h"].fillna(0)

        price = pd.to_numeric(frame.get("sell_price", origin_price), errors="coerce")
        if not isinstance(price, pd.Series):
            price = pd.Series(price, index=frame.index, dtype=float)
        price = price.fillna(origin_price)
        result["sell_price_h"] = price.astype(float)
        # Cùng LightGBM: target-date price / price at the forecast origin.
        result["price_ratio_h"] = (
            result["sell_price_h"] / origin_price if origin_price > 0 else 1.0
        )
        return result.loc[:, list(EXOGENOUS_COLUMNS)].astype(np.float32)

    def _origin_price(self, item_history: pd.DataFrame) -> float:
        prices = pd.to_numeric(item_history.get("sell_price"), errors="coerce").dropna()
        return float(prices.iloc[-1]) if len(prices) else 0.0

    def _chronos2_inputs(
        self, history: pd.DataFrame, future_context_df: Optional[pd.DataFrame]
    ) -> list:
        inputs = []
        for item_id, item_history in history.groupby("item_id", sort=True):
            item_history = item_history.sort_values("date").tail(CONTEXT_LENGTH)
            target = item_history["sales"].to_numpy(dtype=np.float32)
            item_input = {"target": target}
            if self.uses_exogenous:
                if future_context_df is None:
                    raise ValueError("Chronos2-Exogenous requires future_context_df")
                origin_price = self._origin_price(
                    self._history[self._history["item_id"] == item_id]
                )
                past_features = self._feature_frame(item_history, origin_price)
                future_item = future_context_df[future_context_df["item_id"] == item_id]
                future_item = future_item.sort_values("date")
                future_features = self._feature_frame(future_item, origin_price)
                item_input["past_covariates"] = {
                    col: past_features[col].to_numpy(dtype=np.float32)
                    for col in EXOGENOUS_COLUMNS
                }
                item_input["future_covariates"] = {
                    col: future_features[col].to_numpy(dtype=np.float32)
                    for col in EXOGENOUS_COLUMNS
                }
            inputs.append(item_input)
        return inputs

    def predict(
        self,
        horizon: int,
        future_df: Optional[pd.DataFrame] = None,
        future_context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return q0.5 as the nonnegative point forecast and retain quantiles."""
        if self._history is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self._load_pipeline()

        history = self._history
        if future_df is None:
            last_date = history["date"].max()
            items = history["item_id"].drop_duplicates().tolist()
            future_df = pd.DataFrame(
                [
                    {"item_id": item, "date": date}
                    for item in items
                    for date in pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon)
                ]
            )
        future = future_df[["date", "item_id"]].copy()
        future["date"] = pd.to_datetime(future["date"])
        future = future.sort_values(["item_id", "date"]).reset_index(drop=True)
        if future.groupby("item_id").size().ne(horizon).any():
            raise ValueError("future_df must contain exactly horizon dates per item")

        contexts = []
        for item_id, item_history in history.groupby("item_id", sort=True):
            values = item_history.sort_values("date")["sales"].tail(CONTEXT_LENGTH).to_numpy(
                dtype=np.float32
            )
            contexts.append(values)

        torch = self._torch
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            if self.variant.startswith("chronos2"):
                inputs = self._chronos2_inputs(history, future_context_df)
                quantile_list, _ = self._pipeline.predict_quantiles(
                    inputs,
                    prediction_length=horizon,
                    quantile_levels=self._quantile_levels,
                    context_length=CONTEXT_LENGTH,
                )
                quantile_values = np.stack(
                    [self._to_numpy(q).reshape(horizon, len(self._quantile_levels))
                     for q in quantile_list]
                )
            else:
                # Mọi SKU có cùng ngày origin nên context sau cắt có cùng độ dài.
                context_tensor = torch.as_tensor(np.stack(contexts), dtype=torch.float32)
                quantiles, _ = self._pipeline.predict_quantiles(
                    context_tensor,
                    prediction_length=horizon,
                    quantile_levels=self._quantile_levels,
                )
                quantile_values = self._to_numpy(quantiles)

        if quantile_values.shape != (len(contexts), horizon, len(self._quantile_levels)):
            raise RuntimeError(f"Unexpected quantile output shape: {quantile_values.shape}")
        quantile_values = np.maximum(quantile_values, 0.0)
        if not np.isfinite(quantile_values).all():
            raise RuntimeError("Chronos returned non-finite quantiles")

        ordered_items = sorted(history["item_id"].unique())
        item_index = {item_id: i for i, item_id in enumerate(ordered_items)}
        rows = []
        for _, row in future.iterrows():
            item_i = item_index[row["item_id"]]
            forecast_row = {
                "date": row["date"],
                "item_id": row["item_id"],
                "forecast": float(quantile_values[item_i, len(rows) % horizon,
                                                   self._quantile_levels.index(0.5)]),
            }
            for q_i, q in enumerate(self._quantile_levels):
                forecast_row[f"forecast_q{int(round(q * 100)):02d}"] = float(
                    quantile_values[item_i, len(rows) % horizon, q_i]
                )
            rows.append(forecast_row)
        return pd.DataFrame(rows).sort_values(["item_id", "date"]).reset_index(drop=True)
