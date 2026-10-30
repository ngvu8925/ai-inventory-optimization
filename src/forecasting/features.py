"""Xay dung feature cho mo hinh LightGBM (Phase 3b).

Nguyen tac bat buoc chong ro ri du lieu:
- Feature tai origin t: moi lag/rolling PHAI chi dung lich su tu d_1 den d_t.
- Khong co bat ky gia tri nao sau ngay t trong feature cua tap train.
- Feature ngay dich (t+h): gia ban + lich su kien duoc xem la biet truoc
  (oracle/future-known). Gia dinh nay duoc ghi ro trong docs/ va DATA_CARD.md.

So do feature:
  Moi hang = (item_id, origin_day t, horizon_step h)
  -> y = sales cua ngay t+h

Feature nhom A (tai origin t, chi dung lich su <= t):
  lag_t0, lag_t6, lag_t13, lag_t27       (4 lag)
  roll_mean_7, roll_std_7                 (7 ngay cuoi)
  roll_mean_28, roll_std_28               (28 ngay cuoi)
  roll_mean_56, roll_std_56               (56 ngay cuoi)
  zero_ratio_28                           (ty le ngay ban = 0 trong 28 ngay cuoi)
  days_since_last_sale                    (so ngay tu lan cuoi co ban hang)
  cat_encoded                             (ma nganh hang: 0/1/2)

Feature nhom B (ngay dich t+h, biet truoc):
  horizon_h                               (1..28)
  weekday_h                               (0..6, Mon=0)
  month_h                                 (1..12)
  dayofmonth_h                            (1..31)
  has_event_h                             (0/1: co su kien khong)
  snap_CA_h                               (0/1)
  sell_price_h                            (gia ban ngay dich)
  price_ratio_h                           (sell_price_h / sell_price_t)

Toi uu hieu nang:
- Them cot d_num (int) de filter nhanh thay vi string set.
- build_features_for_origin dung numpy vectorized thay vi python loop per SKU.
- build_train_dataset dung d_num filter O(1) thay vi set membership.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ma hoa nganh hang co dinh (khong phu thuoc thu tu trong du lieu)
CAT_ENCODING: Dict[str, int] = {"FOODS": 0, "HOBBIES": 1, "HOUSEHOLD": 2}
FEATURE_COLS: List[str] = [
    "lag_t0", "lag_t6", "lag_t13", "lag_t27",
    "roll_mean_7", "roll_std_7",
    "roll_mean_28", "roll_std_28",
    "roll_mean_56", "roll_std_56",
    "zero_ratio_28",
    "days_since_last_sale",
    "cat_encoded",
    "horizon_h",
    "weekday_h", "month_h", "dayofmonth_h",
    "has_event_h",
    "snap_CA_h",
    "sell_price_h",
    "price_ratio_h",
]


def _safe_std(arr: np.ndarray) -> float:
    """Tinh std khong nan; tra 0.0 neu mang co 1 phan tu."""
    if len(arr) < 2:
        return 0.0
    return float(np.std(arr, ddof=0))


def _ensure_d_num(df: pd.DataFrame) -> pd.DataFrame:
    """Them cot d_num (int) neu chua co. d_num = so nguyen tu cot d ('d_123' -> 123)."""
    if "d_num" not in df.columns:
        df = df.copy()
        df["d_num"] = df["d"].str.removeprefix("d_").astype(np.int32)
    return df


def build_features_for_origin(
    history_df: pd.DataFrame,
    future_context_df: pd.DataFrame,
    origin_day: int,
    horizon: int = 28,
) -> pd.DataFrame:
    """Xay dung bang feature cho 1 origin (tat ca SKU, tat ca h=1..28).

    history_df: DataFrame hang ngay TU d_1 DEN d_{origin_day} (da loc).
                Cot can co: ['date', 'item_id', 'cat_id', 'sales', 'd'].
    future_context_df: DataFrame chua thong tin ngay du bao (d_{origin+1}..d_{origin+28}).
                Cot can co: ['date', 'item_id', 'sell_price', 'wday', 'month',
                             'event_name_1', 'snap_CA'].
    origin_day: Index ngay origin (so nguyen, vi du 1577).
    horizon: So ngay du bao (mac dinh 28).

    Tra ve: DataFrame voi cac cot FEATURE_COLS + ['item_id']
    """
    # Sap xep theo ngay de dam bao thu tu
    hist = history_df.sort_values(by=["item_id", "date"]).copy()
    hist["date"] = pd.to_datetime(hist["date"])

    # Gia bao gom gia tai origin t (de tinh price_ratio)
    price_at_origin: Dict[str, float] = (
        hist.dropna(subset=["sell_price"])
        .groupby("item_id")["sell_price"]
        .last()
        .to_dict()
    )

    # Chuan bi future_context: sort va danh so horizon
    fctx = future_context_df.copy()
    fctx["date"] = pd.to_datetime(fctx["date"])
    fctx = fctx.sort_values(["item_id", "date"]).reset_index(drop=True)
    fctx["horizon_h"] = fctx.groupby("item_id").cumcount() + 1
    fctx = fctx[fctx["horizon_h"] <= horizon]
    fctx["has_event_h"] = (
        fctx["event_name_1"].notna() & (fctx["event_name_1"] != "")
    ).astype(float)
    fctx["dayofmonth_h"] = pd.to_datetime(fctx["date"]).dt.day.astype(float)

    records = []

    for item_id, grp in hist.groupby("item_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        sales_arr = grp["sales"].values.astype(float)
        n = len(sales_arr)
        cat_code = CAT_ENCODING.get(grp["cat_id"].iloc[0], -1)
        price_t = float(price_at_origin.get(item_id, np.nan))

        # Feature nhom A: tinh toan 1 lan tai origin
        def _lag(offset: int) -> float:
            idx = n - 1 - offset
            return float(sales_arr[idx]) if idx >= 0 else 0.0

        w7 = sales_arr[max(0, n - 7):]
        w28 = sales_arr[max(0, n - 28):]
        w56 = sales_arr[max(0, n - 56):]

        feat_a = {
            "lag_t0": _lag(0),
            "lag_t6": _lag(6),
            "lag_t13": _lag(13),
            "lag_t27": _lag(27),
            "roll_mean_7": float(np.mean(w7)) if len(w7) > 0 else 0.0,
            "roll_std_7": _safe_std(w7),
            "roll_mean_28": float(np.mean(w28)) if len(w28) > 0 else 0.0,
            "roll_std_28": _safe_std(w28),
            "roll_mean_56": float(np.mean(w56)) if len(w56) > 0 else 0.0,
            "roll_std_56": _safe_std(w56),
            "zero_ratio_28": float(np.mean(w28 == 0)) if len(w28) > 0 else 1.0,
            "days_since_last_sale": float(
                n - 1 - int(np.where(sales_arr > 0)[0][-1])
                if (sales_arr > 0).any() else n
            ),
            "cat_encoded": float(cat_code),
        }

        # Feature nhom B: lay tu future_context_df cho tung ngay dich
        ctx_item = fctx[fctx["item_id"] == item_id].reset_index(drop=True)

        for _, ctx_row in ctx_item.iterrows():
            h = int(ctx_row["horizon_h"])
            sell_price_h = float(ctx_row.get("sell_price", price_t))
            if np.isnan(sell_price_h):
                sell_price_h = price_t if not np.isnan(price_t) else 0.0
            price_ratio = sell_price_h / price_t if (price_t and price_t > 0) else 1.0

            feat_b = {
                "horizon_h": float(h),
                "weekday_h": float(ctx_row.get("wday", 0)),
                "month_h": float(ctx_row.get("month", 0)),
                "dayofmonth_h": float(ctx_row["dayofmonth_h"]),
                "has_event_h": float(ctx_row["has_event_h"]),
                "snap_CA_h": float(ctx_row.get("snap_CA", 0)),
                "sell_price_h": sell_price_h,
                "price_ratio_h": price_ratio,
            }

            row = {"item_id": item_id, **feat_a, **feat_b}
            records.append(row)

    df_feat = pd.DataFrame(records, columns=["item_id"] + FEATURE_COLS)
    return df_feat


def build_train_dataset(
    daily_df: pd.DataFrame,
    train_origins: List[int],
    horizon: int = 28,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Xay dung tap huan luyen LightGBM tu nhieu origin.

    Toi uu: dung d_num (int) de filter nhanh, khong tao string set.
    Moi origin o trong train_origins:
        - history = d_1..d_o (d_num <= o)
        - target = sales(d_{o+1}..d_{o+28})

    Tra ve: (X_df, y_arr) de truyen vao lgb.Dataset.
    """
    # Them cot d_num neu chua co
    daily_df = _ensure_d_num(daily_df)

    X_parts: List[pd.DataFrame] = []
    y_parts: List[np.ndarray] = []

    max_origin = max(train_origins)
    # Pre-sort 1 lan de cac vong lap sau nhanh hon
    df_sorted = daily_df.sort_values(["item_id", "d_num"]).reset_index(drop=True)

    for origin in train_origins:
        # Filter nhanh bang so nguyen thay vi string set
        history_df = df_sorted[df_sorted["d_num"] <= origin]
        future_ctx = df_sorted[
            (df_sorted["d_num"] > origin) & (df_sorted["d_num"] <= origin + horizon)
        ][["date", "item_id", "sell_price", "wday", "month",
           "event_name_1", "snap_CA", "sales", "d_num"]].copy()

        if future_ctx.empty:
            continue  # origin sap ra ngoai du lieu

        X = build_features_for_origin(
            history_df=history_df,
            future_context_df=future_ctx,
            origin_day=origin,
            horizon=horizon,
        )

        # Lay target theo dung (item_id, h) tuong ung
        future_ctx_sorted = future_ctx.sort_values(["item_id", "date"]).copy()
        future_ctx_sorted["horizon_h"] = (
            future_ctx_sorted.groupby("item_id").cumcount() + 1
        )

        y_merged = X[["item_id", "horizon_h"]].merge(
            future_ctx_sorted[["item_id", "horizon_h", "sales"]],
            on=["item_id", "horizon_h"],
            how="left",
        )
        y = y_merged["sales"].values.astype(float)

        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        return pd.DataFrame(columns=["item_id"] + FEATURE_COLS), np.array([])

    X_all = pd.concat(X_parts, ignore_index=True)
    y_all = np.concatenate(y_parts)
    return X_all, y_all
