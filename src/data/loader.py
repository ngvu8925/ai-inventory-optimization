"""Module doc va bien doi du lieu M5 thanh dinh dang dai (long format).

Xu ly:
1. Doc calendar.csv, sell_prices.csv, sales_train_evaluation.csv.
2. Loc theo store_id va tap 60 SKU da chon.
3. Chuyen sales tu dang rong sang dang dai.
4. Ghep noi voi calendar de lay ngay thuc te, su kien, co SNAP.
5. Ghep noi voi sell_prices de lay gia ban theo tuan.
6. Luu sales_daily.parquet va sales_weekly.parquet.
"""

import sys
from pathlib import Path

# Dam bao import duoc src khi chay script truc tiep
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.subset import load_data_config, select_subset_skus


def load_raw_data(
    raw_dir: Union[str, Path] = "data/raw",
    store_id: str = "CA_1",
    selected_items: Optional[List[str]] = None,
    max_eval_day: int = 1941,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Doc 3 file du lieu tho va loc truoc theo store_id va danh sach SKU.
    
    Khong bao gio sua bat ky file nao trong data/raw/.
    """
    raw_path = Path(raw_dir)
    
    # 1. Doc calendar
    cal_file = raw_path / "calendar.csv"
    if not cal_file.exists():
        raise FileNotFoundError(f"Khong tim thay {cal_file}")
    calendar_df = pd.read_csv(cal_file)
    
    # Gioi han calendar toi ngay max_eval_day (mac dinh d_1941)
    eval_d_cols = set(f"d_{i}" for i in range(1, max_eval_day + 1))
    calendar_df = calendar_df[calendar_df["d"].isin(eval_d_cols)].copy()
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    
    # 2. Doc sales
    sales_file = raw_path / "sales_train_evaluation.csv"
    if not sales_file.exists():
        raise FileNotFoundError(f"Khong tim thay {sales_file}")
    
    # Chi doc cac dong thuoc store_id va nam trong selected_items (neu co)
    sales_df = pd.read_csv(sales_file)
    sales_df = sales_df[sales_df["store_id"] == store_id].copy()
    if selected_items is not None:
        sales_df = sales_df[sales_df["item_id"].isin(selected_items)].copy()
        
    # 3. Doc prices
    prices_file = raw_path / "sell_prices.csv"
    if not prices_file.exists():
        raise FileNotFoundError(f"Khong tim thay {prices_file}")
    prices_df = pd.read_csv(prices_file)
    prices_df = prices_df[prices_df["store_id"] == store_id].copy()
    if selected_items is not None:
        prices_df = prices_df[prices_df["item_id"].isin(selected_items)].copy()
        
    return sales_df, calendar_df, prices_df


def build_daily_sales(
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    max_eval_day: int = 1941,
) -> pd.DataFrame:
    """Chuyen doi sales sang dang dai va ghep voi lich cung nhu gia ban.
    
    Tra ve DataFrame dang dai gom cac cot:
        date, item_id, dept_id, cat_id, store_id, state_id, d, sales,
        wm_yr_wk, weekday, wday, month, year, event_name_1, event_type_1,
        event_name_2, event_type_2, snap_CA, sell_price.
    """
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    val_vars = [f"d_{i}" for i in range(1, max_eval_day + 1)]
    
    # Kiem tra cac cot co day du khong
    missing_cols = [c for c in val_vars if c not in sales_df.columns]
    if missing_cols:
        raise ValueError(f"Sales DataFrame thieu cac cot ngay: {missing_cols[:5]}...")
        
    # 1. Melt sang dang dai
    melted = pd.melt(
        sales_df,
        id_vars=id_vars,
        value_vars=val_vars,
        var_name="d",
        value_name="sales",
    )
    melted["sales"] = melted["sales"].astype(np.int32)
    
    # 2. Ghep voi calendar de lay thong tin thoi gian va su kien
    cal_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA",
    ]
    daily = melted.merge(calendar_df[cal_cols], on="d", how="left")
    
    # 3. Ghep voi sell_prices theo (store_id, item_id, wm_yr_wk)
    daily = daily.merge(
        prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    
    # Sap xep va dinh dang lai
    daily = daily.sort_values(by=["item_id", "date"]).reset_index(drop=True)
    
    final_cols = [
        "date", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d",
        "sales", "sell_price", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA",
    ]
    return daily[final_cols]


def aggregate_weekly_sales(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Gop du lieu ban hang tu hang ngay sang hang tuan (theo wm_yr_wk).
    
    - Doanh so tuan = Tong sales cac ngay trong tuan.
    - Gia ban tuan = Gia ban trong tuan do (gia M5 khong doi trong 1 tuan).
    - Ngay bat dau tuan = Ngay som nhat cua tuan do.
    """
    # Xac dinh xem trong tuan co su kien hay khong
    has_event = (
        daily_df["event_name_1"].notna() | daily_df["event_name_2"].notna()
    ).astype(int)
    
    temp_df = daily_df.copy()
    temp_df["has_event"] = has_event
    
    group_cols = [
        "store_id", "item_id", "dept_id", "cat_id", "state_id", "wm_yr_wk",
    ]
    
    weekly = temp_df.groupby(group_cols, as_index=False).agg(
        date=("date", "min"),
        sales=("sales", "sum"),
        sell_price=("sell_price", "mean"),
        days_in_week=("date", "count"),
        event_days_count=("has_event", "sum"),
        snap_days_count=("snap_CA", "sum"),
    )
    
    # Sap xep theo item_id va date
    weekly = weekly.sort_values(by=["item_id", "date"]).reset_index(drop=True)
    
    cols = [
        "date", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "wm_yr_wk", "sales", "sell_price", "days_in_week",
        "event_days_count", "snap_days_count",
    ]
    return weekly[cols]


def process_and_save(
    config_path: Union[str, Path] = "configs/data_config.yaml",
    output_dir: Union[str, Path] = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chay toan bo quy trinh chon SKU, xu ly du lieu va luu file Parquet."""
    config = load_data_config(config_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_dir = Path("data/raw")
    store_id = config.get("store_id", "CA_1")
    
    # 1. Chon 60 SKU
    print("--- Buoc 1: Chon 60 SKU theo quy tac nghien cuu ---")
    sales_raw = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    prices_raw = pd.read_csv(raw_dir / "sell_prices.csv")
    
    subset_df = select_subset_skus(sales_raw, prices_raw, config=config)
    selected_skus = subset_df["item_id"].tolist()
    print(f"Da chon {len(selected_skus)} SKU ({len(subset_df[subset_df['cat_id']=='FOODS'])} FOODS, "
          f"{len(subset_df[subset_df['cat_id']=='HOBBIES'])} HOBBIES, "
          f"{len(subset_df[subset_df['cat_id']=='HOUSEHOLD'])} HOUSEHOLD).")
          
    # Luu danh sach SKU vao configs/selected_skus.csv de tien tra cuu
    subset_df.to_csv("configs/selected_skus.csv", index=False)
    print("Da luu danh sach 60 SKU vao configs/selected_skus.csv")
    
    # 2. Doc va loc du lieu cho 60 SKU
    print("--- Buoc 2: Doc va loc du lieu chi tiet ---")
    sales_filtered, cal_df, prices_filtered = load_raw_data(
        raw_dir=raw_dir,
        store_id=store_id,
        selected_items=selected_skus,
        max_eval_day=config["timeline"]["last_holdout_day"],
    )
    
    # 3. Tao daily sales
    print("--- Buoc 3: Chuyen sales sang dang dai (daily) ---")
    daily_df = build_daily_sales(
        sales_df=sales_filtered,
        calendar_df=cal_df,
        prices_df=prices_filtered,
        max_eval_day=config["timeline"]["last_holdout_day"],
    )
    daily_parquet_path = out_dir / "sales_daily.parquet"
    daily_df.to_parquet(daily_parquet_path, index=False)
    print(f"Da luu {len(daily_df):,} dong vao {daily_parquet_path}")
    
    # 4. Tao weekly sales
    print("--- Buoc 4: Gop sang dang tuan (weekly) ---")
    weekly_df = aggregate_weekly_sales(daily_df)
    weekly_parquet_path = out_dir / "sales_weekly.parquet"
    weekly_df.to_parquet(weekly_parquet_path, index=False)
    print(f"Da luu {len(weekly_df):,} dong vao {weekly_parquet_path}")
    
    return subset_df, daily_df, weekly_df


def load_processed_daily(data_path: Union[str, Path] = "data/processed/sales_daily.parquet") -> pd.DataFrame:
    """Helper de cac module khac doc du lieu daily."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}. Hay chay process_and_save() truoc.")
    return pd.read_parquet(path)


def load_processed_weekly(data_path: Union[str, Path] = "data/processed/sales_weekly.parquet") -> pd.DataFrame:
    """Helper de cac module khac doc du lieu weekly."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file: {path}. Hay chay process_and_save() truoc.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    process_and_save()
