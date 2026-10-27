"""Module lua chon tap con 60 SKU cho nghien cuu.

Quy tac bat buoc:
1. Chi su dung thong tin tai cua hang duoc chi dinh (mac dinh: CA_1).
2. Ung vien phai mo ban tu tuan dau tien (wm_yr_wk == 11101).
3. Moi tieu chi xep hang va lua chon chi dua tren giai doan train (d_1..d_1577),
   tuyet doi khong su dung bat ky thong tin nao sau d_1577.
4. Trong moi nganh hang, lay 20 SKU theo 20 phan vi deu nhau cua tong doanh so.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml


def load_data_config(config_path: Union[str, Path] = "configs/data_config.yaml") -> dict:
    """Doc file cau hinh du lieu."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file cau hinh: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_candidate_skus_from_prices(
    prices_df: pd.DataFrame,
    store_id: str = "CA_1",
    initial_wm_yr_wk: int = 11101,
) -> List[str]:
    """Loc cac SKU co ban tai store_id ngay tu tuan dau tien.
    
    Chi kiem tra su ton tai cua gia tai initial_wm_yr_wk (11101),
    khong nhin vao bat ky tuan nao sau do.
    """
    first_week_prices = prices_df[
        (prices_df["store_id"] == store_id)
        & (prices_df["wm_yr_wk"] == initial_wm_yr_wk)
    ]
    candidate_skus = sorted(first_week_prices["item_id"].unique().tolist())
    return candidate_skus


def select_quantiles_from_group(
    group_df: pd.DataFrame,
    num_items: int = 20,
    sales_col: str = "total_sales_train",
    tie_breaker_col: str = "item_id",
) -> pd.DataFrame:
    """Chon num_items tu group_df theo cac phan vi deu nhau.
    
    Sap xep theo sales_col tang dan; khi trung gia tri thi sap theo tie_breaker_col tang dan.
    Lay cac chi so: int(round(i * (N - 1) / (num_items - 1))) voi i = 0..(num_items - 1).
    """
    sorted_df = group_df.sort_values(
        by=[sales_col, tie_breaker_col],
        ascending=[True, True],
    ).reset_index(drop=True)
    
    n_candidates = len(sorted_df)
    if n_candidates < num_items:
        raise ValueError(
            f"So luong ung vien ({n_candidates}) it hon so SKU can chon ({num_items})"
        )
    
    indices = [
        int(round(i * (n_candidates - 1) / (num_items - 1)))
        for i in range(num_items)
    ]
    
    selected = sorted_df.iloc[indices].copy()
    selected["quantile_rank"] = range(num_items)
    selected["quantile_pct"] = [round(i / (num_items - 1) * 100, 2) for i in range(num_items)]
    return selected


def select_subset_skus(
    sales_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Thuc hien toan bo quy trinh lua chon 60 SKU theo dung luat nghien cuu.
    
    Tham so:
        sales_df: DataFrame chua sales_train_evaluation hoac sales_train_validation.
        prices_df: DataFrame chua sell_prices.
        config: Dictionary cau hinh (neu None se tu dong doc tu configs/data_config.yaml).
        
    Tra ve:
        pd.DataFrame chua danh sach 60 SKU kem metadata lua chon.
    """
    if config is None:
        config = load_data_config()
        
    store_id = config.get("store_id", "CA_1")
    categories = config.get("categories", ["FOODS", "HOBBIES", "HOUSEHOLD"])
    num_skus_per_cat = config.get("num_skus_per_category", 20)
    last_train_day = config.get("timeline", {}).get("last_train_day", 1577)
    initial_wm_yr_wk = config.get("sku_selection", {}).get("initial_wm_yr_wk", 11101)
    
    # 1. Loc ung vien: Chi SKU tai store_id co mat tu tuan 11101
    candidate_items = get_candidate_skus_from_prices(
        prices_df=prices_df,
        store_id=store_id,
        initial_wm_yr_wk=initial_wm_yr_wk,
    )
    
    # 2. Loc sales cho store_id va cac ung vien
    store_sales = sales_df[
        (sales_df["store_id"] == store_id)
        & (sales_df["item_id"].isin(candidate_items))
    ].copy()
    
    # 3. Tinh tong doanh so chi tren d_1..d_last_train_day
    # TUYET DOI KHONG dung bat ky cot d nao sau last_train_day
    train_day_cols = [f"d_{i}" for i in range(1, last_train_day + 1)]
    available_train_cols = [col for col in train_day_cols if col in store_sales.columns]
    
    if len(available_train_cols) != last_train_day:
        raise ValueError(
            f"Du lieu sales thieu cac cot train. Yeu cau du {last_train_day} cot, "
            f"thuc te chi co {len(available_train_cols)} cot."
        )
        
    store_sales["total_sales_train"] = store_sales[available_train_cols].sum(axis=1)
    
    # 4. Chon phan vi trong tung nganh hang
    selected_dfs = []
    for cat in categories:
        cat_group = store_sales[store_sales["cat_id"] == cat]
        selected_cat = select_quantiles_from_group(
            group_df=cat_group,
            num_items=num_skus_per_cat,
            sales_col="total_sales_train",
            tie_breaker_col="item_id",
        )
        selected_dfs.append(selected_cat)
        
    result_df = pd.concat(selected_dfs, ignore_index=True)
    
    cols_to_keep = [
        "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "total_sales_train", "quantile_rank", "quantile_pct",
    ]
    return result_df[cols_to_keep]


if __name__ == "__main__":
    import sys
    print("Dang chay thu nghiem subset selection...")
    sales_path = Path("data/raw/sales_train_evaluation.csv")
    prices_path = Path("data/raw/sell_prices.csv")
    
    if sales_path.exists() and prices_path.exists():
        sales = pd.read_csv(sales_path)
        prices = pd.read_csv(prices_path)
        subset_df = select_subset_skus(sales, prices)
        print(f"Chon thanh cong {len(subset_df)} SKU:")
        print(subset_df["cat_id"].value_counts())
        print("\nTop 5 SKU doanh so cao nhat:")
        print(subset_df.sort_values(by="total_sales_train", ascending=False)[["item_id", "cat_id", "total_sales_train"]].head())
        print("\nTop 5 SKU doanh so thap nhat (nhu cau gian doan):")
        print(subset_df.sort_values(by="total_sales_train", ascending=True)[["item_id", "cat_id", "total_sales_train"]].head())
    else:
        print("Khong tim thay file du lieu thô trong data/raw/.")
