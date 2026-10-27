"""Kiem thu cho module loader va subset (Phase 2).

Kiem tra:
1. Dung 60 SKU, dung 20 SKU moi nganh hang (FOODS, HOBBIES, HOUSEHOLD).
2. Tinh doc lap voi du lieu hold-out: Tap SKU duoc chon khong thay doi
   khi xoa toan bo du lieu sau d_1577 hoac sua doi hold-out.
3. Tinh toan ven cua sales_daily.parquet: khong co ngay trung, khong thieu gia,
   du 1.941 ngay lien tuc cho moi SKU.
4. Tinh toan ven cua sales_weekly.parquet: khong trung lap tuan,
   tong doanh so tuan bang tong doanh so ngay.
5. Tinh don dieu cua phan vi doanh so trong tap train.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import (
    load_processed_daily,
    load_processed_weekly,
    load_raw_data,
)
from src.data.subset import (
    load_data_config,
    select_subset_skus,
)


@pytest.fixture(scope="module")
def data_config():
    """Fixture doc cau hinh."""
    return load_data_config("configs/data_config.yaml")


@pytest.fixture(scope="module")
def raw_data():
    """Fixture doc du lieu tho (chi tai khi chay test)."""
    sales_file = Path("data/raw/sales_train_evaluation.csv")
    prices_file = Path("data/raw/sell_prices.csv")
    if not (sales_file.exists() and prices_file.exists()):
        pytest.skip("Chua co du lieu tho trong data/raw/, bo qua test tich hop")
    sales = pd.read_csv(sales_file)
    prices = pd.read_csv(prices_file)
    return sales, prices


def test_subset_sku_counts_and_categories(raw_data, data_config):
    """Kiem tra tap con duoc chon dung 60 SKU, moi nganh hang dung 20 SKU."""
    sales, prices = raw_data
    subset_df = select_subset_skus(sales, prices, config=data_config)
    
    # 1. Tong so luong 60 SKU
    assert len(subset_df) == 60, f"Ky vong 60 SKU, thuc te co {len(subset_df)}"
    assert subset_df["item_id"].nunique() == 60, "Co SKU bi trung lap trong tap chon"
    
    # 2. Moi nganh hang dung 20 SKU
    cat_counts = subset_df["cat_id"].value_counts().to_dict()
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        assert cat_counts.get(cat, 0) == 20, f"Nganh hang {cat} khong du 20 SKU: {cat_counts.get(cat)}"


def test_subset_independent_of_holdout(raw_data, data_config):
    """Kiem tra quy tac bat buoc: Xoa toan bo du lieu sau d_1577 thi ket qua KHONG doi."""
    sales, prices = raw_data
    
    # Chon SKU tren du lieu day du
    subset_full = select_subset_skus(sales, prices, config=data_config)
    
    # Tao ban sao sales bi cat bo toan bo cot tu d_1578 den d_1941
    cols_to_drop = [f"d_{i}" for i in range(1578, 1942) if f"d_{i}" in sales.columns]
    sales_truncated = sales.drop(columns=cols_to_drop).copy()
    
    # Tao ban sao prices bi xoa toan bo gia cua cac tuan sau tuan 11517 (sau d_1577)
    prices_truncated = prices[prices["wm_yr_wk"] <= 11517].copy()
    
    # Chon SKU tren du lieu bi cat
    subset_truncated = select_subset_skus(sales_truncated, prices_truncated, config=data_config)
    
    # Kiem tra 2 danh sach SKU va thu tu phan vi phai giong nhau 100%
    pd.testing.assert_frame_equal(
        subset_full[["item_id", "cat_id", "total_sales_train", "quantile_rank"]],
        subset_truncated[["item_id", "cat_id", "total_sales_train", "quantile_rank"]],
        check_dtype=True,
    )


def test_quantile_sales_monotonicity(raw_data, data_config):
    """Kiem tra doanh so xep theo phan vi phai tang don dieu trong tung nganh hang."""
    sales, prices = raw_data
    subset_df = select_subset_skus(sales, prices, config=data_config)
    
    for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]:
        cat_df = subset_df[subset_df["cat_id"] == cat].sort_values(by="quantile_rank")
        sales_series = cat_df["total_sales_train"].values
        # Moi phan vi sau phai co total_sales >= phan vi truoc
        assert (sales_series[1:] >= sales_series[:-1]).all(), (
            f"Doanh so khong tang don dieu trong nganh hang {cat}: {sales_series}"
        )


def test_daily_sales_integrity():
    """Kiem tra toan ven file sales_daily.parquet."""
    daily_path = Path("data/processed/sales_daily.parquet")
    if not daily_path.exists():
        pytest.skip("Chua co file sales_daily.parquet, bo qua test")
        
    df = load_processed_daily(daily_path)
    
    # 1. So SKU va so dong
    assert df["item_id"].nunique() == 60
    assert len(df) == 60 * 1941, f"Ky vong {60 * 1941} dong, thuc te {len(df)}"
    
    # 2. Khong co ngay trung lap cho tung SKU
    dup_counts = df.duplicated(subset=["item_id", "date"]).sum()
    assert dup_counts == 0, f"Phat hien {dup_counts} ban ghi trung lap (item_id, date)"
    
    # 3. Moi chuoi SKU phai co du 1941 ngay lien tuc
    days_per_sku = df.groupby("item_id")["date"].count()
    assert (days_per_sku == 1941).all(), "Co SKU khong du 1.941 ngay"
    
    # 4. Gia tri sales: khong NaN, >= 0
    assert df["sales"].isna().sum() == 0, "Cot sales co gia tri NaN"
    assert (df["sales"] >= 0).all(), "Cot sales co gia tri am"
    
    # 5. Gia ban sell_price: khong NaN, > 0
    assert df["sell_price"].isna().sum() == 0, "Cot sell_price co gia tri NaN"
    assert (df["sell_price"] > 0).all(), "Cot sell_price co gia tri <= 0"


def test_weekly_sales_integrity():
    """Kiem tra toan ven file sales_weekly.parquet."""
    weekly_path = Path("data/processed/sales_weekly.parquet")
    daily_path = Path("data/processed/sales_daily.parquet")
    if not (weekly_path.exists() and daily_path.exists()):
        pytest.skip("Chua co file parquet, bo qua test")
        
    w_df = load_processed_weekly(weekly_path)
    d_df = load_processed_daily(daily_path)
    
    # 1. So SKU va so tuan
    assert w_df["item_id"].nunique() == 60
    assert len(w_df) == 60 * 278, f"Ky vong {60 * 278} dong tuan, thuc te {len(w_df)}"
    
    # 2. Khong trung tuan cho tung SKU
    dup_weeks = w_df.duplicated(subset=["item_id", "wm_yr_wk"]).sum()
    assert dup_weeks == 0, f"Phat hien {dup_weeks} ban ghi trung tuan"
    
    # 3. Tong doanh so tuan phai khop voi tong doanh so ngay theo tung SKU
    daily_sum = d_df.groupby("item_id")["sales"].sum()
    weekly_sum = w_df.groupby("item_id")["sales"].sum()
    pd.testing.assert_series_equal(daily_sum, weekly_sum, check_names=False)
