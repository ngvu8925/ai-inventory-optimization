"""Script phan tich kham pha du lieu (EDA) cho 60 SKU tai CA_1.

Luu y bat buoc:
- Phan mo ta va bieu do CHI duoc dung du lieu giai doan train (d_1..d_1577, 2011-01-29 den 2015-05-24).
- Tuyet doi khong dung thong tin hold-out (d_1578..d_1941) de mo ta.
- Cac bieu do duoc luu vao results/eda/.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Union

# Thiet lap matplotlib headless va cache directory an toan
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dam bao import duoc src
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_processed_daily
from src.data.subset import load_data_config


def run_eda(
    data_path: Union[str, Path] = "data/processed/sales_daily.parquet",
    config_path: Union[str, Path] = "configs/data_config.yaml",
    output_dir: Union[str, Path] = "results/eda",
) -> dict:
    """Thuc hien toan bo phan tich EDA va luu bieu do vao results/eda/."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_data_config(config_path)
    last_train_day = config.get("timeline", {}).get("last_train_day", 1577)
    
    # 1. Doc du lieu daily
    df = load_processed_daily(data_path)
    
    # CHI lay giai doan train (d_1..d_1577) cho EDA mo ta
    train_d_set = set(f"d_{i}" for i in range(1, last_train_day + 1))
    train_df = df[df["d"].isin(train_d_set)].copy()
    train_df["date"] = pd.to_datetime(train_df["date"])
    
    print(f"EDA tap train: {len(train_df):,} dong (tu {train_df['date'].min().date()} den {train_df['date'].max().date()})")
    print(f"So SKU: {train_df['item_id'].nunique()}")
    
    # Font va style co ban
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.size"] = 10
    
    # --- Bieu do 1: Xu huong tong va theo nganh hang ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Tong toan bo 60 SKU
    daily_total = train_df.groupby("date")["sales"].sum()
    rolling_28 = daily_total.rolling(28, center=True).mean()
    axes[0].plot(daily_total.index, daily_total.values, color="lightgray", alpha=0.7, label="Tong sales ngay (60 SKU)")
    axes[0].plot(rolling_28.index, rolling_28.values, color="navy", linewidth=2, label="Trung binh truot 28 ngay")
    axes[0].set_title("Xu huong tong doanh so ban hang hang ngay (d_1..d_1577)")
    axes[0].set_ylabel("So luong ban (units)")
    axes[0].legend(loc="upper left")
    
    # Theo tung nganh hang
    cat_daily = train_df.groupby(["date", "cat_id"])["sales"].sum().unstack()
    for cat in cat_daily.columns:
        cat_rolling = cat_daily[cat].rolling(28, center=True).mean()
        axes[1].plot(cat_rolling.index, cat_rolling.values, linewidth=1.8, label=f"{cat} (MA 28d)")
    axes[1].set_title("Xu huong theo nganh hang (MA 28 ngay)")
    axes[1].set_xlabel("Thoi gian (Nam)")
    axes[1].set_ylabel("So luong ban (units)")
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    fig1_path = out_dir / "01_overall_trend.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Luu bieu do 1: {fig1_path}")
    
    # --- Bieu do 2: Mua vu theo thu trong tuan va theo thang ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Thu trong tuan (M5: wday 1=Saturday, 7=Friday; hoac dung weekday name)
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_cat = train_df.groupby(["weekday", "cat_id"])["sales"].mean().unstack().reindex(dow_order)
    dow_cat.plot(kind="bar", ax=axes[0], width=0.8, colormap="viridis")
    axes[0].set_title("Mua vu theo thu trong tuan (Trung binh sales/ngay)")
    axes[0].set_ylabel("Sales trung binh (units)")
    axes[0].set_xlabel("Thu trong tuan")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="Nganh hang")
    
    # Theo thang (1..12)
    month_cat = train_df.groupby(["month", "cat_id"])["sales"].mean().unstack()
    month_cat.plot(kind="line", marker="o", ax=axes[1], linewidth=2, colormap="tab10")
    axes[1].set_title("Mua vu theo thang trong nam")
    axes[1].set_ylabel("Sales trung binh (units)")
    axes[1].set_xlabel("Thang")
    axes[1].set_xticks(range(1, 13))
    axes[1].legend(title="Nganh hang")
    
    plt.tight_layout()
    fig2_path = out_dir / "02_seasonality_dow_month.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Luu bieu do 2: {fig2_path}")
    
    # --- Bieu do 3: Ty le ngay ban bang 0 (% zero-demand) tung SKU ---
    sku_stats = train_df.groupby(["cat_id", "item_id"]).agg(
        zero_ratio=("sales", lambda s: (s == 0).mean()),
        mean_sales=("sales", "mean"),
        total_sales=("sales", "sum"),
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sku_stats_sorted = sku_stats.sort_values(by="zero_ratio", ascending=True).reset_index(drop=True)
    
    cat_colors = {"FOODS": "forestgreen", "HOBBIES": "royalblue", "HOUSEHOLD": "darkorange"}
    bar_colors = [cat_colors.get(c, "gray") for c in sku_stats_sorted["cat_id"]]
    
    bars = ax.bar(range(len(sku_stats_sorted)), sku_stats_sorted["zero_ratio"], color=bar_colors, width=0.8)
    ax.set_title("Ty le ngay khong co luot ban nao (Zero Sales Ratio) cua 60 SKU (d_1..d_1577)")
    ax.set_ylabel("Ty le ngay sales = 0 (0.0 - 1.0)")
    ax.set_xlabel("60 SKU (sap xep tu it so 0 nhat den nhieu so 0 nhat)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Moc 0.5 (50% ngay khong co ban)")
    
    # Chu thich nganh hang
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="forestgreen", label="FOODS (20 SKU)"),
        Patch(facecolor="royalblue", label="HOBBIES (20 SKU)"),
        Patch(facecolor="darkorange", label="HOUSEHOLD (20 SKU)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Nguong 0.5"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    
    plt.tight_layout()
    fig3_path = out_dir / "03_zero_sales_distribution.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Luu bieu do 3: {fig3_path}")
    
    # --- Bieu do 4: Phan bo doanh so (Histogram / Boxplot) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram tong the (log-scale y de thay ro duoi dai)
    sales_values = train_df["sales"].values
    axes[0].hist(sales_values, bins=50, range=(0, 50), color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Phan bo luot ban hang ngay (Log-scale y-axis)")
    axes[0].set_xlabel("So luong ban trong ngay (sales)")
    axes[0].set_ylabel("Tan suat (log scale)")
    axes[0].axvline(0, color="crimson", linestyle=":", label=f"Sales = 0: {(sales_values == 0).mean():.1%}")
    axes[0].legend()
    
    # Boxplot theo nganh hang (loai bo 0 de xem gia tri ban duoc khi co nhu cau)
    non_zero = train_df[train_df["sales"] > 0]
    box_data = [non_zero[non_zero["cat_id"] == cat]["sales"].values for cat in ["FOODS", "HOBBIES", "HOUSEHOLD"]]
    axes[1].boxplot(box_data, tick_labels=["FOODS", "HOBBIES", "HOUSEHOLD"], showfliers=False)
    axes[1].set_title("Phan bo doanh so khi co phat sinh ban hang (sales > 0)")
    axes[1].set_ylabel("Sales (loai bo ngoai lai tren bieu do)")
    axes[1].set_xlabel("Nganh hang")
    
    plt.tight_layout()
    fig4_path = out_dir / "04_sales_distribution.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"Luu bieu do 4: {fig4_path}")
    
    # --- Bieu do 5: Hieu ung bien dong gia ban ---
    # Chon 3 SKU tieu bieu co su thay doi gia ro net nhat
    sku_price_var = train_df.groupby("item_id")["sell_price"].agg(["min", "max", "std"])
    sku_price_var["range"] = sku_price_var["max"] - sku_price_var["min"]
    top_var_skus = sku_price_var.sort_values(by="range", ascending=False).head(3).index.tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, sku in enumerate(top_var_skus):
        sub_sku = train_df[train_df["item_id"] == sku].copy()
        # Gom theo muc gia va tinh sales trung binh
        price_group = sub_sku.groupby("sell_price")["sales"].agg(["mean", "count"]).reset_index()
        price_group = price_group[price_group["count"] >= 7]  # it nhat 7 ngay quan sat
        
        axes[i].scatter(sub_sku["sell_price"], sub_sku["sales"], alpha=0.3, color="teal", s=15)
        if len(price_group) > 1:
            axes[i].plot(price_group["sell_price"], price_group["mean"], color="red", marker="o", linewidth=2, label="Mean Sales")
        axes[i].set_title(f"SKU: {sku}\n({sub_sku['cat_id'].iloc[0]})")
        axes[i].set_xlabel("Gia ban ($)")
        axes[i].set_ylabel("Sales / ngay")
        axes[i].legend(loc="upper right")
        
    plt.tight_layout()
    fig5_path = out_dir / "05_price_effect.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    print(f"Luu bieu do 5: {fig5_path}")
    
    # Thong ke so lieu chi tren tap train (d_1..d_1577), thong nhat ty le dang 0-1
    summary = {
        "train_rows": len(train_df),
        "start_date": str(train_df["date"].min().date()),
        "end_date": str(train_df["date"].max().date()),
        "overall_zero_ratio_train": float((train_df["sales"] == 0).mean()),
        "foods_zero_ratio_train": float((train_df[train_df["cat_id"] == "FOODS"]["sales"] == 0).mean()),
        "hobbies_zero_ratio_train": float((train_df[train_df["cat_id"] == "HOBBIES"]["sales"] == 0).mean()),
        "household_zero_ratio_train": float((train_df[train_df["cat_id"] == "HOUSEHOLD"]["sales"] == 0).mean()),
        "min_sku_zero_ratio": float(sku_stats["zero_ratio"].min()),
        "max_sku_zero_ratio": float(sku_stats["zero_ratio"].max()),
        "mean_daily_sales_train": float(train_df["sales"].mean()),
        "mean_price": float(train_df["sell_price"].mean()),
        "missing_sales": int(train_df["sales"].isna().sum()),
        "missing_prices": int(train_df["sell_price"].isna().sum()),
    }
    return summary


if __name__ == "__main__":
    res = run_eda()
    print("\n--- KET QUA THONG KE EDA CHINH ---")
    for k, v in res.items():
        print(f"  {k}: {v}")
