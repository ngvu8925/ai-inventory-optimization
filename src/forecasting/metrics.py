"""Module tinh toan cac chi so danh gia do chinh xac du bao nhu cau.

Cac chi so ho tro:
1. MAE (Mean Absolute Error)
2. RMSE (Root Mean Squared Error)
3. WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
4. MASE (Mean Absolute Scaled Error = MAE / in_sample_naive_mae)
5. Bias (Mean Signed Error = mean(y_hat - y); duong la du bao thua, am la thieu)

TUYET DOI KHONG DUNG MAPE (vi du lieu co nhieu ngay sales = 0, gay loi chia 0).
"""

from typing import Dict, Optional, Union
import numpy as np


def calc_mae(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tinh MAE: trung binh tri tuyet doi sai so."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        raise ValueError("Mang du lieu rong")
    if len(yt) != len(yp):
        raise ValueError(f"Kich thuoc khong khop: y_true={len(yt)}, y_pred={len(yp)}")
    return float(np.mean(np.abs(yt - yp)))


def calc_rmse(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tinh RMSE: can bac hai trung binh binh phuong sai so."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        raise ValueError("Mang du lieu rong")
    if len(yt) != len(yp):
        raise ValueError(f"Kich thuoc khong khop: y_true={len(yt)}, y_pred={len(yp)}")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def calc_wape(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tinh WAPE: tong |sai so| chia cho tong thuc te.
    
    Neu tong thuc te = 0:
    - Neu tong sai so cung bang 0 (du bao dung 0): WAPE = 0.0.
    - Neu tong sai so > 0: dung va bao loi (khong the tinh chia cho 0).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        raise ValueError("Mang du lieu rong")
    if len(yt) != len(yp):
        raise ValueError(f"Kich thuoc khong khop: y_true={len(yt)}, y_pred={len(yp)}")
        
    sum_abs_err = float(np.sum(np.abs(yt - yp)))
    sum_actual = float(np.sum(yt))
    
    if sum_actual == 0.0:
        if sum_abs_err == 0.0:
            return 0.0
        raise ValueError("WAPE khong xac dinh vi tong sales thuc te bang 0 ma sai so > 0.")
        
    return float(sum_abs_err / sum_actual)


def calc_bias(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> float:
    """Tinh Bias (Mean Signed Error = mean(y_pred - y_true)).
    
    - Bias > 0: Mo hinh co xu huong du bao THUA (over-forecasting).
    - Bias < 0: Mo hinh co xu huong du bao THIEU (under-forecasting).
    - Bias = 0: Du bao khong chech tong the.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        raise ValueError("Mang du lieu rong")
    if len(yt) != len(yp):
        raise ValueError(f"Kich thuoc khong khop: y_true={len(yt)}, y_pred={len(yp)}")
    return float(np.mean(yp - yt))


def calc_mase(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_train_history: Union[np.ndarray, list],
) -> float:
    """Tinh MASE: MAE cua du bao chia cho in-sample MAE cua Naive lag-1 tren tap train.
    
    scale = (1 / (T - 1)) * sum_{t=2}^T |y_train[t] - y_train[t-1]|.
    Neu scale = 0 (chuoi train la hang so), dung va bao loi theo dung yeu cau.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    y_tr = np.asarray(y_train_history, dtype=float)
    
    if len(yt) == 0 or len(y_tr) < 2:
        raise ValueError(f"Du lieu khong du de tinh MASE: y_true={len(yt)}, y_train={len(y_tr)}")
    if len(yt) != len(yp):
        raise ValueError(f"Kich thuoc khong khop: y_true={len(yt)}, y_pred={len(yp)}")
        
    mae_val = calc_mae(yt, yp)
    
    # In-sample naive lag-1 scale
    naive_diffs = np.abs(np.diff(y_tr))
    scale = float(np.mean(naive_diffs))
    
    if scale == 0.0:
        raise ValueError(
            "MASE scale bang 0 vi chuoi lich su huan luyen la hang so "
            "(khong co su thay doi giua cac ngay lien tiep). Dung va bao theo luat."
        )
        
    return float(mae_val / scale)


def calc_metrics_dict(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_train_history: Optional[Union[np.ndarray, list]] = None,
) -> Dict[str, float]:
    """Helper tinh nhanh tat ca cac chi so duoc phep su dung."""
    res = {
        "MAE": calc_mae(y_true, y_pred),
        "RMSE": calc_rmse(y_true, y_pred),
        "WAPE": calc_wape(y_true, y_pred),
        "Bias": calc_bias(y_true, y_pred),
    }
    if y_train_history is not None:
        res["MASE"] = calc_mase(y_true, y_pred, y_train_history)
    return res
