# Forecasting models and Phase 4 configuration

## Kết quả Chronos-T5-Tiny cũ: chỉ tham khảo, chưa duyệt, không dùng trong bảng so sánh

Các file `chronos_*.csv` lưu kết quả Chronos-T5-Tiny cũ và được giữ lại để kiểm tra/audit. Đây là kết quả chưa được duyệt; không dùng chúng trong `model_comparison_*.csv`. Khi tạo lại bảng, chỉ lấy baseline, LightGBM và ba biến thể Chronos hiện tại (Chronos-2 History, Chronos-2 Exogenous, Chronos-Bolt-Small History).

## Evaluation protocol

- The three Chronos variants are zero-shot. `fit()` stores history only; no fine-tuning or hold-out tuning is performed.
- All models use the existing rolling-origin configuration in `configs/backtest.yaml`: 13 non-overlapping folds, 60 CA_1 SKUs, 28 days per fold, origins `1577 + 28k` (`k=0..12`), seed 42.
- Each Chronos variant receives at most the latest 1,024 observations ending at its origin. Future sales are joined only after forecasting for evaluation.
- Point forecast is `forecast_q50`, the median (quantile 0.5). Negative point and quantile forecasts are clipped to zero; values are not rounded.
- The official Chronos-2 and Chronos-Bolt `predict_quantiles` implementations return q0.5 in their point field (named `mean` by the API). We explicitly use the q0.5 column for point metrics. See the [Chronos-2 pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py) and [Chronos-Bolt pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos_bolt.py).

## Variants

| Variant | Checkpoint | Input |
|---|---|---|
| `bolt_history` | `amazon/chronos-bolt-small` | Sales history only |
| `chronos2_history` | `amazon/chronos-2` | Sales history only |
| `chronos2_exogenous` | `amazon/chronos-2` | Sales history plus known past and future covariates |

The Chronos-2 variants share one downloaded checkpoint. Both checkpoints are Apache-2.0. The selected cards report 191 MB for Bolt-Small and 478 MB for Chronos-2. See the [Bolt-Small card](https://huggingface.co/amazon/chronos-bolt-small) and [Chronos-2 card](https://huggingface.co/amazon/chronos-2).

## Exogenous feature definition

`chronos2_exogenous` uses the same target-date feature set as LightGBM:

- `weekday_h` (`wday`), `month_h`, and `dayofmonth_h`;
- `has_event_h` (1 when `event_name_1` is present), and `snap_CA_h`;
- `sell_price_h`;
- `price_ratio_h = sell_price_h / sell_price_at_origin`.

The future 28-day prices, calendar, event flags, and SNAP flags are assumed to be known at the origin, matching the existing LightGBM oracle-feature assumption. `sell_price_at_origin` is the last non-null price available at or before the origin. For Chronos-2 past covariates, the historical `price_ratio` uses the same origin price as denominator; the covariate window is cut to the last 1,024 days. No future sales are included in either past or future covariates. If a target-day price is missing, it falls back to the last known origin price, matching LightGBM's fallback.

## Quantiles and Phase 5 handoff

Prediction files contain the shared point forecast column `forecast`, plus `forecast_q10`, `forecast_q50`, `forecast_q90`, and `forecast_q95` only when the checkpoint natively provides q0.95. Current Chronos-2 provides q0.95; Chronos-Bolt-Small is trained through q0.9, so no q0.95 column is emitted for it. Pinball loss is written to separate `*_pinball_overall.csv` and `*_pinball_by_fold.csv` files. Those values are reserved for probabilistic forecast analysis / Phase 5 and are excluded from point-model comparison.

## Runtime environment

- Windows, Python 3.12.6; CPU: Intel Core i7-1255U (12 logical processors).
- Isolated environment: `.venv-tsfm`; no TSFM package was installed in the main `.venv` by this phase.
- Cache paths: `HF_HOME=D:\hf_cache`, `PIP_CACHE_DIR=D:\pip_cache`.
- Installed versions: `torch 2.15.0+cpu`, `chronos-forecasting 2.3.2`, `transformers 5.22.1`, `accelerate 1.15.0`, `pandas 3.1.0`, `numpy 2.5.4`, `pyarrow 26.0.0`, `PyYAML 6.0.3`, `pytest 9.2.0`, `einops 0.8.2`, and `huggingface-hub 2.4.0`.
- Pilot fold 0 (origin d_1577) completed with 1,680 rows per variant. End-to-end pilot time from `run_one_fold` was 26.34 s for Bolt-Small history, 54.06 s for Chronos-2 history, and 54.62 s for Chronos-2 with covariates. Inference-only times were 26.28 s, 53.96 s, and 54.51 s respectively. Pilot Bias and per-tier q0.5 zero rates are in `results/forecasting/*_pilot_summary.json`.
- Quantile monotonicity check found q0.1 > q0.5 for 10/1,680 Bolt-Small rows (maximum inversion 0.00387) and 1/1,680 Chronos-2 exogenous rows (maximum inversion 0.00092); Chronos-2 history had no crossing. Raw values are preserved, with no monotonicity correction. Further backtesting is paused for review.

## Re-run commands

Set the cache variables before invoking the TSFM environment:

```powershell
$env:HF_HOME = 'D:\hf_cache'
$env:PIP_CACHE_DIR = 'D:\pip_cache'
$env:TEMP = 'D:\pip_cache'
$env:TMP = 'D:\pip_cache'
```

Install only in the isolated environment:

```powershell
python -m venv .venv-tsfm
.venv-tsfm\Scripts\python.exe -m pip install -r requirements-tsfm.txt
```

Run a one-fold pilot for each variant:

```powershell
.venv-tsfm\Scripts\python.exe src/forecasting/run_chronos.py --variant bolt_history --fold 0
.venv-tsfm\Scripts\python.exe src/forecasting/run_chronos.py --variant chronos2_history --fold 0
.venv-tsfm\Scripts\python.exe src/forecasting/run_chronos.py --variant chronos2_exogenous --fold 0
```

The runner defaults to one fold. `--all-folds` explicitly selects all 13 folds and must only be used after approval.
