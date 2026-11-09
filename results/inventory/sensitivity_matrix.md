# Ma trận độ nhạy (kịch bản normal)

Mỗi khối chỉ đổi MỘT tham số; các tham số còn lại ở mức cơ sở.
`Oracle` là cận dưới chi phí (perfect foresight), **không phải mô hình**.

Hai loại khối:

- **cost accounting (no behavior change)**: h, K, p chỉ đổi trọng số
  chi phí. ROP/Q/fill/tổng thiếu/số đơn không đổi; tính lại từ
  `simulation_detail_normal.csv`, không chạy lại mô phỏng. Thứ hạng
  chi phí vẫn có thể đổi vì trọng số chi phí thay đổi.
- **behavior change (ROP/SS/Q)**: service_level đổi z nên đổi SS, ROP
  và hành vi đặt hàng; phải chạy lại mô phỏng.

## holding_cost_per_unit_per_day (h)

Loại: **cost accounting (no behavior change)**.

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 0.001355 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,635.75 | 0.9488 | 2,738.00 |
| 0.001355 | 2 | Baseline_full_history | baseline_full_history | 18,768.95 | 0.9335 | 3,555.14 |
| 0.001355 | 3 | Baseline | baseline | 18,790.73 | 0.9307 | 3,704.57 |
| 0.001355 | 4 | LightGBM | point_forecast | 20,552.45 | 0.9194 | 4,308.16 |
| 0.001355 | 5 | LightGBM | point_forecast_window_sigma | 20,739.36 | 0.9116 | 4,724.49 |
| 0.001355 | 6 | LightGBM | point_forecast_error_sigma | 20,956.87 | 0.9105 | 4,783.45 |
| 0.001355 | 7 | MovingAverage | point_forecast_error_sigma | 22,593.58 | 0.8975 | 5,474.75 |
| 0.001355 | 8 | MovingAverage | point_forecast_window_sigma | 22,634.40 | 0.8961 | 5,550.50 |
| 0.001355 | 9 | MovingAverage | point_forecast | 22,978.02 | 0.8990 | 5,395.00 |
| 0.001355 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 27,714.77 | 0.8681 | 7,048.66 |
| 0.001355 | 11 | Chronos2_Covariates | quantile_forecast | 27,772.37 | 0.8609 | 7,431.04 |
| 0.001355 | 12 | Chronos2_Covariates | point_forecast | 27,801.99 | 0.8728 | 6,797.72 |
| 0.001355 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 27,801.99 | 0.8728 | 6,797.72 |
| 0.001355 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,045.70 | 0.8655 | 7,185.28 |
| 0.001355 | 15 | Chronos2_History | quantile_forecast | 28,322.45 | 0.8556 | 7,716.93 |
| 0.001355 | 16 | Chronos2_History | point_forecast_window_sigma | 28,464.09 | 0.8607 | 7,441.47 |
| 0.001355 | 17 | Chronos2_History | point_forecast | 28,619.66 | 0.8647 | 7,232.66 |
| 0.001355 | 17 | Chronos2_History | quantile_forecast_history_sigma | 28,619.66 | 0.8647 | 7,232.66 |
| 0.001355 | 19 | Chronos2_History | point_forecast_error_sigma | 28,619.94 | 0.8599 | 7,488.68 |
| 0.001355 | 20 | Chronos_Bolt | point_forecast | 29,575.10 | 0.8574 | 7,617.95 |
| 0.001355 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,653.03 | 0.8517 | 7,927.24 |
| 0.001355 | 22 | Chronos_Bolt | point_forecast_error_sigma | 29,790.08 | 0.8510 | 7,959.48 |
| 0.002032 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.002032 | 2 | Baseline | baseline | 19,131.27 | 0.9307 | 3,704.57 |
| 0.002032 | 3 | Baseline_full_history | baseline_full_history | 19,182.97 | 0.9335 | 3,555.14 |
| 0.002032 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 0.002032 | 5 | LightGBM | point_forecast_window_sigma | 21,056.82 | 0.9116 | 4,724.49 |
| 0.002032 | 6 | LightGBM | point_forecast_error_sigma | 21,269.12 | 0.9105 | 4,783.45 |
| 0.002032 | 7 | MovingAverage | point_forecast_error_sigma | 22,940.36 | 0.8975 | 5,474.75 |
| 0.002032 | 8 | MovingAverage | point_forecast_window_sigma | 22,978.34 | 0.8961 | 5,550.50 |
| 0.002032 | 9 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 0.002032 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 27,968.30 | 0.8681 | 7,048.66 |
| 0.002032 | 11 | Chronos2_Covariates | quantile_forecast | 28,012.34 | 0.8609 | 7,431.04 |
| 0.002032 | 12 | Chronos2_Covariates | point_forecast | 28,067.57 | 0.8728 | 6,797.72 |
| 0.002032 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,067.57 | 0.8728 | 6,797.72 |
| 0.002032 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,293.08 | 0.8655 | 7,185.28 |
| 0.002032 | 15 | Chronos2_History | quantile_forecast | 28,559.06 | 0.8556 | 7,716.93 |
| 0.002032 | 16 | Chronos2_History | point_forecast_window_sigma | 28,714.47 | 0.8607 | 7,441.47 |
| 0.002032 | 17 | Chronos2_History | point_forecast_error_sigma | 28,863.54 | 0.8599 | 7,488.68 |
| 0.002032 | 18 | Chronos2_History | point_forecast | 28,879.14 | 0.8647 | 7,232.66 |
| 0.002032 | 18 | Chronos2_History | quantile_forecast_history_sigma | 28,879.14 | 0.8647 | 7,232.66 |
| 0.002032 | 20 | Chronos_Bolt | point_forecast | 29,827.01 | 0.8574 | 7,617.95 |
| 0.002032 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,894.63 | 0.8517 | 7,927.24 |
| 0.002032 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,027.97 | 0.8510 | 7,959.48 |
| 0.002709 | 1 | Oracle (lower bound, not a model) | oracle_reference | 13,055.18 | 0.9488 | 2,738.00 |
| 0.002709 | 2 | Baseline | baseline | 19,471.81 | 0.9307 | 3,704.57 |
| 0.002709 | 3 | Baseline_full_history | baseline_full_history | 19,597.00 | 0.9335 | 3,555.14 |
| 0.002709 | 4 | LightGBM | point_forecast | 21,218.09 | 0.9194 | 4,308.16 |
| 0.002709 | 5 | LightGBM | point_forecast_window_sigma | 21,374.28 | 0.9116 | 4,724.49 |
| 0.002709 | 6 | LightGBM | point_forecast_error_sigma | 21,581.37 | 0.9105 | 4,783.45 |
| 0.002709 | 7 | MovingAverage | point_forecast_error_sigma | 23,287.14 | 0.8975 | 5,474.75 |
| 0.002709 | 8 | MovingAverage | point_forecast_window_sigma | 23,322.29 | 0.8961 | 5,550.50 |
| 0.002709 | 9 | MovingAverage | point_forecast | 23,690.52 | 0.8990 | 5,395.00 |
| 0.002709 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 28,221.84 | 0.8681 | 7,048.66 |
| 0.002709 | 11 | Chronos2_Covariates | quantile_forecast | 28,252.31 | 0.8609 | 7,431.04 |
| 0.002709 | 12 | Chronos2_Covariates | point_forecast | 28,333.15 | 0.8728 | 6,797.72 |
| 0.002709 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,333.15 | 0.8728 | 6,797.72 |
| 0.002709 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,540.47 | 0.8655 | 7,185.28 |
| 0.002709 | 15 | Chronos2_History | quantile_forecast | 28,795.68 | 0.8556 | 7,716.93 |
| 0.002709 | 16 | Chronos2_History | point_forecast_window_sigma | 28,964.86 | 0.8607 | 7,441.47 |
| 0.002709 | 17 | Chronos2_History | point_forecast_error_sigma | 29,107.15 | 0.8599 | 7,488.68 |
| 0.002709 | 18 | Chronos2_History | point_forecast | 29,138.62 | 0.8647 | 7,232.66 |
| 0.002709 | 18 | Chronos2_History | quantile_forecast_history_sigma | 29,138.62 | 0.8647 | 7,232.66 |
| 0.002709 | 20 | Chronos_Bolt | point_forecast | 30,078.92 | 0.8574 | 7,617.95 |
| 0.002709 | 21 | Chronos_Bolt | point_forecast_window_sigma | 30,136.23 | 0.8517 | 7,927.24 |
| 0.002709 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,265.86 | 0.8510 | 7,959.48 |

## shortage_cost_per_unit (p)

Loại: **cost accounting (no behavior change)**.

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 1.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 10,107.47 | 0.9488 | 2,738.00 |
| 1.0 | 2 | Baseline | baseline | 15,426.70 | 0.9307 | 3,704.57 |
| 1.0 | 3 | Baseline_full_history | baseline_full_history | 15,627.83 | 0.9335 | 3,555.14 |
| 1.0 | 4 | LightGBM | point_forecast_window_sigma | 16,332.33 | 0.9116 | 4,724.49 |
| 1.0 | 5 | LightGBM | point_forecast_error_sigma | 16,485.67 | 0.9105 | 4,783.45 |
| 1.0 | 6 | LightGBM | point_forecast | 16,577.11 | 0.9194 | 4,308.16 |
| 1.0 | 7 | MovingAverage | point_forecast_window_sigma | 17,427.84 | 0.8961 | 5,550.50 |
| 1.0 | 8 | MovingAverage | point_forecast_error_sigma | 17,465.61 | 0.8975 | 5,474.75 |
| 1.0 | 9 | MovingAverage | point_forecast | 17,939.27 | 0.8990 | 5,395.00 |
| 1.0 | 10 | Chronos2_Covariates | quantile_forecast | 20,581.30 | 0.8609 | 7,431.04 |
| 1.0 | 11 | Chronos2_History | quantile_forecast | 20,842.13 | 0.8556 | 7,716.93 |
| 1.0 | 12 | Chronos2_Covariates | point_forecast_window_sigma | 20,919.64 | 0.8681 | 7,048.66 |
| 1.0 | 13 | Chronos2_Covariates | point_forecast_error_sigma | 21,107.80 | 0.8655 | 7,185.28 |
| 1.0 | 14 | Chronos2_Covariates | point_forecast | 21,269.85 | 0.8728 | 6,797.72 |
| 1.0 | 14 | Chronos2_Covariates | quantile_forecast_history_sigma | 21,269.85 | 0.8728 | 6,797.72 |
| 1.0 | 16 | Chronos2_History | point_forecast_window_sigma | 21,273.00 | 0.8607 | 7,441.47 |
| 1.0 | 17 | Chronos2_History | point_forecast_error_sigma | 21,374.86 | 0.8599 | 7,488.68 |
| 1.0 | 18 | Chronos2_History | point_forecast | 21,646.48 | 0.8647 | 7,232.66 |
| 1.0 | 18 | Chronos2_History | quantile_forecast_history_sigma | 21,646.48 | 0.8647 | 7,232.66 |
| 1.0 | 20 | Chronos_Bolt | point_forecast_window_sigma | 21,967.39 | 0.8517 | 7,927.24 |
| 1.0 | 21 | Chronos_Bolt | point_forecast_error_sigma | 22,068.49 | 0.8510 | 7,959.48 |
| 1.0 | 22 | Chronos_Bolt | point_forecast | 22,209.06 | 0.8574 | 7,617.95 |
| 2.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 2.0 | 2 | Baseline | baseline | 19,131.27 | 0.9307 | 3,704.57 |
| 2.0 | 3 | Baseline_full_history | baseline_full_history | 19,182.97 | 0.9335 | 3,555.14 |
| 2.0 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 2.0 | 5 | LightGBM | point_forecast_window_sigma | 21,056.82 | 0.9116 | 4,724.49 |
| 2.0 | 6 | LightGBM | point_forecast_error_sigma | 21,269.12 | 0.9105 | 4,783.45 |
| 2.0 | 7 | MovingAverage | point_forecast_error_sigma | 22,940.36 | 0.8975 | 5,474.75 |
| 2.0 | 8 | MovingAverage | point_forecast_window_sigma | 22,978.34 | 0.8961 | 5,550.50 |
| 2.0 | 9 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 2.0 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 27,968.30 | 0.8681 | 7,048.66 |
| 2.0 | 11 | Chronos2_Covariates | quantile_forecast | 28,012.34 | 0.8609 | 7,431.04 |
| 2.0 | 12 | Chronos2_Covariates | point_forecast | 28,067.57 | 0.8728 | 6,797.72 |
| 2.0 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,067.57 | 0.8728 | 6,797.72 |
| 2.0 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,293.08 | 0.8655 | 7,185.28 |
| 2.0 | 15 | Chronos2_History | quantile_forecast | 28,559.06 | 0.8556 | 7,716.93 |
| 2.0 | 16 | Chronos2_History | point_forecast_window_sigma | 28,714.47 | 0.8607 | 7,441.47 |
| 2.0 | 17 | Chronos2_History | point_forecast_error_sigma | 28,863.54 | 0.8599 | 7,488.68 |
| 2.0 | 18 | Chronos2_History | point_forecast | 28,879.14 | 0.8647 | 7,232.66 |
| 2.0 | 18 | Chronos2_History | quantile_forecast_history_sigma | 28,879.14 | 0.8647 | 7,232.66 |
| 2.0 | 20 | Chronos_Bolt | point_forecast | 29,827.01 | 0.8574 | 7,617.95 |
| 2.0 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,894.63 | 0.8517 | 7,927.24 |
| 2.0 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,027.97 | 0.8510 | 7,959.48 |
| 4.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 18,321.47 | 0.9488 | 2,738.00 |
| 4.0 | 2 | Baseline_full_history | baseline_full_history | 26,293.25 | 0.9335 | 3,555.14 |
| 4.0 | 3 | Baseline | baseline | 26,540.41 | 0.9307 | 3,704.57 |
| 4.0 | 4 | LightGBM | point_forecast | 29,501.59 | 0.9194 | 4,308.16 |
| 4.0 | 5 | LightGBM | point_forecast_window_sigma | 30,505.80 | 0.9116 | 4,724.49 |
| 4.0 | 6 | LightGBM | point_forecast_error_sigma | 30,836.02 | 0.9105 | 4,783.45 |
| 4.0 | 7 | MovingAverage | point_forecast_error_sigma | 33,889.86 | 0.8975 | 5,474.75 |
| 4.0 | 8 | MovingAverage | point_forecast_window_sigma | 34,079.34 | 0.8961 | 5,550.50 |
| 4.0 | 9 | MovingAverage | point_forecast | 34,124.27 | 0.8990 | 5,395.00 |
| 4.0 | 10 | Chronos2_Covariates | point_forecast | 41,663.01 | 0.8728 | 6,797.72 |
| 4.0 | 10 | Chronos2_Covariates | quantile_forecast_history_sigma | 41,663.01 | 0.8728 | 6,797.72 |
| 4.0 | 12 | Chronos2_Covariates | point_forecast_window_sigma | 42,065.62 | 0.8681 | 7,048.66 |
| 4.0 | 13 | Chronos2_Covariates | point_forecast_error_sigma | 42,663.64 | 0.8655 | 7,185.28 |
| 4.0 | 14 | Chronos2_Covariates | quantile_forecast | 42,874.42 | 0.8609 | 7,431.04 |
| 4.0 | 15 | Chronos2_History | point_forecast | 43,344.46 | 0.8647 | 7,232.66 |
| 4.0 | 15 | Chronos2_History | quantile_forecast_history_sigma | 43,344.46 | 0.8647 | 7,232.66 |
| 4.0 | 17 | Chronos2_History | point_forecast_window_sigma | 43,597.41 | 0.8607 | 7,441.47 |
| 4.0 | 18 | Chronos2_History | point_forecast_error_sigma | 43,840.90 | 0.8599 | 7,488.68 |
| 4.0 | 19 | Chronos2_History | quantile_forecast | 43,992.92 | 0.8556 | 7,716.93 |
| 4.0 | 20 | Chronos_Bolt | point_forecast | 45,062.91 | 0.8574 | 7,617.95 |
| 4.0 | 21 | Chronos_Bolt | point_forecast_window_sigma | 45,749.11 | 0.8517 | 7,927.24 |
| 4.0 | 22 | Chronos_Bolt | point_forecast_error_sigma | 45,946.93 | 0.8510 | 7,959.48 |

## ordering_cost (K)

Loại: **cost accounting (no behavior change)**.

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 2.5 | 1 | Oracle (lower bound, not a model) | oracle_reference | 9,475.47 | 0.9488 | 2,738.00 |
| 2.5 | 2 | Baseline_full_history | baseline_full_history | 13,767.97 | 0.9335 | 3,555.14 |
| 2.5 | 3 | Baseline | baseline | 13,781.27 | 0.9307 | 3,704.57 |
| 2.5 | 4 | LightGBM | point_forecast | 15,250.27 | 0.9194 | 4,308.16 |
| 2.5 | 5 | LightGBM | point_forecast_window_sigma | 15,729.32 | 0.9116 | 4,724.49 |
| 2.5 | 6 | LightGBM | point_forecast_error_sigma | 15,886.62 | 0.9105 | 4,783.45 |
| 2.5 | 7 | MovingAverage | point_forecast_error_sigma | 17,465.36 | 0.8975 | 5,474.75 |
| 2.5 | 8 | MovingAverage | point_forecast_window_sigma | 17,555.84 | 0.8961 | 5,550.50 |
| 2.5 | 9 | MovingAverage | point_forecast | 17,596.77 | 0.8990 | 5,395.00 |
| 2.5 | 10 | Chronos2_Covariates | point_forecast | 21,230.07 | 0.8728 | 6,797.72 |
| 2.5 | 10 | Chronos2_Covariates | quantile_forecast_history_sigma | 21,230.07 | 0.8728 | 6,797.72 |
| 2.5 | 12 | Chronos2_Covariates | point_forecast_window_sigma | 21,413.30 | 0.8681 | 7,048.66 |
| 2.5 | 13 | Chronos2_Covariates | point_forecast_error_sigma | 21,703.08 | 0.8655 | 7,185.28 |
| 2.5 | 14 | Chronos2_Covariates | quantile_forecast | 21,797.34 | 0.8609 | 7,431.04 |
| 2.5 | 15 | Chronos2_History | point_forecast | 22,061.64 | 0.8647 | 7,232.66 |
| 2.5 | 15 | Chronos2_History | quantile_forecast_history_sigma | 22,061.64 | 0.8647 | 7,232.66 |
| 2.5 | 17 | Chronos2_History | point_forecast_window_sigma | 22,174.47 | 0.8607 | 7,441.47 |
| 2.5 | 18 | Chronos2_History | point_forecast_error_sigma | 22,286.04 | 0.8599 | 7,488.68 |
| 2.5 | 19 | Chronos2_History | quantile_forecast | 22,351.56 | 0.8556 | 7,716.93 |
| 2.5 | 20 | Chronos_Bolt | point_forecast | 22,909.51 | 0.8574 | 7,617.95 |
| 2.5 | 21 | Chronos_Bolt | point_forecast_window_sigma | 23,237.13 | 0.8517 | 7,927.24 |
| 2.5 | 22 | Chronos_Bolt | point_forecast_error_sigma | 23,330.47 | 0.8510 | 7,959.48 |
| 5.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 5.0 | 2 | Baseline | baseline | 19,131.27 | 0.9307 | 3,704.57 |
| 5.0 | 3 | Baseline_full_history | baseline_full_history | 19,182.97 | 0.9335 | 3,555.14 |
| 5.0 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 5.0 | 5 | LightGBM | point_forecast_window_sigma | 21,056.82 | 0.9116 | 4,724.49 |
| 5.0 | 6 | LightGBM | point_forecast_error_sigma | 21,269.12 | 0.9105 | 4,783.45 |
| 5.0 | 7 | MovingAverage | point_forecast_error_sigma | 22,940.36 | 0.8975 | 5,474.75 |
| 5.0 | 8 | MovingAverage | point_forecast_window_sigma | 22,978.34 | 0.8961 | 5,550.50 |
| 5.0 | 9 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 5.0 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 27,968.30 | 0.8681 | 7,048.66 |
| 5.0 | 11 | Chronos2_Covariates | quantile_forecast | 28,012.34 | 0.8609 | 7,431.04 |
| 5.0 | 12 | Chronos2_Covariates | point_forecast | 28,067.57 | 0.8728 | 6,797.72 |
| 5.0 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,067.57 | 0.8728 | 6,797.72 |
| 5.0 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,293.08 | 0.8655 | 7,185.28 |
| 5.0 | 15 | Chronos2_History | quantile_forecast | 28,559.06 | 0.8556 | 7,716.93 |
| 5.0 | 16 | Chronos2_History | point_forecast_window_sigma | 28,714.47 | 0.8607 | 7,441.47 |
| 5.0 | 17 | Chronos2_History | point_forecast_error_sigma | 28,863.54 | 0.8599 | 7,488.68 |
| 5.0 | 18 | Chronos2_History | point_forecast | 28,879.14 | 0.8647 | 7,232.66 |
| 5.0 | 18 | Chronos2_History | quantile_forecast_history_sigma | 28,879.14 | 0.8647 | 7,232.66 |
| 5.0 | 20 | Chronos_Bolt | point_forecast | 29,827.01 | 0.8574 | 7,617.95 |
| 5.0 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,894.63 | 0.8517 | 7,927.24 |
| 5.0 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,027.97 | 0.8510 | 7,959.48 |
| 10.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 19,585.47 | 0.9488 | 2,738.00 |
| 10.0 | 2 | Baseline | baseline | 29,831.27 | 0.9307 | 3,704.57 |
| 10.0 | 3 | Baseline_full_history | baseline_full_history | 30,012.97 | 0.9335 | 3,555.14 |
| 10.0 | 4 | LightGBM | point_forecast_window_sigma | 31,711.82 | 0.9116 | 4,724.49 |
| 10.0 | 5 | LightGBM | point_forecast_error_sigma | 32,034.12 | 0.9105 | 4,783.45 |
| 10.0 | 6 | LightGBM | point_forecast | 32,155.27 | 0.9194 | 4,308.16 |
| 10.0 | 7 | MovingAverage | point_forecast_window_sigma | 33,823.34 | 0.8961 | 5,550.50 |
| 10.0 | 8 | MovingAverage | point_forecast_error_sigma | 33,890.36 | 0.8975 | 5,474.75 |
| 10.0 | 9 | MovingAverage | point_forecast | 34,809.27 | 0.8990 | 5,395.00 |
| 10.0 | 10 | Chronos2_Covariates | quantile_forecast | 40,442.34 | 0.8609 | 7,431.04 |
| 10.0 | 11 | Chronos2_History | quantile_forecast | 40,974.06 | 0.8556 | 7,716.93 |
| 10.0 | 12 | Chronos2_Covariates | point_forecast_window_sigma | 41,078.30 | 0.8681 | 7,048.66 |
| 10.0 | 13 | Chronos2_Covariates | point_forecast_error_sigma | 41,473.08 | 0.8655 | 7,185.28 |
| 10.0 | 14 | Chronos2_Covariates | point_forecast | 41,742.57 | 0.8728 | 6,797.72 |
| 10.0 | 14 | Chronos2_Covariates | quantile_forecast_history_sigma | 41,742.57 | 0.8728 | 6,797.72 |
| 10.0 | 16 | Chronos2_History | point_forecast_window_sigma | 41,794.47 | 0.8607 | 7,441.47 |
| 10.0 | 17 | Chronos2_History | point_forecast_error_sigma | 42,018.54 | 0.8599 | 7,488.68 |
| 10.0 | 18 | Chronos2_History | point_forecast | 42,514.14 | 0.8647 | 7,232.66 |
| 10.0 | 18 | Chronos2_History | quantile_forecast_history_sigma | 42,514.14 | 0.8647 | 7,232.66 |
| 10.0 | 20 | Chronos_Bolt | point_forecast_window_sigma | 43,209.63 | 0.8517 | 7,927.24 |
| 10.0 | 21 | Chronos_Bolt | point_forecast_error_sigma | 43,422.97 | 0.8510 | 7,959.48 |
| 10.0 | 22 | Chronos_Bolt | point_forecast | 43,662.01 | 0.8574 | 7,617.95 |

## service_level

Loại: **behavior change (ROP/SS/Q)**.

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 0.9 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.9 | 2 | Baseline | baseline | 19,137.68 | 0.9264 | 3,933.75 |
| 0.9 | 3 | Baseline_full_history | baseline_full_history | 19,298.55 | 0.9270 | 3,901.41 |
| 0.9 | 4 | LightGBM | point_forecast | 20,939.82 | 0.9130 | 4,646.94 |
| 0.9 | 5 | LightGBM | point_forecast_window_sigma | 21,011.06 | 0.9069 | 4,975.06 |
| 0.9 | 6 | LightGBM | point_forecast_error_sigma | 21,313.97 | 0.9047 | 5,093.52 |
| 0.9 | 7 | MovingAverage | point_forecast_error_sigma | 22,745.92 | 0.8950 | 5,610.00 |
| 0.9 | 8 | MovingAverage | point_forecast_window_sigma | 22,765.91 | 0.8934 | 5,695.00 |
| 0.9 | 9 | MovingAverage | point_forecast | 22,801.99 | 0.8974 | 5,481.75 |
| 0.9 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 28,075.81 | 0.8625 | 7,348.95 |
| 0.9 | 11 | Chronos2_Covariates | point_forecast | 28,095.99 | 0.8668 | 7,118.16 |
| 0.9 | 11 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,095.99 | 0.8668 | 7,118.16 |
| 0.9 | 13 | Chronos2_Covariates | quantile_forecast | 28,433.06 | 0.8532 | 7,844.95 |
| 0.9 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,456.12 | 0.8598 | 7,491.42 |
| 0.9 | 15 | Chronos2_History | point_forecast_window_sigma | 28,815.16 | 0.8556 | 7,715.89 |
| 0.9 | 16 | Chronos2_History | point_forecast | 29,068.10 | 0.8574 | 7,621.55 |
| 0.9 | 16 | Chronos2_History | quantile_forecast_history_sigma | 29,068.10 | 0.8574 | 7,621.55 |
| 0.9 | 18 | Chronos2_History | point_forecast_error_sigma | 29,090.31 | 0.8536 | 7,825.68 |
| 0.9 | 19 | Chronos2_History | quantile_forecast | 29,123.60 | 0.8468 | 8,188.58 |
| 0.9 | 20 | Chronos_Bolt | point_forecast_window_sigma | 29,818.07 | 0.8477 | 8,140.32 |
| 0.9 | 21 | Chronos_Bolt | point_forecast_error_sigma | 30,023.03 | 0.8468 | 8,187.18 |
| 0.9 | 22 | Chronos_Bolt | point_forecast | 30,300.40 | 0.8472 | 8,162.69 |
| 0.95 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.95 | 2 | Baseline | baseline | 19,131.28 | 0.9307 | 3,704.57 |
| 0.95 | 3 | Baseline_full_history | baseline_full_history | 19,183.06 | 0.9335 | 3,555.14 |
| 0.95 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 0.95 | 5 | LightGBM | point_forecast_window_sigma | 21,056.86 | 0.9116 | 4,724.49 |
| 0.95 | 6 | LightGBM | point_forecast_error_sigma | 21,269.15 | 0.9105 | 4,783.45 |
| 0.95 | 7 | MovingAverage | point_forecast_error_sigma | 22,940.36 | 0.8975 | 5,474.75 |
| 0.95 | 8 | MovingAverage | point_forecast_window_sigma | 22,978.34 | 0.8961 | 5,550.50 |
| 0.95 | 9 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 0.95 | 10 | Chronos2_Covariates | point_forecast_window_sigma | 27,968.26 | 0.8681 | 7,048.66 |
| 0.95 | 11 | Chronos2_Covariates | quantile_forecast | 28,012.33 | 0.8609 | 7,431.04 |
| 0.95 | 12 | Chronos2_Covariates | point_forecast | 28,067.52 | 0.8728 | 6,797.72 |
| 0.95 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 28,067.52 | 0.8728 | 6,797.72 |
| 0.95 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,293.05 | 0.8655 | 7,185.28 |
| 0.95 | 15 | Chronos2_History | quantile_forecast | 28,559.06 | 0.8556 | 7,716.93 |
| 0.95 | 16 | Chronos2_History | point_forecast_window_sigma | 28,714.44 | 0.8607 | 7,441.47 |
| 0.95 | 17 | Chronos2_History | point_forecast_error_sigma | 28,863.57 | 0.8599 | 7,488.68 |
| 0.95 | 18 | Chronos2_History | point_forecast | 28,879.09 | 0.8647 | 7,232.66 |
| 0.95 | 18 | Chronos2_History | quantile_forecast_history_sigma | 28,879.09 | 0.8647 | 7,232.66 |
| 0.95 | 20 | Chronos_Bolt | point_forecast | 29,827.08 | 0.8574 | 7,617.95 |
| 0.95 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,894.70 | 0.8517 | 7,927.24 |
| 0.95 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,028.05 | 0.8510 | 7,959.48 |
| 0.99 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.99 | 2 | Baseline_full_history | baseline_full_history | 19,549.74 | 0.9381 | 3,309.97 |
| 0.99 | 3 | Baseline | baseline | 19,611.40 | 0.9343 | 3,512.30 |
| 0.99 | 4 | LightGBM | point_forecast | 20,917.94 | 0.9261 | 3,947.50 |
| 0.99 | 5 | LightGBM | point_forecast_window_sigma | 21,189.70 | 0.9179 | 4,387.24 |
| 0.99 | 6 | LightGBM | point_forecast_error_sigma | 21,262.78 | 0.9181 | 4,378.52 |
| 0.99 | 7 | MovingAverage | point_forecast_window_sigma | 23,394.97 | 0.8993 | 5,383.50 |
| 0.99 | 8 | MovingAverage | point_forecast | 23,450.92 | 0.9042 | 5,117.75 |
| 0.99 | 9 | MovingAverage | point_forecast_error_sigma | 23,467.67 | 0.9001 | 5,337.50 |
| 0.99 | 10 | Chronos2_Covariates | quantile_forecast | 27,547.56 | 0.8702 | 6,934.89 |
| 0.99 | 11 | Chronos2_Covariates | point_forecast_window_sigma | 27,932.07 | 0.8741 | 6,727.36 |
| 0.99 | 12 | Chronos2_Covariates | point_forecast | 27,963.25 | 0.8790 | 6,466.56 |
| 0.99 | 12 | Chronos2_Covariates | quantile_forecast_history_sigma | 27,963.25 | 0.8790 | 6,466.56 |
| 0.99 | 14 | Chronos2_Covariates | point_forecast_error_sigma | 28,153.60 | 0.8729 | 6,794.00 |
| 0.99 | 15 | Chronos2_History | quantile_forecast | 28,269.76 | 0.8636 | 7,286.77 |
| 0.99 | 16 | Chronos2_History | point_forecast_window_sigma | 28,849.37 | 0.8656 | 7,183.08 |
| 0.99 | 17 | Chronos2_History | point_forecast | 28,863.91 | 0.8707 | 6,907.32 |
| 0.99 | 17 | Chronos2_History | quantile_forecast_history_sigma | 28,863.91 | 0.8707 | 6,907.32 |
| 0.99 | 19 | Chronos2_History | point_forecast_error_sigma | 28,871.57 | 0.8658 | 7,171.77 |
| 0.99 | 20 | Chronos_Bolt | point_forecast | 29,788.10 | 0.8635 | 7,296.78 |
| 0.99 | 21 | Chronos_Bolt | point_forecast_window_sigma | 29,958.62 | 0.8569 | 7,646.50 |
| 0.99 | 22 | Chronos_Bolt | point_forecast_error_sigma | 30,005.52 | 0.8568 | 7,650.35 |

## Các cặp bị đảo thứ hạng so với cấu hình cơ sở

| Tham số | Loại | Giá trị | Cặp A | Cặp B | Hạng cơ sở (A/B) | Hạng mới (A/B) | Liên quan Oracle |
|---|---|---:|---|---|---|---|---|
| holding_cost_per_unit_per_day | cost accounting (no behavior change) | 0.001355 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| holding_cost_per_unit_per_day | cost accounting (no behavior change) | 0.001355 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_error_sigma | 18/17 | 17/19 | không |
| holding_cost_per_unit_per_day | cost accounting (no behavior change) | 0.001355 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast_history_sigma | 17/18 | 19/17 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | point_forecast_window_sigma | 12/10 | 10/12 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 12/11 | 10/14 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_Covariates | quantile_forecast | 14/11 | 13/14 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_Covariates | quantile_forecast_history_sigma | 10/12 | 12/10 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_Covariates | quantile_forecast | Chronos2_Covariates | quantile_forecast_history_sigma | 11/12 | 14/10 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_error_sigma | 18/17 | 15/18 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_window_sigma | 18/16 | 15/17 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 18/15 | 15/19 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast | 17/15 | 18/19 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast_history_sigma | 17/18 | 18/15 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast_window_sigma | Chronos2_History | quantile_forecast | 16/15 | 17/19 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | point_forecast_window_sigma | Chronos2_History | quantile_forecast_history_sigma | 16/18 | 17/15 | không |
| ordering_cost | cost accounting (no behavior change) | 2.5 | Chronos2_History | quantile_forecast | Chronos2_History | quantile_forecast_history_sigma | 15/18 | 19/15 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | point_forecast_error_sigma | 12/14 | 14/13 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast | Chronos2_History | quantile_forecast | 12/15 | 14/11 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_Covariates | quantile_forecast_history_sigma | 14/12 | 13/14 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_History | quantile_forecast | 14/15 | 13/11 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_Covariates | quantile_forecast | 10/11 | 12/10 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_History | quantile_forecast | 10/15 | 12/11 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos2_Covariates | quantile_forecast_history_sigma | Chronos2_History | quantile_forecast | 12/15 | 14/11 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_error_sigma | 20/22 | 22/21 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_window_sigma | 20/21 | 22/20 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | LightGBM | point_forecast | LightGBM | point_forecast_error_sigma | 4/6 | 6/5 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | LightGBM | point_forecast | LightGBM | point_forecast_window_sigma | 4/5 | 6/4 | không |
| ordering_cost | cost accounting (no behavior change) | 10.0 | MovingAverage | point_forecast_error_sigma | MovingAverage | point_forecast_window_sigma | 7/8 | 8/7 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 12/11 | 11/13 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_Covariates | quantile_forecast | Chronos2_Covariates | quantile_forecast_history_sigma | 11/12 | 13/11 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_error_sigma | 18/17 | 16/18 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 18/15 | 16/19 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast | 17/15 | 18/19 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast_history_sigma | 17/18 | 18/16 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | point_forecast_window_sigma | Chronos2_History | quantile_forecast | 16/15 | 15/19 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos2_History | quantile_forecast | Chronos2_History | quantile_forecast_history_sigma | 15/18 | 19/16 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_error_sigma | 20/22 | 22/21 | không |
| service_level | behavior change (ROP/SS/Q) | 0.9 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_window_sigma | 20/21 | 22/20 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_Covariates | quantile_forecast | 10/11 | 11/10 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_error_sigma | 18/17 | 17/19 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast_history_sigma | 17/18 | 19/17 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | MovingAverage | point_forecast | MovingAverage | point_forecast_error_sigma | 9/7 | 8/9 | không |
| service_level | behavior change (ROP/SS/Q) | 0.99 | MovingAverage | point_forecast_error_sigma | MovingAverage | point_forecast_window_sigma | 7/8 | 9/7 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | point_forecast_error_sigma | 12/14 | 14/13 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast | Chronos2_History | quantile_forecast | 12/15 | 14/11 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_Covariates | quantile_forecast_history_sigma | 14/12 | 13/14 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_History | quantile_forecast | 14/15 | 13/11 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_Covariates | quantile_forecast | 10/11 | 12/10 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_History | quantile_forecast | 10/15 | 12/11 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos2_Covariates | quantile_forecast_history_sigma | Chronos2_History | quantile_forecast | 12/15 | 14/11 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_error_sigma | 20/22 | 22/21 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | Chronos_Bolt | point_forecast | Chronos_Bolt | point_forecast_window_sigma | 20/21 | 22/20 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | LightGBM | point_forecast | LightGBM | point_forecast_error_sigma | 4/6 | 6/5 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | LightGBM | point_forecast | LightGBM | point_forecast_window_sigma | 4/5 | 6/4 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 1.0 | MovingAverage | point_forecast_error_sigma | MovingAverage | point_forecast_window_sigma | 7/8 | 8/7 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | point_forecast_window_sigma | 12/10 | 10/12 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 12/11 | 10/14 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_Covariates | point_forecast_error_sigma | Chronos2_Covariates | quantile_forecast | 14/11 | 13/14 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_Covariates | point_forecast_window_sigma | Chronos2_Covariates | quantile_forecast_history_sigma | 10/12 | 12/10 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_Covariates | quantile_forecast | Chronos2_Covariates | quantile_forecast_history_sigma | 11/12 | 14/10 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_error_sigma | 18/17 | 15/18 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast | Chronos2_History | point_forecast_window_sigma | 18/16 | 15/17 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 18/15 | 15/19 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast | 17/15 | 18/19 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast_error_sigma | Chronos2_History | quantile_forecast_history_sigma | 17/18 | 18/15 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast_window_sigma | Chronos2_History | quantile_forecast | 16/15 | 17/19 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | point_forecast_window_sigma | Chronos2_History | quantile_forecast_history_sigma | 16/18 | 17/15 | không |
| shortage_cost_per_unit | cost accounting (no behavior change) | 4.0 | Chronos2_History | quantile_forecast | Chronos2_History | quantile_forecast_history_sigma | 15/18 | 19/15 | không |
