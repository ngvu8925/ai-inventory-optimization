# Ma trận độ nhạy (kịch bản normal)

Mỗi khối chỉ đổi MỘT tham số; các tham số còn lại ở mức cơ sở.
`Oracle` là cận dưới chi phí (perfect foresight), **không phải mô hình**.

## holding_cost_per_unit_per_day (h)

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 0.001355 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,635.75 | 0.9488 | 2,738.00 |
| 0.001355 | 2 | Baseline_full_history | baseline_full_history | 18,769.03 | 0.9335 | 3,555.14 |
| 0.001355 | 3 | Baseline | baseline | 18,790.74 | 0.9307 | 3,704.57 |
| 0.001355 | 4 | LightGBM | point_forecast | 20,552.45 | 0.9194 | 4,308.16 |
| 0.001355 | 5 | MovingAverage | point_forecast | 22,978.02 | 0.8990 | 5,395.00 |
| 0.001355 | 6 | Chronos2_Covariates | quantile_forecast | 27,772.36 | 0.8609 | 7,431.04 |
| 0.001355 | 7 | Chronos2_Covariates | point_forecast | 27,801.94 | 0.8728 | 6,797.72 |
| 0.001355 | 8 | Chronos2_History | quantile_forecast | 28,322.46 | 0.8556 | 7,716.94 |
| 0.001355 | 9 | Chronos2_History | point_forecast | 28,619.62 | 0.8647 | 7,232.67 |
| 0.001355 | 10 | Chronos_Bolt | quantile_forecast | 29,135.60 | 0.8476 | 8,146.47 |
| 0.001355 | 11 | Chronos_Bolt | point_forecast | 29,575.17 | 0.8574 | 7,617.95 |
| 0.002032 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.002032 | 2 | Baseline | baseline | 19,131.28 | 0.9307 | 3,704.57 |
| 0.002032 | 3 | Baseline_full_history | baseline_full_history | 19,183.06 | 0.9335 | 3,555.14 |
| 0.002032 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 0.002032 | 5 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 0.002032 | 6 | Chronos2_Covariates | quantile_forecast | 28,012.33 | 0.8609 | 7,431.04 |
| 0.002032 | 7 | Chronos2_Covariates | point_forecast | 28,067.52 | 0.8728 | 6,797.72 |
| 0.002032 | 8 | Chronos2_History | quantile_forecast | 28,559.08 | 0.8556 | 7,716.94 |
| 0.002032 | 9 | Chronos2_History | point_forecast | 28,879.10 | 0.8647 | 7,232.67 |
| 0.002032 | 10 | Chronos_Bolt | quantile_forecast | 29,366.75 | 0.8476 | 8,146.47 |
| 0.002032 | 11 | Chronos_Bolt | point_forecast | 29,827.08 | 0.8574 | 7,617.95 |
| 0.002709 | 1 | Oracle (lower bound, not a model) | oracle_reference | 13,055.18 | 0.9488 | 2,738.00 |
| 0.002709 | 2 | Baseline | baseline | 19,471.82 | 0.9307 | 3,704.57 |
| 0.002709 | 3 | Baseline_full_history | baseline_full_history | 19,597.09 | 0.9335 | 3,555.14 |
| 0.002709 | 4 | LightGBM | point_forecast | 21,218.10 | 0.9194 | 4,308.16 |
| 0.002709 | 5 | MovingAverage | point_forecast | 23,690.52 | 0.8990 | 5,395.00 |
| 0.002709 | 6 | Chronos2_Covariates | quantile_forecast | 28,252.30 | 0.8609 | 7,431.04 |
| 0.002709 | 7 | Chronos2_Covariates | point_forecast | 28,333.11 | 0.8728 | 6,797.72 |
| 0.002709 | 8 | Chronos2_History | quantile_forecast | 28,795.69 | 0.8556 | 7,716.94 |
| 0.002709 | 9 | Chronos2_History | point_forecast | 29,138.58 | 0.8647 | 7,232.67 |
| 0.002709 | 10 | Chronos_Bolt | quantile_forecast | 29,597.90 | 0.8476 | 8,146.47 |
| 0.002709 | 11 | Chronos_Bolt | point_forecast | 30,079.00 | 0.8574 | 7,617.95 |

## shortage_cost_per_unit (p)

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 1.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 10,107.47 | 0.9488 | 2,738.00 |
| 1.0 | 2 | Baseline | baseline | 15,426.70 | 0.9307 | 3,704.57 |
| 1.0 | 3 | Baseline_full_history | baseline_full_history | 15,627.88 | 0.9335 | 3,555.14 |
| 1.0 | 4 | LightGBM | point_forecast | 16,577.11 | 0.9194 | 4,308.16 |
| 1.0 | 5 | MovingAverage | point_forecast | 17,939.27 | 0.8990 | 5,395.00 |
| 1.0 | 6 | Chronos2_Covariates | quantile_forecast | 20,581.30 | 0.8609 | 7,431.04 |
| 1.0 | 7 | Chronos2_History | quantile_forecast | 20,842.14 | 0.8556 | 7,716.94 |
| 1.0 | 8 | Chronos_Bolt | quantile_forecast | 21,220.27 | 0.8476 | 8,146.47 |
| 1.0 | 9 | Chronos2_Covariates | point_forecast | 21,269.83 | 0.8728 | 6,797.72 |
| 1.0 | 10 | Chronos2_History | point_forecast | 21,646.46 | 0.8647 | 7,232.67 |
| 1.0 | 11 | Chronos_Bolt | point_forecast | 22,209.10 | 0.8574 | 7,617.95 |
| 2.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 2.0 | 2 | Baseline | baseline | 19,131.28 | 0.9307 | 3,704.57 |
| 2.0 | 3 | Baseline_full_history | baseline_full_history | 19,183.06 | 0.9335 | 3,555.14 |
| 2.0 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 2.0 | 5 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 2.0 | 6 | Chronos2_Covariates | quantile_forecast | 28,012.33 | 0.8609 | 7,431.04 |
| 2.0 | 7 | Chronos2_Covariates | point_forecast | 28,067.52 | 0.8728 | 6,797.72 |
| 2.0 | 8 | Chronos2_History | quantile_forecast | 28,559.08 | 0.8556 | 7,716.94 |
| 2.0 | 9 | Chronos2_History | point_forecast | 28,879.10 | 0.8647 | 7,232.67 |
| 2.0 | 10 | Chronos_Bolt | quantile_forecast | 29,366.75 | 0.8476 | 8,146.47 |
| 2.0 | 11 | Chronos_Bolt | point_forecast | 29,827.08 | 0.8574 | 7,617.95 |
| 4.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 18,321.47 | 0.9488 | 2,738.00 |
| 4.0 | 2 | Baseline_full_history | baseline_full_history | 26,293.43 | 0.9335 | 3,555.14 |
| 4.0 | 3 | Baseline | baseline | 26,540.43 | 0.9307 | 3,704.57 |
| 4.0 | 4 | LightGBM | point_forecast | 29,501.59 | 0.9194 | 4,308.16 |
| 4.0 | 5 | MovingAverage | point_forecast | 34,124.27 | 0.8990 | 5,395.00 |
| 4.0 | 6 | Chronos2_Covariates | point_forecast | 41,662.92 | 0.8728 | 6,797.72 |
| 4.0 | 7 | Chronos2_Covariates | quantile_forecast | 42,874.39 | 0.8609 | 7,431.04 |
| 4.0 | 8 | Chronos2_History | point_forecast | 43,344.39 | 0.8647 | 7,232.67 |
| 4.0 | 9 | Chronos2_History | quantile_forecast | 43,992.95 | 0.8556 | 7,716.94 |
| 4.0 | 10 | Chronos_Bolt | point_forecast | 45,063.06 | 0.8574 | 7,617.95 |
| 4.0 | 11 | Chronos_Bolt | quantile_forecast | 45,659.70 | 0.8476 | 8,146.47 |

## ordering_cost (K)

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 2.5 | 1 | Oracle (lower bound, not a model) | oracle_reference | 9,475.47 | 0.9488 | 2,738.00 |
| 2.5 | 2 | Baseline_full_history | baseline_full_history | 13,768.06 | 0.9335 | 3,555.14 |
| 2.5 | 3 | Baseline | baseline | 13,781.28 | 0.9307 | 3,704.57 |
| 2.5 | 4 | LightGBM | point_forecast | 15,250.27 | 0.9194 | 4,308.16 |
| 2.5 | 5 | MovingAverage | point_forecast | 17,596.77 | 0.8990 | 5,395.00 |
| 2.5 | 6 | Chronos2_Covariates | point_forecast | 21,230.02 | 0.8728 | 6,797.72 |
| 2.5 | 7 | Chronos2_Covariates | quantile_forecast | 21,797.33 | 0.8609 | 7,431.04 |
| 2.5 | 8 | Chronos2_History | point_forecast | 22,061.60 | 0.8647 | 7,232.67 |
| 2.5 | 9 | Chronos2_History | quantile_forecast | 22,351.58 | 0.8556 | 7,716.94 |
| 2.5 | 10 | Chronos_Bolt | point_forecast | 22,909.58 | 0.8574 | 7,617.95 |
| 2.5 | 11 | Chronos_Bolt | quantile_forecast | 23,176.75 | 0.8476 | 8,146.47 |
| 5.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 5.0 | 2 | Baseline | baseline | 19,131.28 | 0.9307 | 3,704.57 |
| 5.0 | 3 | Baseline_full_history | baseline_full_history | 19,183.06 | 0.9335 | 3,555.14 |
| 5.0 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 5.0 | 5 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 5.0 | 6 | Chronos2_Covariates | quantile_forecast | 28,012.33 | 0.8609 | 7,431.04 |
| 5.0 | 7 | Chronos2_Covariates | point_forecast | 28,067.52 | 0.8728 | 6,797.72 |
| 5.0 | 8 | Chronos2_History | quantile_forecast | 28,559.08 | 0.8556 | 7,716.94 |
| 5.0 | 9 | Chronos2_History | point_forecast | 28,879.10 | 0.8647 | 7,232.67 |
| 5.0 | 10 | Chronos_Bolt | quantile_forecast | 29,366.75 | 0.8476 | 8,146.47 |
| 5.0 | 11 | Chronos_Bolt | point_forecast | 29,827.08 | 0.8574 | 7,617.95 |
| 10.0 | 1 | Oracle (lower bound, not a model) | oracle_reference | 19,585.47 | 0.9488 | 2,738.00 |
| 10.0 | 2 | Baseline | baseline | 29,831.28 | 0.9307 | 3,704.57 |
| 10.0 | 3 | Baseline_full_history | baseline_full_history | 30,013.06 | 0.9335 | 3,555.14 |
| 10.0 | 4 | LightGBM | point_forecast | 32,155.27 | 0.9194 | 4,308.16 |
| 10.0 | 5 | MovingAverage | point_forecast | 34,809.27 | 0.8990 | 5,395.00 |
| 10.0 | 6 | Chronos2_Covariates | quantile_forecast | 40,442.33 | 0.8609 | 7,431.04 |
| 10.0 | 7 | Chronos2_History | quantile_forecast | 40,974.08 | 0.8556 | 7,716.94 |
| 10.0 | 8 | Chronos2_Covariates | point_forecast | 41,742.52 | 0.8728 | 6,797.72 |
| 10.0 | 9 | Chronos_Bolt | quantile_forecast | 41,746.75 | 0.8476 | 8,146.47 |
| 10.0 | 10 | Chronos2_History | point_forecast | 42,514.10 | 0.8647 | 7,232.67 |
| 10.0 | 11 | Chronos_Bolt | point_forecast | 43,662.08 | 0.8574 | 7,617.95 |

## service_level

| Giá trị | Hạng | Model | Policy | cost_total | fill_rate_pooled | total_short |
|---:|---:|---|---|---:|---:|---:|
| 0.9 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.9 | 2 | Baseline | baseline | 19,137.68 | 0.9264 | 3,933.75 |
| 0.9 | 3 | Baseline_full_history | baseline_full_history | 19,298.55 | 0.9270 | 3,901.41 |
| 0.9 | 4 | LightGBM | point_forecast | 20,939.82 | 0.9130 | 4,646.94 |
| 0.9 | 5 | MovingAverage | point_forecast | 22,801.99 | 0.8974 | 5,481.75 |
| 0.9 | 6 | Chronos2_Covariates | point_forecast | 28,095.99 | 0.8668 | 7,118.16 |
| 0.9 | 7 | Chronos2_Covariates | quantile_forecast | 28,433.06 | 0.8532 | 7,844.95 |
| 0.9 | 8 | Chronos2_History | point_forecast | 29,068.11 | 0.8574 | 7,621.56 |
| 0.9 | 9 | Chronos2_History | quantile_forecast | 29,123.61 | 0.8468 | 8,188.58 |
| 0.9 | 10 | Chronos_Bolt | quantile_forecast | 29,946.75 | 0.8383 | 8,641.20 |
| 0.9 | 11 | Chronos_Bolt | point_forecast | 30,300.40 | 0.8472 | 8,162.69 |
| 0.95 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.95 | 2 | Baseline | baseline | 19,131.28 | 0.9307 | 3,704.57 |
| 0.95 | 3 | Baseline_full_history | baseline_full_history | 19,183.06 | 0.9335 | 3,555.14 |
| 0.95 | 4 | LightGBM | point_forecast | 20,885.27 | 0.9194 | 4,308.16 |
| 0.95 | 5 | MovingAverage | point_forecast | 23,334.27 | 0.8990 | 5,395.00 |
| 0.95 | 6 | Chronos2_Covariates | quantile_forecast | 28,012.33 | 0.8609 | 7,431.04 |
| 0.95 | 7 | Chronos2_Covariates | point_forecast | 28,067.52 | 0.8728 | 6,797.72 |
| 0.95 | 8 | Chronos2_History | quantile_forecast | 28,559.08 | 0.8556 | 7,716.94 |
| 0.95 | 9 | Chronos2_History | point_forecast | 28,879.10 | 0.8647 | 7,232.67 |
| 0.95 | 10 | Chronos_Bolt | quantile_forecast | 29,366.75 | 0.8476 | 8,146.47 |
| 0.95 | 11 | Chronos_Bolt | point_forecast | 29,827.08 | 0.8574 | 7,617.95 |
| 0.99 | 1 | Oracle (lower bound, not a model) | oracle_reference | 12,845.47 | 0.9488 | 2,738.00 |
| 0.99 | 2 | Baseline_full_history | baseline_full_history | 19,549.74 | 0.9381 | 3,309.97 |
| 0.99 | 3 | Baseline | baseline | 19,611.40 | 0.9343 | 3,512.30 |
| 0.99 | 4 | LightGBM | point_forecast | 20,917.94 | 0.9261 | 3,947.50 |
| 0.99 | 5 | MovingAverage | point_forecast | 23,450.92 | 0.9042 | 5,117.75 |
| 0.99 | 6 | Chronos2_Covariates | quantile_forecast | 27,547.56 | 0.8702 | 6,934.89 |
| 0.99 | 7 | Chronos2_Covariates | point_forecast | 27,963.25 | 0.8790 | 6,466.56 |
| 0.99 | 8 | Chronos2_History | quantile_forecast | 28,269.77 | 0.8636 | 7,286.78 |
| 0.99 | 9 | Chronos2_History | point_forecast | 28,863.92 | 0.8707 | 6,907.33 |
| 0.99 | 10 | Chronos_Bolt | quantile_forecast | 29,235.36 | 0.8546 | 7,770.30 |
| 0.99 | 11 | Chronos_Bolt | point_forecast | 29,788.10 | 0.8635 | 7,296.78 |

## Các cặp bị đảo thứ hạng so với cấu hình cơ sở

| Tham số | Giá trị | Cặp A | Cặp B | Hạng cơ sở (A/B) | Hạng mới (A/B) | Liên quan Oracle |
|---|---:|---|---|---|---|---|
| holding_cost_per_unit_per_day | 0.001355 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| ordering_cost | 2.5 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| ordering_cost | 2.5 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 7/6 | 6/7 | không |
| ordering_cost | 2.5 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 9/8 | 8/9 | không |
| ordering_cost | 2.5 | Chronos_Bolt | point_forecast | Chronos_Bolt | quantile_forecast | 11/10 | 10/11 | không |
| ordering_cost | 10.0 | Chronos2_Covariates | point_forecast | Chronos2_History | quantile_forecast | 7/8 | 8/7 | không |
| ordering_cost | 10.0 | Chronos2_History | point_forecast | Chronos_Bolt | quantile_forecast | 9/10 | 10/9 | không |
| service_level | 0.9 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 7/6 | 6/7 | không |
| service_level | 0.9 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 9/8 | 8/9 | không |
| service_level | 0.99 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| shortage_cost_per_unit | 1.0 | Chronos2_Covariates | point_forecast | Chronos2_History | quantile_forecast | 7/8 | 9/7 | không |
| shortage_cost_per_unit | 1.0 | Chronos2_Covariates | point_forecast | Chronos_Bolt | quantile_forecast | 7/10 | 9/8 | không |
| shortage_cost_per_unit | 1.0 | Chronos2_History | point_forecast | Chronos_Bolt | quantile_forecast | 9/10 | 10/8 | không |
| shortage_cost_per_unit | 4.0 | Baseline | baseline | Baseline_full_history | baseline_full_history | 2/3 | 3/2 | không |
| shortage_cost_per_unit | 4.0 | Chronos2_Covariates | point_forecast | Chronos2_Covariates | quantile_forecast | 7/6 | 6/7 | không |
| shortage_cost_per_unit | 4.0 | Chronos2_History | point_forecast | Chronos2_History | quantile_forecast | 9/8 | 8/9 | không |
| shortage_cost_per_unit | 4.0 | Chronos_Bolt | point_forecast | Chronos_Bolt | quantile_forecast | 11/10 | 10/11 | không |
