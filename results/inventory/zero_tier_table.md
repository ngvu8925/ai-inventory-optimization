| Tier | Hạng* | Model | Policy | Fill gộp | Tổng thiếu | Tổng chi phí |
|---|---:|---|---|---:|---:|---:|
| smooth | — | Oracle (lower bound, not a model) | Oracle* | 0.9541 | 1,794.50 | 5,420.31 |
| smooth | 1 | Baseline_full_history | Baseline_full | 0.9436 | 2,206.91 | 7,225.92 |
| smooth | 2 | Baseline | Baseline | 0.9397 | 2,356.77 | 7,371.48 |
| smooth | 3 | Chronos2_Covariates | point (sigma full hist) | 0.9258 | 2,899.84 | 8,650.48 |
| smooth | 3 | Chronos2_Covariates | quantile (sigma hist) | 0.9258 | 2,899.84 | 8,650.48 |
| smooth | 5 | LightGBM | point (sigma full hist) | 0.9210 | 3,089.75 | 8,900.38 |
| smooth | 6 | Chronos2_Covariates | point (sigma 91) | 0.9204 | 3,110.33 | 8,996.60 |
| smooth | 7 | Chronos2_Covariates | point (sigma error) | 0.9169 | 3,249.41 | 9,227.33 |
| smooth | 8 | Chronos2_Covariates | quantile | 0.9130 | 3,403.38 | 9,489.25 |
| smooth | 9 | LightGBM | point (sigma 91) | 0.9118 | 3,447.93 | 9,535.40 |
| smooth | 10 | LightGBM | point (sigma error) | 0.9103 | 3,505.59 | 9,640.62 |
| smooth | 11 | Chronos2_History | point (sigma full hist) | 0.9127 | 3,413.08 | 9,657.13 |
| smooth | 11 | Chronos2_History | quantile (sigma hist) | 0.9127 | 3,413.08 | 9,657.13 |
| smooth | 13 | Chronos2_History | point (sigma 91) | 0.9086 | 3,573.22 | 9,901.22 |
| smooth | 14 | Chronos2_History | point (sigma error) | 0.9072 | 3,629.38 | 9,984.67 |
| smooth | 15 | Chronos2_History | quantile | 0.9042 | 3,745.77 | 10,172.95 |
| smooth | 16 | MovingAverage | point (sigma full hist) | 0.8981 | 3,983.00 | 10,703.46 |
| smooth | 17 | Chronos_Bolt | point (sigma full hist) | 0.8984 | 3,970.54 | 10,757.15 |
| smooth | 18 | MovingAverage | point (sigma error) | 0.8968 | 4,033.50 | 10,767.25 |
| smooth | 19 | MovingAverage | point (sigma 91) | 0.8960 | 4,066.50 | 10,810.51 |
| smooth | 20 | Chronos_Bolt | point (sigma 91) | 0.8911 | 4,258.09 | 11,252.78 |
| smooth | 21 | Chronos_Bolt | point (sigma error) | 0.8906 | 4,278.74 | 11,293.95 |
| medium | — | Oracle (lower bound, not a model) | Oracle* | 0.9373 | 765.00 | 4,993.10 |
| medium | 1 | LightGBM | point (sigma 91) | 0.9183 | 996.42 | 7,372.57 |
| medium | 2 | LightGBM | point (sigma error) | 0.9182 | 998.37 | 7,410.26 |
| medium | 3 | Baseline_full_history | Baseline_full | 0.9144 | 1,043.97 | 7,488.66 |
| medium | 4 | Baseline | Baseline | 0.9129 | 1,062.42 | 7,498.07 |
| medium | 5 | LightGBM | point (sigma full hist) | 0.9227 | 942.70 | 7,689.91 |
| medium | 6 | MovingAverage | point (sigma error) | 0.9086 | 1,114.75 | 7,759.08 |
| medium | 7 | MovingAverage | point (sigma 91) | 0.9053 | 1,155.25 | 7,819.44 |
| medium | 8 | MovingAverage | point (sigma full hist) | 0.9103 | 1,095.00 | 8,060.92 |
| medium | 9 | Chronos_Bolt | point (sigma 91) | 0.7785 | 2,702.49 | 12,191.72 |
| medium | 10 | Chronos_Bolt | point (sigma error) | 0.7772 | 2,718.89 | 12,248.50 |
| medium | 11 | Chronos2_History | quantile | 0.7594 | 2,934.96 | 12,254.86 |
| medium | 12 | Chronos2_Covariates | quantile | 0.7589 | 2,942.20 | 12,304.31 |
| medium | 13 | Chronos2_History | point (sigma 91) | 0.7647 | 2,870.55 | 12,332.11 |
| medium | 14 | Chronos2_History | point (sigma error) | 0.7654 | 2,862.10 | 12,348.65 |
| medium | 15 | Chronos2_Covariates | point (sigma 91) | 0.7621 | 2,903.17 | 12,406.60 |
| medium | 16 | Chronos2_Covariates | point (sigma error) | 0.7626 | 2,896.48 | 12,437.07 |
| medium | 17 | Chronos_Bolt | point (sigma full hist) | 0.7800 | 2,683.64 | 12,445.20 |
| medium | 18 | Chronos2_History | quantile (sigma hist) | 0.7686 | 2,822.90 | 12,547.55 |
| medium | 18 | Chronos2_History | point (sigma full hist) | 0.7686 | 2,822.90 | 12,547.55 |
| medium | 20 | Chronos2_Covariates | point (sigma full hist) | 0.7654 | 2,862.58 | 12,641.22 |
| medium | 20 | Chronos2_Covariates | quantile (sigma hist) | 0.7654 | 2,862.58 | 12,641.22 |
| intermittent | — | Oracle (lower bound, not a model) | Oracle* | 0.9165 | 178.50 | 2,432.06 |
| intermittent | 1 | LightGBM | point (sigma 91) | 0.8690 | 280.14 | 4,148.89 |
| intermittent | 2 | LightGBM | point (sigma error) | 0.8693 | 279.49 | 4,218.27 |
| intermittent | 3 | Baseline | Baseline | 0.8666 | 285.38 | 4,261.73 |
| intermittent | 4 | LightGBM | point (sigma full hist) | 0.8711 | 275.71 | 4,294.98 |
| intermittent | 5 | MovingAverage | point (sigma 91) | 0.8463 | 328.75 | 4,348.40 |
| intermittent | 6 | MovingAverage | point (sigma error) | 0.8474 | 326.50 | 4,414.03 |
| intermittent | 7 | Baseline_full_history | Baseline_full | 0.8578 | 304.26 | 4,468.48 |
| intermittent | 8 | MovingAverage | point (sigma full hist) | 0.8518 | 317.00 | 4,569.89 |
| intermittent | 9 | Chronos2_History | quantile | 0.5156 | 1,036.20 | 6,131.26 |
| intermittent | 10 | Chronos2_Covariates | quantile | 0.4925 | 1,085.46 | 6,218.77 |
| intermittent | 11 | Chronos_Bolt | point (sigma 91) | 0.5481 | 966.66 | 6,450.20 |
| intermittent | 12 | Chronos2_History | point (sigma 91) | 0.5336 | 997.70 | 6,481.10 |
| intermittent | 13 | Chronos_Bolt | point (sigma error) | 0.5503 | 961.85 | 6,485.60 |
| intermittent | 14 | Chronos2_History | point (sigma error) | 0.5338 | 997.20 | 6,530.25 |
| intermittent | 15 | Chronos2_Covariates | point (sigma 91) | 0.5161 | 1,035.16 | 6,565.07 |
| intermittent | 16 | Chronos_Bolt | point (sigma full hist) | 0.5494 | 963.77 | 6,624.74 |
| intermittent | 17 | Chronos2_Covariates | point (sigma error) | 0.5141 | 1,039.39 | 6,628.65 |
| intermittent | 18 | Chronos2_History | quantile (sigma hist) | 0.5340 | 996.68 | 6,674.41 |
| intermittent | 18 | Chronos2_History | point (sigma full hist) | 0.5340 | 996.68 | 6,674.41 |
| intermittent | 20 | Chronos2_Covariates | point (sigma full hist) | 0.5160 | 1,035.30 | 6,775.83 |
| intermittent | 20 | Chronos2_Covariates | quantile (sigma hist) | 0.5160 | 1,035.30 | 6,775.83 |
