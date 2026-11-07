# Features x tuning — 2 x 2 factorial (sensitivity analysis after 3.3)

Added 7 Nov 2026 (config `sensitivity_3_3`, frozen before the run). The 3.3 grid measured the feature gain before tuning and the tuning gain after the features, so the two are confounded by order and by the skill ceiling. Adding 'tuned on R' gives both orders. Same budgets, search spaces, sampler, objective and folds as 3.3. Reported only — no effect on the 3.3 selection or the gate.

**Relative gain** = share of the remaining error removed = (skill after − skill before) / (1 − skill before).

## 1. Grouped-CV design skill — the four cells

| model | target | R untuned | R tuned | C-G untuned | C-G tuned |
|---|---|---|---|---|---|
| Random Forest | Heating | 0.704 | 0.735 | 0.942 | 0.947 |
| Random Forest | Cooling | 0.749 | 0.769 | 0.912 | 0.914 |
| Random Forest | EUI | 0.680 | 0.710 | 0.915 | 0.919 |
| XGBoost | Heating | 0.817 | 0.906 | 0.954 | 0.980 |
| XGBoost | Cooling | 0.837 | 0.924 | 0.929 | 0.972 |
| XGBoost | EUI | 0.816 | 0.910 | 0.936 | 0.970 |
| LightGBM | Heating | 0.853 | 0.905 | 0.956 | 0.977 |
| LightGBM | Cooling | 0.877 | 0.929 | 0.944 | 0.967 |
| LightGBM | EUI | 0.854 | 0.919 | 0.951 | 0.969 |
| Gaussian process | Heating | 0.952 | — | 0.956 | — |
| Gaussian process | Cooling | 0.904 | — | 0.956 | — |
| Gaussian process | EUI | 0.919 | — | 0.954 | — |
| Neural network (MLP) | Heating | 0.982 | 0.984 | 0.995 | 0.996 |
| Neural network (MLP) | Cooling | 0.986 | 0.988 | 0.994 | 0.996 |
| Neural network (MLP) | EUI | 0.967 | 0.975 | 0.979 | 0.987 |

GPR has no Optuna tuning (E5), so only its untuned cells exist.

## 2. Feature effect, both orders

| model | target | gain untuned | gain tuned | rel. untuned | rel. tuned |
|---|---|---|---|---|---|
| Random Forest | Heating | 0.238 | 0.212 | 81 % | 80 % |
| Random Forest | Cooling | 0.163 | 0.145 | 65 % | 63 % |
| Random Forest | EUI | 0.235 | 0.209 | 73 % | 72 % |
| XGBoost | Heating | 0.137 | 0.074 | 75 % | 79 % |
| XGBoost | Cooling | 0.092 | 0.048 | 56 % | 63 % |
| XGBoost | EUI | 0.120 | 0.060 | 65 % | 67 % |
| LightGBM | Heating | 0.103 | 0.071 | 70 % | 75 % |
| LightGBM | Cooling | 0.067 | 0.039 | 55 % | 54 % |
| LightGBM | EUI | 0.097 | 0.049 | 67 % | 61 % |
| Gaussian process | Heating | 0.004 | — | 8 % | — % |
| Gaussian process | Cooling | 0.052 | — | 54 % | — % |
| Gaussian process | EUI | 0.035 | — | 43 % | — % |
| Neural network (MLP) | Heating | 0.012 | 0.012 | 70 % | 76 % |
| Neural network (MLP) | Cooling | 0.008 | 0.008 | 56 % | 65 % |
| Neural network (MLP) | EUI | 0.012 | 0.012 | 36 % | 46 % |

## 3. Tuning effect, both orders

| model | target | gain on R | gain on C-G | rel. on R | rel. on C-G |
|---|---|---|---|---|---|
| Random Forest | Heating | 0.031 | 0.005 | 10 % | 8 % |
| Random Forest | Cooling | 0.020 | 0.002 | 8 % | 3 % |
| Random Forest | EUI | 0.030 | 0.004 | 9 % | 5 % |
| XGBoost | Heating | 0.089 | 0.026 | 49 % | 57 % |
| XGBoost | Cooling | 0.088 | 0.043 | 54 % | 61 % |
| XGBoost | EUI | 0.094 | 0.034 | 51 % | 53 % |
| LightGBM | Heating | 0.052 | 0.020 | 36 % | 46 % |
| LightGBM | Cooling | 0.052 | 0.023 | 42 % | 41 % |
| LightGBM | EUI | 0.066 | 0.018 | 45 % | 36 % |
| Neural network (MLP) | Heating | 0.002 | 0.001 | 9 % | 27 % |
| Neural network (MLP) | Cooling | 0.001 | 0.002 | 10 % | 28 % |
| Neural network (MLP) | EUI | 0.008 | 0.008 | 24 % | 36 % |

## 4. Means over the 3 primary targets (models tuned by Optuna)

| model | features rel. (untuned → tuned order) | tuning rel. (on R → on C-G) | both together rel. |
|---|---|---|---|
| Random Forest | 73 % → 72 % | 9 % → 5 % | 74 % |
| XGBoost | 66 % → 70 % | 51 % → 57 % | 85 % |
| LightGBM | 64 % → 63 % | 41 % → 41 % | 79 % |
| Neural network (MLP) | 54 % → 62 % | 14 % → 30 % | 69 % |
