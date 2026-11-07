# Model benchmark (Sub-task 3.3)

Built 2026-11-06T19:13:22 UTC, dataset sha256 `b826d61b541f…`. Grouped CV = the 2.8 folds (seed 42); design skill = share of the differences between designs within a climate that the model explains (heating targets over the 7 heating climates). Training split only.

## 1. RQ1 grid — grouped-CV design skill (higher is better)

| model | Heating R untuned | Heating C-G untuned | Heating tuned | Cooling R untuned | Cooling C-G untuned | Cooling tuned | EUI R untuned | EUI C-G untuned | EUI tuned |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.704 | 0.942 | 0.947 | 0.749 | 0.912 | 0.914 | 0.680 | 0.915 | 0.919 |
| XGBoost | 0.817 | 0.954 | 0.980 | 0.837 | 0.929 | 0.972 | 0.816 | 0.936 | 0.970 |
| LightGBM | 0.853 | 0.956 | 0.977 | 0.877 | 0.944 | 0.967 | 0.854 | 0.951 | 0.969 |
| Gaussian process | 0.952 | 0.956 | 0.956 | 0.904 | 0.956 | 0.956 | 0.919 | 0.954 | 0.954 |
| Neural network (MLP) | 0.982 | 0.995 | 0.996 | 0.986 | 0.994 | 0.996 | 0.967 | 0.979 | 0.987 |

GPR is not tuned by Optuna (E5): its 'tuned' column uses marginal-likelihood fitting only.

## 2. Tuned — pooled R²

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.981 | 0.961 | 0.994 |
| XGBoost | 0.993 | 0.987 | 0.998 |
| LightGBM | 0.991 | 0.985 | 0.998 |
| Gaussian process | 0.984 | 0.980 | 0.997 |
| Neural network (MLP) | 0.998 | 0.998 | 0.999 |

## 3. Tuned — MAE (kWh/m² or W/m²)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.48 | 2.65 | 2.72 |
| XGBoost | 0.32 | 1.55 | 1.70 |
| LightGBM | 0.31 | 1.58 | 1.69 |
| Gaussian process | 0.44 | 1.91 | 2.01 |
| Neural network (MLP) | 0.18 | 0.63 | 1.20 |

## 4. Tuned — RMSE

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.98 | 4.00 | 4.04 |
| XGBoost | 0.60 | 2.29 | 2.46 |
| LightGBM | 0.65 | 2.48 | 2.52 |
| Gaussian process | 0.89 | 2.86 | 3.06 |
| Neural network (MLP) | 0.28 | 0.90 | 1.64 |

## 5. Tuned — CV(RMSE)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.180 | 0.056 | 0.028 |
| XGBoost | 0.111 | 0.032 | 0.017 |
| LightGBM | 0.120 | 0.035 | 0.017 |
| Gaussian process | 0.164 | 0.040 | 0.021 |
| Neural network (MLP) | 0.051 | 0.013 | 0.011 |

## 6. Tuned — within-climate ranking (median Spearman ρ)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.979 | 0.970 | 0.971 |
| XGBoost | 0.992 | 0.988 | 0.989 |
| LightGBM | 0.991 | 0.989 | 0.989 |
| Gaussian process | 0.981 | 0.984 | 0.983 |
| Neural network (MLP) | 0.998 | 0.998 | 0.993 |

## 7. Tuned — top-10 % hit rate

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.786 | 0.813 | 0.823 |
| XGBoost | 0.852 | 0.873 | 0.827 |
| LightGBM | 0.886 | 0.880 | 0.830 |
| Gaussian process | 0.824 | 0.880 | 0.857 |
| Neural network (MLP) | 0.852 | 0.930 | 0.833 |

## 8. Fresh-fold check (E4) — design skill on GroupKFold seed 7

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.950 | 0.913 | 0.916 |
| XGBoost | 0.980 | 0.972 | 0.969 |
| LightGBM | 0.976 | 0.967 | 0.966 |
| Gaussian process | 0.958 | 0.950 | 0.948 |
| Neural network (MLP) | 0.996 | 0.996 | 0.988 |

## 8b. Tuning optimism = skill(seed 42) − skill(seed 7)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | -0.0030 | 0.0008 | 0.0026 |
| XGBoost | 0.0008 | 0.0003 | 0.0011 |
| LightGBM | 0.0003 | -0.0000 | 0.0021 |
| Gaussian process | -0.0021 | 0.0058 | 0.0052 |
| Neural network (MLP) | 0.0002 | 0.0001 | -0.0009 |

Positive = the seed-42 number is flattered by tuning on those folds; a few thousandths is normal.

## 9. Unseen climate (LOCO) — MAPE % (used by M10 for primary targets)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 38.9 | 5.0 | 13.1 |
| XGBoost | 27.0 | 4.9 | 11.3 |
| LightGBM | 31.7 | 4.4 | 13.5 |
| Gaussian process | 27.1 | 12.9 | 5.6 |
| Neural network (MLP) | 25.7 | 3.3 | 8.3 |

## 9b. LOCO median |climate bias| %

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 22.4 | 2.3 | 6.6 |
| XGBoost | 20.6 | 2.6 | 7.5 |
| LightGBM | 28.9 | 2.8 | 12.6 |
| Gaussian process | 26.3 | 5.3 | 3.6 |
| Neural network (MLP) | 11.2 | 2.7 | 4.4 |

## 9c. LOCO bias-removed R² (ranking in an unseen climate)

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 0.897 | 0.909 | 0.849 |
| XGBoost | 0.947 | 0.950 | 0.937 |
| LightGBM | 0.933 | 0.952 | 0.944 |
| Gaussian process | 0.861 | 0.895 | 0.911 |
| Neural network (MLP) | 0.969 | 0.980 | 0.935 |

## 10. Training time — mean fit seconds per fold

| model | Heating | Cooling | EUI |
|---|---|---|---|
| Random Forest | 2.7 | 2.7 | 2.1 |
| XGBoost | 2.0 | 2.2 | 2.0 |
| LightGBM | 5.5 | 2.3 | 1.5 |
| Gaussian process | 87.5 | 70.5 | 49.3 |
| Neural network (MLP) | 44.9 | 44.9 | 44.9 |

MLP: one fit trains all 10 targets, so its time is the whole network's.

## 11. D5 transfer check (E3) — mean untuned skill over 3 primary targets

| model | C-G | R | triggered |
|---|---|---|---|
| XGBoost | 0.9397 | 0.8233 | no |
| Gaussian process | 0.9553 | 0.9251 | no |
| Neural network (MLP) | 0.9892 | 0.9786 | no |

## 12. Secondary targets — design skill (LightGBM + best other family)

| model | Peak heating | Peak cooling | Zone core | Zone south | Zone east | Zone north | Zone west |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.973 | 0.957 | 0.985 | 0.958 | 0.961 | 0.959 | 0.961 |
| Neural network (MLP) | 0.982 | 0.980 | 0.994 | 0.994 | 0.992 | 0.992 | 0.988 |

## 13. Selection (M10, pre-registered) and CV-stage gate

| target | selected | feature set | status | rule | S1/S2 (CV stage) |
|---|---|---|---|---|---|
| Heating | Neural network (MLP) | C-G | provisional_until_3_4 | lowest LOCO MAPE (ties within 0.5 points -> LightGBM) | S1 PASS (skill 0.996, R² 0.998) |
| Cooling | Neural network (MLP) | C-G | provisional_until_3_4 | lowest LOCO MAPE (ties within 0.5 points -> LightGBM) | S1 PASS (skill 0.996, R² 0.998) |
| EUI | Gaussian process | C-G | provisional_until_3_4 | lowest LOCO MAPE (ties within 0.5 points -> LightGBM) | S1 PASS (skill 0.954, R² 0.997) |
| Peak heating | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.982) |
| Peak cooling | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.980) |
| Zone core | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.994) |
| Zone south | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.994) |
| Zone east | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.992) |
| Zone north | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.992) |
| Zone west | Neural network (MLP) | C-G | final | highest grouped skill (ties within 0.005 -> LightGBM) | S2 PASS (skill 0.988) |

Primary selections are provisional until 3.4 adds the physics target forms (E1). The binding gate (S1–S8) is evaluated in 3.7 on the holdout.
