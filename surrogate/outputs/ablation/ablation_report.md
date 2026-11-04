# Feature ablation (Sub-task 3.2)

Run 2026-11-04T13:44:02 UTC, 479 s, dataset sha256 `b826d61b541f…`, 720 temporary fits. Untuned models (RF = 2.8 forest, LightGBM = library defaults). Training split only.

## 1. Selection score (D4) — mean grouped-CV design skill over 3 targets × 2 models

Design skill = share of the differences between designs *within a climate* that the model explains (heating over the 7 heating climates).

| set | features | score | tied with best (±0.005) | chosen |
|---|---|---|---|---|
| R  raw 28 | 28 | 0.7862 |  |  |
| G  + geometry | 36 | 0.8047 |  |  |
| S  + sun | 60 | 0.8828 |  |  |
| U  + heat loss | 66 | 0.9309 |  |  |
| C  + climate x building | 71 | 0.9367 | yes |  |
| C - geometry | 63 | 0.9368 | yes | **WINNER** |
| C - sun | 47 | 0.9285 |  |  |
| C - heat loss | 65 | 0.9264 |  |  |

**Winner: C - geometry** (63 features) — highest score; among sets within 0.005 of the best, the one with the fewest features.

## 2. Grouped CV — design skill (higher is better)

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 0.704 | 0.749 | 0.680 | 0.853 | 0.877 | 0.854 |
| G  + geometry | 0.740 | 0.767 | 0.706 | 0.860 | 0.889 | 0.865 |
| S  + sun | 0.771 | 0.881 | 0.866 | 0.902 | 0.940 | 0.937 |
| U  + heat loss | 0.919 | 0.900 | 0.905 | 0.961 | 0.950 | 0.951 |
| C  + climate x building | 0.942 | 0.910 | 0.915 | 0.957 | 0.944 | 0.952 |
| C - geometry | 0.942 | 0.912 | 0.915 | 0.956 | 0.944 | 0.951 |
| C - sun | 0.939 | 0.900 | 0.906 | 0.958 | 0.931 | 0.937 |
| C - heat loss | 0.924 | 0.897 | 0.901 | 0.955 | 0.936 | 0.945 |

## 3. Grouped CV — pooled R²

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 0.891 | 0.886 | 0.977 | 0.945 | 0.944 | 0.990 |
| G  + geometry | 0.904 | 0.894 | 0.979 | 0.948 | 0.950 | 0.991 |
| S  + sun | 0.915 | 0.946 | 0.991 | 0.963 | 0.973 | 0.996 |
| U  + heat loss | 0.970 | 0.954 | 0.993 | 0.985 | 0.977 | 0.997 |
| C  + climate x building | 0.978 | 0.959 | 0.994 | 0.984 | 0.974 | 0.997 |
| C - geometry | 0.979 | 0.960 | 0.994 | 0.984 | 0.975 | 0.997 |
| C - sun | 0.977 | 0.954 | 0.993 | 0.984 | 0.969 | 0.996 |
| C - heat loss | 0.972 | 0.953 | 0.993 | 0.983 | 0.971 | 0.996 |

## 4. Grouped CV — within-climate ranking (median Spearman ρ)

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 0.900 | 0.919 | 0.886 | 0.951 | 0.956 | 0.939 |
| G  + geometry | 0.900 | 0.918 | 0.860 | 0.944 | 0.959 | 0.934 |
| S  + sun | 0.905 | 0.954 | 0.949 | 0.958 | 0.976 | 0.969 |
| U  + heat loss | 0.973 | 0.970 | 0.969 | 0.985 | 0.983 | 0.984 |
| C  + climate x building | 0.978 | 0.971 | 0.968 | 0.986 | 0.981 | 0.985 |
| C - geometry | 0.979 | 0.970 | 0.970 | 0.987 | 0.982 | 0.985 |
| C - sun | 0.975 | 0.968 | 0.967 | 0.984 | 0.978 | 0.981 |
| C - heat loss | 0.975 | 0.960 | 0.959 | 0.986 | 0.979 | 0.980 |

## 5. Leave one group out — loss of design skill when a group is removed from the full set C

| removed group | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI | mean |
|---|---|---|---|---|---|---|---|
| geometry (C-G) | -0.0006 | -0.0020 | 0.0004 | 0.0011 | -0.0004 | 0.0012 | -0.0001 |
| sun/orientation (C-S) | 0.0030 | 0.0100 | 0.0095 | -0.0005 | 0.0127 | 0.0147 | 0.0082 |
| heat loss (C-U) | 0.0176 | 0.0127 | 0.0141 | 0.0027 | 0.0081 | 0.0070 | 0.0103 |
| climate x building (C − climate × building = U) | 0.0231 | 0.0101 | 0.0099 | -0.0032 | -0.0060 | 0.0011 | 0.0058 |

Positive = skill is lost without that group (the group carries information no other group has); near zero or negative = the group adds nothing beyond the others.

## 6. Unseen climate (leave-one-climate-out) — REPORTED ONLY, not used to choose

## 6a. LOCO MAPE % (lower is better)

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 81.3 | 7.3 | 14.8 | 46.0 | 6.7 | 15.9 |
| G  + geometry | 79.1 | 7.1 | 15.2 | 45.9 | 6.7 | 15.9 |
| S  + sun | 76.1 | 5.9 | 14.9 | 45.7 | 5.9 | 15.8 |
| U  + heat loss | 75.2 | 5.9 | 14.6 | 45.1 | 5.7 | 15.9 |
| C  + climate x building | 47.0 | 5.5 | 12.3 | 31.9 | 4.1 | 13.5 |
| C - geometry | 47.0 | 5.3 | 12.9 | 32.0 | 4.1 | 13.5 |
| C - sun | 46.9 | 5.5 | 13.2 | 31.7 | 4.2 | 13.8 |
| C - heat loss | 38.9 | 5.7 | 13.1 | 31.3 | 4.5 | 13.2 |

## 6b. LOCO median |climate bias| %

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 21.4 | 3.7 | 8.6 | 38.6 | 5.8 | 14.6 |
| G  + geometry | 25.3 | 3.8 | 8.5 | 38.6 | 6.0 | 14.6 |
| S  + sun | 24.8 | 3.3 | 8.8 | 39.3 | 4.6 | 14.9 |
| U  + heat loss | 22.8 | 4.1 | 9.4 | 39.9 | 4.6 | 15.3 |
| C  + climate x building | 26.2 | 3.0 | 6.4 | 31.0 | 1.6 | 14.9 |
| C - geometry | 25.4 | 2.8 | 5.8 | 30.7 | 1.5 | 14.7 |
| C - sun | 25.1 | 2.5 | 6.6 | 29.4 | 1.3 | 14.8 |
| C - heat loss | 21.9 | 3.1 | 6.4 | 31.0 | 1.7 | 13.7 |

## 6c. LOCO bias-removed R² (ranking quality in an unseen climate)

| set | Random Forest Heating | Random Forest Cooling | Random Forest EUI | LightGBM Heating | LightGBM Cooling | LightGBM EUI |
|---|---|---|---|---|---|---|
| R  raw 28 | 0.689 | 0.796 | 0.725 | 0.828 | 0.881 | 0.834 |
| G  + geometry | 0.699 | 0.817 | 0.747 | 0.833 | 0.887 | 0.837 |
| S  + sun | 0.731 | 0.894 | 0.860 | 0.860 | 0.921 | 0.885 |
| U  + heat loss | 0.822 | 0.905 | 0.874 | 0.893 | 0.929 | 0.880 |
| C  + climate x building | 0.883 | 0.903 | 0.846 | 0.919 | 0.914 | 0.908 |
| C - geometry | 0.886 | 0.903 | 0.848 | 0.916 | 0.916 | 0.908 |
| C - sun | 0.879 | 0.892 | 0.831 | 0.917 | 0.913 | 0.898 |
| C - heat loss | 0.864 | 0.893 | 0.841 | 0.917 | 0.912 | 0.914 |

## Settings

- RF: `{"n_estimators": 300, "min_samples_leaf": 2, "max_features": 0.3333333333333333, "n_jobs": -1, "random_state": 42}`
- LightGBM: `{"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "max_depth": -1, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0, "random_state": 42, "n_jobs": -1, "deterministic": true, "force_col_wise": true, "verbose": -1}`
- Environment: Python 3.13.9, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, matplotlib 3.11.2
