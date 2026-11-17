# Final holdout test (Sub-task 3.7)

Run 2026-11-17T07:38:17.151761+00:00 · commit `751c6b1c1908f86723a3308b80fc75918ba200c0` · config MD5 `e5487b737b2dbd60bf714325de43ac1f` · holdout sha256 `d14bc0b03d1e2e804f09b32b6d10e4c73a2f794cfde416dbf3ed8fcd6d501ea4` · runtime 2.0 s

All errors in original units. Heating targets scored over the heating climates only.

## In-distribution holdout (300 designs, 10 training climates) vs grouped CV

| Variant | Target | Holdout design skill | CV design skill | Holdout pooled R² | CV pooled R² | Drop | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|---|---|
| primary | Heating | 0.989 | 0.988 | 0.9959 | 0.9955 | -0.0003 | 0.239 | 0.460 | PASS |
| primary | Cooling | 0.996 | 0.996 | 0.9983 | 0.9980 | -0.0003 | 0.554 | 0.850 | PASS |
| primary | EUI | 0.939 | 0.935 | 0.9955 | 0.9954 | -0.0000 | 2.808 | 3.603 | PASS |
| primary | Peak heat | 0.990 | 0.985 | 0.9969 | 0.9951 | -0.0018 | 0.246 | 0.350 | ok |
| primary | Peak cool | 0.986 | 0.980 | 0.9879 | 0.9820 | -0.0059 | 0.470 | 0.628 | ok |
| primary | Core | 0.996 | 0.994 | 0.9988 | 0.9985 | -0.0004 | 0.171 | 0.246 | ok |
| primary | South | 0.994 | 0.994 | 0.9953 | 0.9955 | +0.0002 | 3.056 | 5.006 | ok |
| primary | East | 0.993 | 0.992 | 0.9952 | 0.9937 | -0.0015 | 3.790 | 5.514 | ok |
| primary | North | 0.995 | 0.992 | 0.9968 | 0.9949 | -0.0019 | 2.896 | 4.143 | ok |
| primary | West | 0.994 | 0.988 | 0.9965 | 0.9938 | -0.0027 | 2.766 | 3.976 | ok |
| all_mlp | Heating | 0.998 | 0.996 | 0.9993 | 0.9985 | -0.0007 | 0.112 | 0.192 | PASS (report) |
| all_mlp | Cooling | 0.996 | 0.996 | 0.9983 | 0.9980 | -0.0003 | 0.554 | 0.850 | PASS (report) |
| all_mlp | EUI | 0.963 | 0.959 | 0.9972 | 0.9971 | -0.0001 | 2.371 | 2.831 | PASS (report) |
| all_mlp | Peak heat | 0.990 | 0.985 | 0.9969 | 0.9951 | -0.0018 | 0.246 | 0.350 | ok (report) |
| all_mlp | Peak cool | 0.986 | 0.980 | 0.9879 | 0.9820 | -0.0059 | 0.470 | 0.628 | ok (report) |
| all_mlp | Core | 0.996 | 0.994 | 0.9988 | 0.9985 | -0.0004 | 0.171 | 0.246 | ok (report) |
| all_mlp | South | 0.994 | 0.994 | 0.9953 | 0.9955 | +0.0002 | 3.056 | 5.006 | ok (report) |
| all_mlp | East | 0.993 | 0.992 | 0.9952 | 0.9937 | -0.0015 | 3.790 | 5.514 | ok (report) |
| all_mlp | North | 0.995 | 0.992 | 0.9968 | 0.9949 | -0.0019 | 2.896 | 4.143 | ok (report) |
| all_mlp | West | 0.994 | 0.988 | 0.9965 | 0.9938 | -0.0027 | 2.766 | 3.976 | ok (report) |

**S3 PASS** (primary targets: holdout design skill ≥ 0.88 and pooled R² ≤ 0.03 below CV).

## London (100 designs, unseen climate)

Climate level `extrapolation`, distance 2.388 (nearest chicago); median heating 6.21 kWh/m².

| Variant | Target | Bias % | MAPE % | MAE | RMSE | R² | R² bias removed | Spearman | Top-10 % hit |
|---|---|---|---|---|---|---|---|---|---|
| primary | Heating | -15.9 | 13.5 | 1.286 | 1.764 | 0.856 | 0.927 | 0.994 | 0.90 |
| primary | Cooling | +13.7 | 14.3 | 7.062 | 7.310 | 0.426 | 0.962 | 0.978 | 0.80 |
| primary | EUI | -0.7 | 1.5 | 1.260 | 1.686 | 0.943 | 0.950 | 0.978 | 0.80 |
| primary | Peak heat | -15.9 | 18.2 | 1.842 | 2.000 | 0.706 | 0.922 | 0.978 | 0.80 |
| primary | Peak cool | +25.6 | 27.4 | 5.715 | 6.092 | -0.258 | 0.849 | 0.891 | 0.80 |
| primary | Core | +5.4 | 5.6 | 2.701 | 2.841 | 0.504 | 0.953 | 0.972 | 0.80 |
| primary | South | +47.5 | 59.7 | 27.377 | 29.771 | 0.439 | 0.913 | 0.969 | 0.90 |
| primary | East | +25.4 | 39.8 | 18.889 | 21.339 | 0.822 | 0.948 | 0.969 | 0.90 |
| primary | North | +20.5 | 39.5 | 12.352 | 13.384 | 0.858 | 0.952 | 0.980 | 0.90 |
| primary | West | +39.8 | 54.0 | 17.611 | 19.435 | 0.515 | 0.889 | 0.933 | 0.80 |
| all_mlp | Heating | -0.3 | 6.1 | 0.326 | 0.399 | 0.993 | 0.993 | 0.998 | 0.90 |
| all_mlp | Cooling | +13.7 | 14.3 | 7.062 | 7.310 | 0.426 | 0.962 | 0.978 | 0.80 |
| all_mlp | EUI | +6.7 | 6.7 | 5.443 | 5.637 | 0.364 | 0.952 | 0.982 | 0.80 |
| all_mlp | Peak heat | -15.9 | 18.2 | 1.842 | 2.000 | 0.706 | 0.922 | 0.978 | 0.80 |
| all_mlp | Peak cool | +25.6 | 27.4 | 5.715 | 6.092 | -0.258 | 0.849 | 0.891 | 0.80 |
| all_mlp | Core | +5.4 | 5.6 | 2.701 | 2.841 | 0.504 | 0.953 | 0.972 | 0.80 |
| all_mlp | South | +47.5 | 59.7 | 27.377 | 29.771 | 0.439 | 0.913 | 0.969 | 0.90 |
| all_mlp | East | +25.4 | 39.8 | 18.889 | 21.339 | 0.822 | 0.948 | 0.969 | 0.90 |
| all_mlp | North | +20.5 | 39.5 | 12.352 | 13.384 | 0.858 | 0.952 | 0.980 | 0.90 |
| all_mlp | West | +39.8 | 54.0 | 17.611 | 19.435 | 0.515 | 0.889 | 0.933 | 0.80 |

**S4 WARN** (MAPE ≤ 10 % cooling/EUI, ≤ 25 % heating; Spearman ≥ 0.90; primary variant).

## 90 % intervals

| Target | In-distribution coverage (standard) | Mean width | London coverage (new-site) | London coverage (standard, for comparison) | Status |
|---|---|---|---|---|---|
| Heating | 0.919 (n 210) | 1.505 | 0.870 | 0.420 | PASS |
| Cooling | 0.947 (n 300) | 2.990 | 0.040 | 0.020 | PASS |
| EUI | 0.933 (n 300) | 12.095 | 1.000 | 1.000 | PASS |
| Peak heat | 0.976 (n 210) | 1.747 | 1.000 | 0.150 | WARN |
| Peak cool | 0.927 (n 300) | 2.543 | 0.960 | 0.050 | PASS |
| Core | 0.940 (n 300) | 0.934 | 1.000 | 0.000 | PASS |
| South | 0.943 (n 300) | 14.943 | 0.400 | 0.030 | PASS |
| East | 0.943 (n 300) | 21.320 | 0.690 | 0.210 | PASS |
| North | 0.957 (n 300) | 16.310 | 0.830 | 0.200 | WARN |
| West | 0.943 (n 300) | 15.270 | 0.580 | 0.150 | PASS |

**S7 WARN** (in-distribution coverage 0.85–0.95, all 10 targets).

## Gate and Phase 4 decision

| Check | Status |
|---|---|
| S0 | PASS |
| S1 | PASS |
| S2 | PASS |
| S3 | PASS |
| S4 | WARN |
| S5 | PASS |
| S6 | OK_after_adjudication |
| S7 | WARN |
| S8 | PASS |

**Phase 4: REVIEW** (NO-GO on any FAIL, REVIEW on any WARN, else GO).
