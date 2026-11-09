# Sub-task 3.4 — climate generalisation

Created 2026-11-09T06:54:31.728515+00:00 · status **final** · rule: M10 (skill ≥ 0.9 → lowest LOCO MAPE; ties ≤ 0.5 points → best LightGBM)

## Selection

| Target | Model | Inputs | Form | Grouped skill | LOCO MAPE % | LOCO median \|bias\| % | Rule |
|---|---|---|---|---|---|---|---|
| Heating | Gaussian process | C-G+I | log1p | 0.988 | 10.0 | 6.2 | lowest LOCO MAPE (ties within 0.5 points -> best LightGBM) |
| Cooling | Neural network (MLP) | C-G | direct | 0.996 | 3.3 | 2.7 | lowest LOCO MAPE (ties within 0.5 points -> best LightGBM) |
| EUI | XGBoost | C-G+I | decomposed | 0.935 | 4.8 | 3.9 | lowest LOCO MAPE (ties within 0.5 points -> best LightGBM) |
| Peak heating | Neural network (MLP) | C-G+I | direct | 0.985 | 22.2 | 13.2 | highest grouped skill (ties within 0.005 -> best LightGBM) |

Fix 4 (F5, trigger > 5.0 % median |bias|): needed for [{'target': 'heating_demand_kwh_m2', 'model': 'gpr', 'form': 'log1p', 'loco_median_abs_bias_pct': 6.2021808366127225}]; run: True.

**M7:** not triggered (limit 10.0 %): Heating 6.2 %, EUI 3.9 %

Families of the final picks: gpr, mlp, xgb (one family: False).

## Heating — all candidates

| Model | Inputs | Form | Grouped skill | Pooled R² | LOCO MAPE % | median \|bias\| % | max \|bias\| % (climate) | bias-removed R² | LOCO Spearman |
|---|---|---|---|---|---|---|---|---|---|
| gpr | C-G+I-r | log1p | 0.988 | 0.996 | 10.0 | 6.2 | 15.9 (atlanta) | 0.982 | 0.991 |
| gpr | C-G+I | log1p | 0.988 | 0.996 | 10.0 | 6.2 | 15.9 (atlanta) | 0.982 | 0.991 |
| mlp | C-G+I | ratio | 0.996 | 0.999 | 13.3 | 4.8 | 26.7 (san_francisco) | 0.988 | 0.996 |
| xgb | C-G+I | ratio | 0.976 | 0.991 | 14.4 | 9.4 | 19.8 (phoenix) | 0.960 | 0.989 |
| gpr | C-G+I | ratio | 0.992 | 0.997 | 14.7 | 6.3 | 38.9 (san_francisco) | 0.977 | 0.986 |
| rf | C-G+I | ratio | 0.955 | 0.983 | 18.2 | 8.5 | 42.4 (san_francisco) | 0.961 | 0.987 |
| lgbm | C-G+I | ratio | 0.871 | 0.952 | 21.9 | 10.0 | 36.0 (san_francisco) | 0.880 | 0.959 |
| mlp | C-G+I | direct | 0.997 | 0.999 | 22.5 | 15.6 | 95.7 (san_francisco) | 0.968 | 0.996 |
| xgb | C-G+I | log1p | 0.983 | 0.994 | 25.5 | 20.6 | 61.4 (san_francisco) | 0.947 | 0.993 |
| mlp | C-G | direct | 0.996 | 0.998 | 25.7 | 11.2 | 117.3 (san_francisco) | 0.969 | 0.996 |
| xgb | C-G | direct | 0.980 | 0.993 | 27.0 | 20.6 | 58.4 (san_francisco) | 0.947 | 0.991 |
| gpr | C-G | direct | 0.956 | 0.984 | 27.1 | 26.3 | 74.9 (san_francisco) | 0.861 | 0.981 |
| gpr | C-G+I | direct | 0.958 | 0.984 | 27.1 | 26.3 | 74.9 (san_francisco) | 0.861 | 0.981 |
| xgb | C-G+I | direct | 0.980 | 0.993 | 27.1 | 18.0 | 73.8 (phoenix) | 0.947 | 0.991 |
| lgbm | C-G+I | direct | 0.976 | 0.991 | 31.2 | 26.5 | 101.8 (san_francisco) | 0.933 | 0.989 |
| lgbm | C-G | direct | 0.977 | 0.991 | 31.7 | 28.9 | 100.7 (san_francisco) | 0.933 | 0.991 |
| mlp | C-G+I | log1p | 0.994 | 0.998 | 33.3 | 13.0 | 177.6 (san_francisco) | 0.930 | 0.996 |
| lgbm | C-G+I | log1p | 0.973 | 0.990 | 33.6 | 21.8 | 106.9 (san_francisco) | 0.921 | 0.990 |
| rf | C-G+I | log1p | 0.943 | 0.979 | 35.0 | 23.5 | 119.8 (san_francisco) | 0.883 | 0.987 |
| rf | C-G | direct | 0.947 | 0.981 | 38.9 | 22.4 | 175.1 (san_francisco) | 0.897 | 0.985 |
| rf | C-G+I | direct | 0.947 | 0.980 | 39.3 | 20.8 | 174.6 (san_francisco) | 0.899 | 0.983 |

## Cooling — all candidates

| Model | Inputs | Form | Grouped skill | Pooled R² | LOCO MAPE % | median \|bias\| % | max \|bias\| % (climate) | bias-removed R² | LOCO Spearman |
|---|---|---|---|---|---|---|---|---|---|
| mlp | C-G | direct | 0.996 | 0.998 | 3.3 | 2.7 | 6.9 (phoenix) | 0.980 | 0.993 |
| mlp | C-G+I | direct | 0.995 | 0.998 | 3.8 | 2.2 | 11.5 (phoenix) | 0.975 | 0.993 |
| lgbm | C-G+I | direct | 0.967 | 0.985 | 3.8 | 2.8 | 7.1 (miami) | 0.957 | 0.987 |
| xgb | C-G+I | direct | 0.971 | 0.987 | 4.4 | 2.1 | 12.6 (san_francisco) | 0.953 | 0.988 |
| lgbm | C-G | direct | 0.967 | 0.985 | 4.4 | 2.8 | 8.0 (miami) | 0.952 | 0.982 |
| xgb | C-G | direct | 0.972 | 0.987 | 4.9 | 2.6 | 13.4 (san_francisco) | 0.950 | 0.987 |
| rf | C-G | direct | 0.914 | 0.961 | 5.0 | 2.3 | 9.5 (phoenix) | 0.909 | 0.972 |
| rf | C-G+I | direct | 0.915 | 0.961 | 5.1 | 2.8 | 10.0 (phoenix) | 0.907 | 0.972 |
| gpr | C-G | direct | 0.956 | 0.980 | 12.9 | 5.3 | 44.7 (denver) | 0.895 | 0.973 |
| gpr | C-G+I | direct | 0.956 | 0.980 | 13.5 | 5.3 | 44.7 (denver) | 0.895 | 0.971 |

## EUI — all candidates

| Model | Inputs | Form | Grouped skill | Pooled R² | LOCO MAPE % | median \|bias\| % | max \|bias\| % (climate) | bias-removed R² | LOCO Spearman |
|---|---|---|---|---|---|---|---|---|---|
| xgb | C-G+I | decomposed | 0.935 | 0.995 | 4.8 | 3.9 | 12.8 (phoenix) | 0.948 | 0.979 |
| gpr | C-G | direct | 0.954 | 0.997 | 5.6 | 3.6 | 23.8 (phoenix) | 0.911 | 0.976 |
| lgbm | C-G+I | decomposed | 0.923 | 0.995 | 5.8 | 3.9 | 13.0 (san_francisco) | 0.944 | 0.973 |
| rf | C-G+I | decomposed | 0.886 | 0.992 | 5.8 | 4.5 | 14.8 (phoenix) | 0.911 | 0.969 |
| mlp | C-G+I | decomposed | 0.959 | 0.997 | 5.9 | 3.2 | 17.1 (san_francisco) | 0.968 | 0.990 |
| gpr | C-G+I | direct | 0.953 | 0.997 | 6.2 | 2.3 | 23.8 (phoenix) | 0.918 | 0.976 |
| mlp | C-G+I | direct | 0.990 | 0.999 | 6.7 | 2.9 | 27.2 (san_francisco) | 0.951 | 0.980 |
| gpr | C-G+I | decomposed | 0.924 | 0.995 | 7.4 | 3.6 | 20.7 (denver) | 0.890 | 0.952 |
| mlp | C-G | direct | 0.987 | 0.999 | 8.3 | 4.4 | 35.0 (san_francisco) | 0.935 | 0.970 |
| xgb | C-G | direct | 0.970 | 0.998 | 11.3 | 7.5 | 48.5 (san_francisco) | 0.937 | 0.983 |
| rf | C-G+I | direct | 0.919 | 0.994 | 11.9 | 7.5 | 53.6 (san_francisco) | 0.888 | 0.969 |
| xgb | C-G+I | direct | 0.970 | 0.998 | 12.0 | 7.9 | 45.1 (san_francisco) | 0.936 | 0.983 |
| rf | C-G | direct | 0.919 | 0.994 | 13.1 | 6.6 | 55.3 (san_francisco) | 0.849 | 0.957 |
| lgbm | C-G | direct | 0.969 | 0.998 | 13.5 | 12.6 | 38.4 (san_francisco) | 0.944 | 0.981 |
| lgbm | C-G+I | direct | 0.969 | 0.998 | 13.6 | 13.4 | 38.2 (san_francisco) | 0.941 | 0.981 |

## Peak heating — all candidates

| Model | Inputs | Form | Grouped skill | Pooled R² | LOCO MAPE % | median \|bias\| % | max \|bias\| % (climate) | bias-removed R² | LOCO Spearman |
|---|---|---|---|---|---|---|---|---|---|
| mlp | C-G+I | direct | 0.985 | 0.995 | 22.2 | 13.2 | 52.2 (san_francisco) | 0.929 | 0.985 |
| mlp | C-G | direct | 0.982 | 0.994 | — | — | — (nan) | — | — |
| mlp | C-G+I | ratio | 0.977 | 0.993 | 22.6 | 27.3 | 34.6 (phoenix) | 0.806 | 0.958 |
| lgbm | C-G+I | direct | 0.973 | 0.992 | 18.7 | 9.5 | 43.2 (san_francisco) | 0.906 | 0.982 |
| lgbm | C-G | direct | 0.973 | 0.992 | — | — | — (nan) | — | — |
| lgbm | C-G+I | ratio | 0.946 | 0.984 | 21.6 | 23.5 | 44.5 (atlanta) | 0.812 | 0.973 |
