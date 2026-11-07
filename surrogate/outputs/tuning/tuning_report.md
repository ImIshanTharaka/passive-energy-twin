# Hyper-parameter tuning (Sub-task 3.3, run B)

Updated 2026-11-06T19:08:04 UTC. Optuna TPE (seed 42, 10 random start-up trials), no pruning, budgets {'rf': 60, 'xgb': 60, 'lgbm': 60, 'mlp': 30, 'gpr': 0}. Objective: grouped-CV RMSE on scored rows (MLP: mean RMSE/SD over the 3 primary targets). Search spaces: config.yaml `benchmark_3_3.search_spaces`.

| study | trials | best objective | best trial | design skill at best | trials to reach best + 1 % | parameters at a range edge |
|---|---|---|---|---|---|---|
| lgbm__cooling_demand_kwh_m2 | 60 | 2.4772 | 58 | Cooling 0.967 | 59 | num_leaves at low end (8) |
| lgbm__eui_kwh_m2 | 60 | 2.5161 | 59 | EUI 0.969 | 58 | num_leaves at low end (8) |
| lgbm__heating_demand_kwh_m2 | 60 | 0.7777 | 51 | Heating 0.977 | 52 | — |
| lgbm__peak_cooling_w_m2 | 60 | 1.1275 | 45 | Peak cooling 0.957 | 32 | — |
| lgbm__peak_heating_w_m2 | 60 | 0.6632 | 56 | Peak heating 0.973 | 46 | min_child_samples at low end (6) |
| lgbm__zone_cooling_core_kwh_m2 | 60 | 0.4386 | 57 | Zone core 0.985 | 58 | — |
| lgbm__zone_cooling_east_kwh_m2 | 60 | 14.3012 | 54 | Zone east 0.961 | 52 | num_leaves at low end (8) |
| lgbm__zone_cooling_north_kwh_m2 | 60 | 12.0333 | 42 | Zone north 0.959 | 43 | num_leaves at low end (8) |
| lgbm__zone_cooling_south_kwh_m2 | 60 | 12.0617 | 53 | Zone south 0.958 | 54 | reg_lambda at low end (1.130601868885165e-08) |
| lgbm__zone_cooling_west_kwh_m2 | 60 | 9.1991 | 22 | Zone west 0.961 | 23 | — |
| mlp__all | 30 | 0.0393 | 26 | Cooling 0.996, EUI 0.987, Heating 0.996 | 27 | dropout at low end (0.0006709767534828354) |
| rf__cooling_demand_kwh_m2 | 60 | 3.9999 | 36 | Cooling 0.914 | 37 | min_samples_leaf at low end (1) |
| rf__eui_kwh_m2 | 60 | 4.0397 | 42 | EUI 0.919 | 20 | min_samples_leaf at low end (1); max_samples at high end (0.9939069990048981) |
| rf__heating_demand_kwh_m2 | 60 | 1.1682 | 58 | Heating 0.947 | 18 | min_samples_leaf at low end (1) |
| xgb__cooling_demand_kwh_m2 | 60 | 2.2860 | 19 | Cooling 0.972 | 20 | n_estimators at high end (1948); max_depth at low end (3) |
| xgb__eui_kwh_m2 | 60 | 2.4649 | 52 | EUI 0.970 | 40 | max_depth at low end (3) |
| xgb__heating_demand_kwh_m2 | 60 | 0.7127 | 45 | Heating 0.980 | 46 | max_depth at low end (3) |

A parameter at a range edge means the best value may lie outside the searched range; it is reported (sensitivity note), not acted on — the ranges are frozen.

## Parameter importance (fANOVA share of objective variance)

- **lgbm__cooling_demand_kwh_m2**: reg_alpha 0.39, min_child_samples 0.23, n_estimators 0.20, learning_rate 0.11, subsample 0.05
- **lgbm__eui_kwh_m2**: learning_rate 0.29, n_estimators 0.25, min_child_samples 0.22, reg_alpha 0.17, colsample_bytree 0.02
- **lgbm__heating_demand_kwh_m2**: n_estimators 0.75, num_leaves 0.08, min_child_samples 0.06, subsample 0.06, colsample_bytree 0.03
- **lgbm__peak_cooling_w_m2**: n_estimators 0.49, reg_alpha 0.20, min_child_samples 0.16, subsample 0.09, learning_rate 0.03
- **lgbm__peak_heating_w_m2**: reg_alpha 0.51, min_child_samples 0.24, n_estimators 0.13, learning_rate 0.09, colsample_bytree 0.02
- **lgbm__zone_cooling_core_kwh_m2**: n_estimators 0.51, reg_alpha 0.26, subsample 0.09, colsample_bytree 0.05, min_child_samples 0.05
- **lgbm__zone_cooling_east_kwh_m2**: reg_alpha 0.60, learning_rate 0.12, n_estimators 0.11, colsample_bytree 0.06, num_leaves 0.04
- **lgbm__zone_cooling_north_kwh_m2**: reg_alpha 0.72, n_estimators 0.14, learning_rate 0.05, colsample_bytree 0.03, min_child_samples 0.03
- **lgbm__zone_cooling_south_kwh_m2**: reg_alpha 0.68, n_estimators 0.17, min_child_samples 0.07, learning_rate 0.05, num_leaves 0.03
- **lgbm__zone_cooling_west_kwh_m2**: n_estimators 0.55, learning_rate 0.27, min_child_samples 0.12, subsample 0.04, num_leaves 0.01
- **mlp__all**: dropout 0.33, width 0.29, lr 0.20, n_layers 0.11, batch_size 0.05
- **rf__cooling_demand_kwh_m2**: max_features 0.51, min_samples_leaf 0.45, max_samples 0.02, max_depth 0.02
- **rf__eui_kwh_m2**: min_samples_leaf 0.54, max_features 0.40, max_samples 0.04, max_depth 0.02
- **rf__heating_demand_kwh_m2**: max_features 0.55, min_samples_leaf 0.34, max_depth 0.09, max_samples 0.03
- **xgb__cooling_demand_kwh_m2**: reg_alpha 0.45, n_estimators 0.45, max_depth 0.04, learning_rate 0.02, subsample 0.02
- **xgb__eui_kwh_m2**: n_estimators 0.33, learning_rate 0.26, reg_alpha 0.19, min_child_weight 0.16, subsample 0.04
- **xgb__heating_demand_kwh_m2**: reg_alpha 0.59, n_estimators 0.24, learning_rate 0.09, colsample_bytree 0.05, max_depth 0.01
