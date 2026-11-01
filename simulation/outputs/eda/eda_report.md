# Phase 2.8 - EDA report (Phase 3 go/no-go)

Generated 2026-10-18T09:41:56+00:00 by `eda/run_eda.py`.
Dataset: `dataset_training.csv` sha256 `b826d61b541f…`, `dataset_holdout.csv` `d14bc0b03d1e…`. RF: {'n_estimators': 300, 'min_samples_leaf': 2, 'max_features': 0.3333333333333333, 'n_jobs': -1, 'random_state': 42}.

## Gate and decision

- **Gate: NO-GO**
- **Decision: PROCEED (documented exception)**: all data checks pass; only baseline-model checks failed (['R2:eui_kwh_m2']) and design skill still rises with training size, so the shortfall is attributed to the untuned RF, not the dataset. Phase 3 must exceed the thresholds.
- Rule: Option C (plan §5.1, agreed 17 Oct 2026). Failed checks: R2:eui_kwh_m2

## Checks

| id | check | severity | status | detail |
|---|---|---|---|---|
| G0 | data loaded via load_dataset (sha256 verified) | FAIL | PASS | training 3000 rows, holdout 400 rows |
| C1 | input uniformity per climate (KS / chi2) | WARN | PASS | 0/210 (input, climate) pairs with p < 0.01; min p 1 |
| C2 | design-input decorrelation per climate | WARN | PASS | max |rho| 0.033 (floor_area ~ n_floors, phoenix); limit 0.15 |
| P1 | physics direction P1 | FAIL | PASS | rho +0.65..+0.83 |
| P2 | physics direction P2 | FAIL | PASS | rho +0.70..+0.74 |
| P3-wall | physics direction P3-wall | FAIL | WARN | rho -0.12..-0.08; not significant in ['atlanta', 'chicago', 'denver', 'minneapolis', 'new_york', 'phoenix', 'san_francisco'] |
| P3-roof | physics direction P3-roof | FAIL | WARN | rho -0.16..-0.13; not significant in ['atlanta', 'phoenix', 'san_francisco'] |
| P4-north | physics direction P4-north | FAIL | PASS | rho +0.27..+0.69 |
| P4-east | physics direction P4-east | FAIL | PASS | rho +0.40..+0.70 |
| P4-south | physics direction P4-south | FAIL | PASS | rho +0.26..+0.67 |
| P4-west | physics direction P4-west | FAIL | PASS | rho +0.24..+0.68 |
| P5-north | physics direction P5-north | WARN | INFO | rho -0.23..-0.13; not significant in ['phoenix'] |
| P5-east | physics direction P5-east | WARN | PASS | rho -0.26..-0.18 |
| P5-south | physics direction P5-south | WARN | INFO | rho -0.20..-0.14; not significant in ['new_york', 'san_francisco'] |
| P5-west | physics direction P5-west | WARN | INFO | rho -0.13..-0.03; not significant in ['atlanta', 'chennai', 'chicago', 'colombo', 'denver', 'miami', 'minneapolis', 'new_york', 'phoenix', 'san_francisco'] |
| P6 | physics direction P6 | WARN | INFO | rho +0.06..+0.12; not significant in ['atlanta', 'chicago', 'denver', 'minneapolis', 'new_york', 'phoenix', 'san_francisco'] |
| P7 | physics direction P7 | WARN | PASS | rho -0.36..-0.27 |
| O1 | W/SW-facing facades cool more than N-facing (lat > 25) | WARN | PASS | 8/8 climates pass |
| R1:heating_demand_kwh_m2 | pooled R2 >= 0.85 | FAIL | PASS | R2 0.891 |
| R2:heating_demand_kwh_m2 | design skill >= 0.7 | FAIL | PASS | skill 0.704 over 7 climates |
| R1:cooling_demand_kwh_m2 | pooled R2 >= 0.85 | FAIL | PASS | R2 0.886 |
| R2:cooling_demand_kwh_m2 | design skill >= 0.7 | FAIL | PASS | skill 0.749 over 10 climates |
| R1:eui_kwh_m2 | pooled R2 >= 0.85 | FAIL | PASS | R2 0.977 |
| R2:eui_kwh_m2 | design skill >= 0.7 | FAIL | FAIL | skill 0.680 over 10 climates |
| R3:peak_heating_w_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.930, design skill 0.764 |
| R3:peak_cooling_w_m2 | pooled R2 >= 0.85 | WARN | WARN | R2 0.776, design skill 0.745 |
| R3:zone_cooling_core_kwh_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.965, design skill 0.868 |
| R3:zone_cooling_south_kwh_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.881, design skill 0.830 |
| R3:zone_cooling_east_kwh_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.888, design skill 0.852 |
| R3:zone_cooling_north_kwh_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.886, design skill 0.826 |
| R3:zone_cooling_west_kwh_m2 | pooled R2 >= 0.85 | WARN | PASS | R2 0.876, design skill 0.766 |

## Signal: grouped 5-fold CV on design_id (E2-E4)

`design_skill` = 1 − SSE(model)/SSE(climate-mean predictor). Heating targets: skill over the heating climates only (atlanta, chicago, denver, minneapolis, new_york, phoenix, san_francisco).

| target | r2 | climate_only_r2 | design_skill | skill_climates | within_r2_mean | mae | rmse | cv_rmse |
|---|---|---|---|---|---|---|---|---|
| heating_demand_kwh_m2 | 0.891 | 0.631 | 0.704 | 7 | 0.687 | 1.232 | 2.314 | 0.426 |
| cooling_demand_kwh_m2 | 0.886 | 0.545 | 0.749 | 10 | 0.753 | 4.784 | 6.845 | 0.096 |
| eui_kwh_m2 | 0.977 | 0.930 | 0.680 | 10 | 0.661 | 5.861 | 8.024 | 0.056 |
| peak_heating_w_m2 | 0.930 | 0.705 | 0.764 | 7 | 0.772 | 1.028 | 1.663 | 0.224 |
| peak_cooling_w_m2 | 0.776 | 0.122 | 0.745 | 10 | 0.754 | 1.948 | 2.758 | 0.105 |
| zone_cooling_core_kwh_m2 | 0.965 | 0.733 | 0.868 | 10 | 0.849 | 0.897 | 1.298 | 0.023 |
| zone_cooling_south_kwh_m2 | 0.881 | 0.298 | 0.830 | 10 | 0.835 | 16.469 | 24.145 | 0.216 |
| zone_cooling_east_kwh_m2 | 0.888 | 0.244 | 0.852 | 10 | 0.857 | 19.254 | 27.742 | 0.209 |
| zone_cooling_north_kwh_m2 | 0.886 | 0.345 | 0.826 | 10 | 0.831 | 16.464 | 24.699 | 0.221 |
| zone_cooling_west_kwh_m2 | 0.876 | 0.469 | 0.766 | 10 | 0.758 | 15.180 | 22.389 | 0.232 |

### Within-climate R² (primary targets)

| climate_key | heating_demand_kwh_m2 | cooling_demand_kwh_m2 | eui_kwh_m2 |
|---|---|---|---|
| minneapolis | 0.689 | 0.734 | 0.598 |
| chicago | 0.716 | 0.770 | 0.627 |
| denver | 0.712 | 0.694 | 0.583 |
| new_york | 0.737 | 0.796 | 0.600 |
| atlanta | 0.689 | 0.744 | 0.697 |
| san_francisco | 0.652 | 0.788 | 0.667 |
| phoenix | 0.611 | 0.749 | 0.680 |
| miami | 0.694 | 0.779 | 0.725 |
| chennai | 0.722 | 0.734 | 0.717 |
| colombo | 0.731 | 0.743 | 0.713 |

### Random 80/20 split vs grouped CV (anchor leakage gap)

| target | r2_grouped | design_skill_grouped | r2_random | design_skill_random | r2_gap | skill_gap |
|---|---|---|---|---|---|---|
| heating_demand_kwh_m2 | 0.891 | 0.704 | 0.888 | 0.703 | -0.003 | -0.001 |
| cooling_demand_kwh_m2 | 0.886 | 0.749 | 0.917 | 0.820 | 0.032 | 0.071 |
| eui_kwh_m2 | 0.977 | 0.680 | 0.980 | 0.708 | 0.003 | 0.028 |
| peak_heating_w_m2 | 0.930 | 0.764 | 0.948 | 0.823 | 0.018 | 0.058 |
| peak_cooling_w_m2 | 0.776 | 0.745 | 0.823 | 0.796 | 0.047 | 0.051 |
| zone_cooling_core_kwh_m2 | 0.965 | 0.868 | 0.974 | 0.896 | 0.010 | 0.028 |
| zone_cooling_south_kwh_m2 | 0.881 | 0.830 | 0.917 | 0.885 | 0.036 | 0.055 |
| zone_cooling_east_kwh_m2 | 0.888 | 0.852 | 0.920 | 0.900 | 0.031 | 0.048 |
| zone_cooling_north_kwh_m2 | 0.886 | 0.826 | 0.914 | 0.865 | 0.029 | 0.039 |
| zone_cooling_west_kwh_m2 | 0.876 | 0.766 | 0.901 | 0.824 | 0.025 | 0.057 |

### Learning curve (diagnostic for the decision rule)

| fraction | n_designs | n_rows | target | design_skill | r2 |
|---|---|---|---|---|---|
| 0.400 | 840 | 1236 | heating_demand_kwh_m2 | 0.624 | 0.862 |
| 0.400 | 840 | 1236 | cooling_demand_kwh_m2 | 0.641 | 0.841 |
| 0.400 | 840 | 1236 | eui_kwh_m2 | 0.540 | 0.968 |
| 0.700 | 1470 | 2145 | heating_demand_kwh_m2 | 0.658 | 0.876 |
| 0.700 | 1470 | 2145 | cooling_demand_kwh_m2 | 0.729 | 0.875 |
| 0.700 | 1470 | 2145 | eui_kwh_m2 | 0.650 | 0.975 |
| 1.000 | 2100 | 3000 | heating_demand_kwh_m2 | 0.704 | 0.891 |
| 1.000 | 2100 | 3000 | cooling_demand_kwh_m2 | 0.749 | 0.886 |
| 1.000 | 2100 | 3000 | eui_kwh_m2 | 0.680 | 0.977 |

## Physics directions (Spearman ρ within climate)

| rule | severity | status | rho_min | rho_max | opposite | weak |
|---|---|---|---|---|---|---|
| P1 | FAIL | PASS | 0.649 | 0.829 |  |  |
| P2 | FAIL | PASS | 0.697 | 0.740 |  |  |
| P3-wall | FAIL | WARN | -0.122 | -0.085 |  | atlanta, chicago, denver, minneapolis, new_york, phoenix, san_francisco |
| P3-roof | FAIL | WARN | -0.158 | -0.129 |  | atlanta, phoenix, san_francisco |
| P4-north | FAIL | PASS | 0.275 | 0.689 |  |  |
| P4-east | FAIL | PASS | 0.403 | 0.697 |  |  |
| P4-south | FAIL | PASS | 0.265 | 0.673 |  |  |
| P4-west | FAIL | PASS | 0.239 | 0.683 |  |  |
| P5-north | WARN | INFO | -0.229 | -0.132 |  | phoenix |
| P5-east | WARN | PASS | -0.260 | -0.183 |  |  |
| P5-south | WARN | INFO | -0.199 | -0.140 |  | new_york, san_francisco |
| P5-west | WARN | INFO | -0.129 | -0.027 |  | atlanta, chennai, chicago, colombo, denver, miami, minneapolis, new_york, phoenix, san_francisco |
| P6 | WARN | INFO | 0.058 | 0.116 |  | atlanta, chicago, denver, minneapolis, new_york, phoenix, san_francisco |
| P7 | WARN | PASS | -0.358 | -0.271 |  |  |

### Roof insulation and core cooling, by storey count (P8, hot climates)

| climate_key | subset | n | rho | p | core_cooling_median |
|---|---|---|---|---|---|
| chennai | n_floors=1 | 60 | -0.956 | 0.000 | 70.661 |
| chennai | n_floors>=2 | 240 | -0.652 | 0.000 | 63.800 |
| colombo | n_floors=1 | 60 | -0.952 | 0.000 | 70.119 |
| colombo | n_floors>=2 | 240 | -0.739 | 0.000 | 63.776 |
| miami | n_floors=1 | 60 | -0.903 | 0.000 | 62.901 |
| miami | n_floors>=2 | 240 | -0.458 | 0.000 | 60.030 |
| phoenix | n_floors=1 | 60 | -0.779 | 0.000 | 61.082 |
| phoenix | n_floors>=2 | 240 | -0.421 | 0.000 | 59.779 |

## Facade azimuth (O1)

| climate_key | latitude | applies | mean_resid_W_SW | mean_resid_N | pass |
|---|---|---|---|---|---|
| atlanta | 33.630 | True | 3.453 | -35.043 | True |
| chennai | 12.994 | False | 15.861 | -32.326 | None |
| chicago | 41.960 | True | 16.665 | -30.317 | True |
| colombo | 6.905 | False | 10.140 | -31.189 | None |
| denver | 39.833 | True | 20.275 | -42.908 | True |
| miami | 25.755 | True | 18.641 | -41.571 | True |
| minneapolis | 44.883 | True | 15.751 | -30.579 | True |
| new_york | 40.779 | True | 16.325 | -30.019 | True |
| phoenix | 33.428 | True | 14.687 | -58.709 | True |
| san_francisco | 37.620 | True | 29.408 | -44.384 | True |

## SHAP: share of mean |SHAP| by group

| target | climate | geometry | opaque_envelope | glazing | window_area | shading | orientation |
|---|---|---|---|---|---|---|---|
| heating_demand_kwh_m2 | 64.6% | 8.2% | 2.5% | 20.7% | 2.8% | 0.9% | 0.3% |
| cooling_demand_kwh_m2 | 53.8% | 9.9% | 2.0% | 26.7% | 5.6% | 1.6% | 0.3% |
| eui_kwh_m2 | 78.6% | 7.3% | 1.2% | 8.8% | 3.3% | 0.7% | 0.1% |
| peak_heating_w_m2 | 63.9% | 17.4% | 2.7% | 13.4% | 1.7% | 0.8% | 0.2% |
| peak_cooling_w_m2 | 20.1% | 28.0% | 2.2% | 32.3% | 14.2% | 2.4% | 0.8% |
| zone_cooling_core_kwh_m2 | 69.9% | 11.5% | 12.0% | 5.2% | 0.8% | 0.6% | 0.1% |
| zone_cooling_south_kwh_m2 | 33.8% | 3.2% | 1.0% | 23.5% | 19.7% | 4.8% | 14.0% |
| zone_cooling_east_kwh_m2 | 31.1% | 4.6% | 1.0% | 29.5% | 24.5% | 8.7% | 0.6% |
| zone_cooling_north_kwh_m2 | 35.7% | 2.2% | 0.9% | 22.6% | 19.8% | 5.0% | 13.8% |
| zone_cooling_west_kwh_m2 | 46.1% | 3.1% | 1.3% | 21.7% | 17.6% | 2.0% | 8.2% |

## EUI decomposition (mean kWh/m²·yr)

EUI = elec + system heating + system cooling (`closure` ≈ 0). `oa_net` = (system − zone) heating + (system − zone) cooling: positive where outdoor air must be conditioned, negative where it gives free cooling.

| climate_key | elec_kwh_m2 | sys_heat_kwh_m2 | sys_cool_kwh_m2 | heating_demand_kwh_m2 | cooling_demand_kwh_m2 | oa_net | eui_kwh_m2 | closure |
|---|---|---|---|---|---|---|---|---|
| minneapolis | 48.9 | 44.4 | 32.4 | 17.0 | 55.5 | 4.3 | 125.7 | -0.0 |
| chicago | 48.9 | 31.1 | 36.7 | 12.3 | 57.3 | -1.8 | 116.7 | 0.0 |
| denver | 48.9 | 20.1 | 30.7 | 8.9 | 58.9 | -17.0 | 99.7 | 0.0 |
| new_york | 48.9 | 19.8 | 40.2 | 8.1 | 59.0 | -7.1 | 108.8 | 0.0 |
| atlanta | 48.9 | 7.8 | 63.9 | 3.7 | 65.2 | 2.9 | 120.6 | 0.0 |
| san_francisco | 48.9 | 3.3 | 26.9 | 1.7 | 60.6 | -32.2 | 79.0 | 0.0 |
| phoenix | 48.9 | 1.6 | 96.0 | 1.2 | 88.3 | 8.0 | 146.4 | 0.0 |
| miami | 48.9 | 0.0 | 131.8 | 0.3 | 81.5 | 50.0 | 180.7 | -0.0 |
| chennai | 48.9 | 0.0 | 187.3 | 0.6 | 94.4 | 92.4 | 236.2 | 0.0 |
| colombo | 48.9 | 0.0 | 182.7 | 0.5 | 91.6 | 90.5 | 231.5 | 0.0 |

## Largest out-of-fold residuals (R4, top 5 per primary target; all 20 in top_residuals.csv)

| target | run_id | y | yhat | abs_err | outlier_targets |
|---|---|---|---|---|---|
| cooling_demand_kwh_m2 | main_B0977_denver | 158.381 | 81.881 | 76.500 | cooling_demand_kwh_m2;eui_kwh_m2;peak_cooling_w_m2;zone_cooling_south_kwh_m2 |
| eui_kwh_m2 | main_B0977_denver | 175.619 | 114.175 | 61.444 | cooling_demand_kwh_m2;eui_kwh_m2;peak_cooling_w_m2;zone_cooling_south_kwh_m2 |
| eui_kwh_m2 | main_B1690_phoenix | 222.086 | 173.308 | 48.778 | cooling_demand_kwh_m2;eui_kwh_m2;peak_cooling_w_m2;zone_cooling_south_kwh_m2 |
| eui_kwh_m2 | main_B1783_phoenix | 208.414 | 160.386 | 48.028 |  |
| cooling_demand_kwh_m2 | main_B1690_phoenix | 164.173 | 117.751 | 46.422 | cooling_demand_kwh_m2;eui_kwh_m2;peak_cooling_w_m2;zone_cooling_south_kwh_m2 |
| eui_kwh_m2 | main_B0761_colombo | 304.914 | 261.966 | 42.948 | eui_kwh_m2 |
| eui_kwh_m2 | main_B0460_chicago | 176.431 | 133.934 | 42.497 | eui_kwh_m2;peak_heating_w_m2;peak_cooling_w_m2 |
| cooling_demand_kwh_m2 | main_B1783_phoenix | 152.397 | 112.562 | 39.835 |  |
| cooling_demand_kwh_m2 | main_B1644_phoenix | 151.411 | 112.728 | 38.683 |  |
| cooling_demand_kwh_m2 | main_B0761_colombo | 151.279 | 115.501 | 35.778 | eui_kwh_m2 |
| heating_demand_kwh_m2 | main_B1324_minneapolis | 51.535 | 28.774 | 22.761 |  |
| heating_demand_kwh_m2 | main_B0460_chicago | 39.678 | 19.612 | 20.066 | eui_kwh_m2;peak_heating_w_m2;peak_cooling_w_m2 |
| heating_demand_kwh_m2 | main_A086_minneapolis | 43.761 | 24.440 | 19.321 |  |
| heating_demand_kwh_m2 | main_B1393_minneapolis | 44.038 | 27.398 | 16.640 |  |
| heating_demand_kwh_m2 | main_B0455_chicago | 36.698 | 21.839 | 14.859 | peak_heating_w_m2 |

Each must be physically explainable. Review and record the explanation in the 2.8 plan (Rev 2).

## Figures

- `figures/F01_input_histograms.png` / `.pdf`
- `figures/F02_input_correlation.png` / `.pdf`
- `figures/F03_climate_space.png` / `.pdf`
- `figures/F04_targets_by_climate.png` / `.pdf`
- `figures/F05_target_transforms.png` / `.pdf`
- `figures/F06_target_correlation.png` / `.pdf`
- `figures/F07_eui_decomposition.png` / `.pdf`
- `figures/F08_physics_directions.png` / `.pdf`
- `figures/F09_facade_azimuth.png` / `.pdf`
- `figures/F10_core_nfloors.png` / `.pdf`
- `figures/F11_rf_parity.png` / `.pdf`
- `figures/F12_shap_features.png` / `.pdf`
- `figures/F13_shap_groups.png` / `.pdf`
- `figures/F14_shap_by_climate.png` / `.pdf`
- `figures/F15_learning_curve.png` / `.pdf`

## Environment

```
{
  "python": "3.13.9",
  "platform": "macOS-26.6.2-arm64-arm-64bit-Mach-O",
  "numpy": "2.5.3",
  "pandas": "3.0.6",
  "scipy": "1.18.1",
  "scikit-learn": "1.9.1",
  "shap": "0.52.0",
  "matplotlib": "3.11.2"
}
```
