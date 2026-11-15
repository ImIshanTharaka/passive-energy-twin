# Sub-task 3.6 - interpretability and physics consistency

## Final models (H1)

Fitted on 3000 training rows; files in `outputs/models`.

| Object | SHA-256 (first 12) | Fit s |
|---|---|---|
| gpr_heating | a6dfcf2dab6c | 159.93 |
| xgb_eui | 04df009f0bab | 5.03 |
| mlp_cg | c31e0ef89e55 | 64.55 |
| mlp_physics | f884cadc9668 | 47.74 |

GPR heating features: floor_height, glazing_u, glazing_shgc, hdd18, cdd10, annual_ghi, mean_rh, sol_aperture, sol_aperture_true_S, ua_glass, ua_total, cx_ua_hdd, cx_aperture_ghi, cx_aperture_cdd, cx_infil_hdd

## OAT test (gate S5)

**S5 PASS** - 41 informative (parameter, bound) pairs (1 duplicate of base dropped); 308 counted deltas (|EnergyPlus change| > 2% of base); sign-correct 100.0% (needs >= 90%); strong set 62/62; R2 of surrogate vs EnergyPlus change 0.995.

| Target | Correct / counted |
|---|---|
| cooling_demand_kwh_m2 | 45 / 45 |
| eui_kwh_m2 | 33 / 33 |
| heating_demand_kwh_m2 | 28 / 28 |
| peak_cooling_w_m2 | 43 / 43 |
| peak_heating_w_m2 | 28 / 28 |
| zone_cooling_core_kwh_m2 | 14 / 14 |
| zone_cooling_east_kwh_m2 | 33 / 33 |
| zone_cooling_north_kwh_m2 | 27 / 27 |
| zone_cooling_south_kwh_m2 | 28 / 28 |
| zone_cooling_west_kwh_m2 | 29 / 29 |

Strong-set misses: 0

Other misses: 0

Zero responses (surrogate does not move): 12
- wall_thermal_mass lo / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.099
- wall_thermal_mass hi / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.036
- overhang_depth_n lo / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.254
- overhang_depth_n hi / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.242
- overhang_depth_e lo / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.162
- overhang_depth_e hi / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.219
- overhang_depth_s lo / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.248
- overhang_depth_s hi / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.234
- overhang_depth_w lo / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.019
- overhang_depth_w hi / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.015
- sidefin_depth lo / minneapolis / heating_demand_kwh_m2: EnergyPlus -0.034
- sidefin_depth hi / minneapolis / heating_demand_kwh_m2: EnergyPlus +0.028

## Monotonic sweeps (gate S6)

**S6 WARN** (WARN if more than 1% of designs reverse by more than the CV RMSE).

| Input | Target | Expected | Share reversing | Largest reversal | CV RMSE |
|---|---|---|---|---|---|
| glazing_shgc | cooling_demand_kwh_m2 | up | 0.00% | 0.000 | 0.898 |
| glazing_shgc | heating_demand_kwh_m2 | down | 0.07% | 2.280 | 0.559 |
| glazing_u | heating_demand_kwh_m2 | up | 0.07% | 2.663 | 0.559 |
| wwr_north | zone_cooling_north_kwh_m2 | up | 5.30% | 21.648 | 5.191 |
| wwr_east | zone_cooling_east_kwh_m2 | up | 0.70% | 12.780 | 6.590 |
| wwr_south | zone_cooling_south_kwh_m2 | up | 4.75% | 20.100 | 4.664 |
| wwr_west | zone_cooling_west_kwh_m2 | up | 5.70% | 17.948 | 4.994 |

### Adjudication by EnergyPlus (H6)

**S6 after adjudication: OK** - flagged designs re-simulated at 5 points (min, 25 %, 50 %, 75 %, max); a reversal is confirmed when EnergyPlus reverses by more than the same CV RMSE; adjusted share = observed share x share not confirmed.

| Input | Target | Observed share | Confirmed by EnergyPlus | 95 % interval | Adjusted share | EnergyPlus net decrease | Median 5-point MAE |
|---|---|---|---|---|---|---|---|
| wwr_north | zone_cooling_north_kwh_m2 | 5.30% | 19/20 | 0.76-0.99 | 0.27% | 19/20 | 1.08 |
| wwr_south | zone_cooling_south_kwh_m2 | 4.75% | 17/20 | 0.64-0.95 | 0.71% | 19/20 | 0.80 |
| wwr_west | zone_cooling_west_kwh_m2 | 5.70% | 19/20 | 0.76-0.99 | 0.29% | 20/20 | 1.60 |

## Zone-sum consistency (report)

| Source | Rows | Median abs gap | 95th pct |
|---|---|---|---|
| energyplus | 3000 | 0.000% | 0.000% |
| final_models_new_designs | 2000 | 0.320% | 1.251% |
| grouped_oof | 3000 | 0.405% | 1.589% |

## SHAP in user inputs

Overall view: share of mean |SHAP| per group (climate = one block).

| Target | climate | geometry | opaque_envelope | glazing | window_area | shading | orientation |
|---|---|---|---|---|---|---|---|
| cooling_demand_kwh_m2 | 44.2% | 11.5% | 3.1% | 25.0% | 9.7% | 3.9% | 2.7% |
| eui_kwh_m2 | 68.8% | 9.2% | 2.8% | 10.2% | 6.4% | 1.6% | 1.2% |
| heating_demand_kwh_m2 | 42.2% | 16.2% | 6.5% | 25.0% | 8.8% | 0.0% | 1.4% |
| peak_cooling_w_m2 | 15.9% | 23.9% | 4.6% | 27.3% | 16.7% | 5.5% | 6.1% |
| peak_heating_w_m2 | 35.5% | 27.7% | 7.8% | 20.1% | 6.4% | 0.9% | 1.6% |
| zone_cooling_core_kwh_m2 | 65.1% | 12.0% | 12.0% | 7.7% | 1.5% | 1.0% | 0.7% |
| zone_cooling_east_kwh_m2 | 28.1% | 7.1% | 1.0% | 28.3% | 22.8% | 9.3% | 3.4% |
| zone_cooling_north_kwh_m2 | 31.8% | 6.0% | 1.0% | 22.7% | 18.3% | 6.8% | 13.3% |
| zone_cooling_south_kwh_m2 | 29.7% | 5.9% | 1.0% | 23.1% | 18.7% | 6.6% | 14.9% |
| zone_cooling_west_kwh_m2 | 38.2% | 5.2% | 1.0% | 21.7% | 15.8% | 4.5% | 13.5% |

Comparison with the 2.8 RF SHAP (21 design inputs):

| Target | Spearman | Top-5 overlap | Top 5 (3.6) | Top 5 (2.8) |
|---|---|---|---|---|
| cooling_demand_kwh_m2 | 0.89 | 4 | glazing_shgc, floor_area, wwr_south, wwr_north, floor_height | glazing_shgc, floor_area, glazing_u, wwr_south, wwr_north |
| eui_kwh_m2 | 0.93 | 5 | glazing_shgc, floor_area, floor_height, wwr_south, wwr_north | glazing_shgc, floor_area, wwr_south, floor_height, wwr_north |
| heating_demand_kwh_m2 | 0.94 | 5 | glazing_u, floor_area, glazing_shgc, n_floors, floor_height | glazing_u, floor_area, glazing_shgc, floor_height, n_floors |
| peak_cooling_w_m2 | 0.93 | 5 | glazing_shgc, floor_area, wwr_south, floor_height, wwr_north | glazing_shgc, floor_area, wwr_south, floor_height, wwr_north |
| peak_heating_w_m2 | 0.95 | 5 | n_floors, glazing_u, floor_area, glazing_shgc, roof_r_value | n_floors, glazing_u, floor_area, glazing_shgc, roof_r_value |
| zone_cooling_core_kwh_m2 | 0.83 | 5 | n_floors, roof_r_value, infiltration_ach, glazing_shgc, glazing_u | n_floors, roof_r_value, infiltration_ach, glazing_shgc, glazing_u |
| zone_cooling_east_kwh_m2 | 0.71 | 5 | glazing_shgc, wwr_east, overhang_depth_e, floor_height, glazing_u | glazing_shgc, wwr_east, overhang_depth_e, glazing_u, floor_height |
| zone_cooling_north_kwh_m2 | 0.64 | 4 | glazing_shgc, wwr_north, orientation, overhang_depth_n, floor_height | glazing_shgc, wwr_north, orientation, overhang_depth_n, glazing_u |
| zone_cooling_south_kwh_m2 | 0.61 | 4 | glazing_shgc, wwr_south, orientation, overhang_depth_s, floor_height | glazing_shgc, wwr_south, orientation, overhang_depth_s, glazing_u |
| zone_cooling_west_kwh_m2 | 0.66 | 5 | glazing_shgc, wwr_west, orientation, floor_height, glazing_u | glazing_shgc, wwr_west, orientation, glazing_u, floor_height |

Explaining one design (all 10 targets, per-climate view, 50 background rows): median 1.1 s.
