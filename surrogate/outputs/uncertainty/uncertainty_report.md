# Sub-task 3.5 — uncertainty and applicability domain

All widths in original units (kWh/m² or W/m²). Coverage = share of rows whose EnergyPlus value lies inside the interval; target 0.90. Heating targets scored on the 7 heating climates.

## 1. Deployed method and 90 % coverage

| Target | Method | Standard, grouped CV | New-site, LOCO | Standard used in LOCO | Mean width std / new-site |
|---|---|---|---|---|---|
| Heating | lgbm | 0.893 | 0.886 | 0.673 | 1.56 / 3.92 |
| Cooling | lgbm | 0.900 | 0.914 | 0.360 | 2.92 / 10.58 |
| EUI | lgbm | 0.900 | 0.891 | 0.481 | 11.94 / 56.91 |
| Peak heat | lgbm | 0.904 | 0.935 | 0.235 | 1.77 / 10.09 |
| Peak cool | lgbm | 0.908 | 0.888 | 0.414 | 2.57 / 11.89 |
| Core | lgbm | 0.901 | 0.844 | 0.286 | 0.91 / 12.54 |
| South | lgbm | 0.906 | 0.911 | 0.448 | 15.67 / 61.76 |
| East | lgbm | 0.911 | 0.888 | 0.460 | 21.96 / 68.50 |
| North | lgbm | 0.907 | 0.894 | 0.296 | 16.99 / 74.58 |
| West | lgbm | 0.903 | 0.872 | 0.289 | 14.68 / 71.17 |

## 2. All methods (90 %)

| Target | Mode | Method | Coverage | Mean width | Rel. width |
|---|---|---|---|---|---|
| Heating | standard_grouped | lgbm | 0.893 | 1.56 | 0.206 |
| Heating | standard_grouped | none | 0.902 | 1.61 | 0.213 |
| Heating | standard_grouped | given | 0.900 | 1.74 | 0.230 |
| Heating | standard_grouped | native | 0.838 | 1.41 | 0.186 |
| Heating | new_site_loco | lgbm | 0.886 | 3.92 | 0.518 |
| Heating | new_site_loco | none | 0.888 | 2.67 | 0.353 |
| Heating | new_site_loco | given | 0.874 | 3.29 | 0.435 |
| Heating | new_site_loco | native | 0.773 | 2.19 | 0.289 |
| Heating | standard_in_loco | lgbm | 0.673 | 1.67 | 0.220 |
| Heating | standard_in_loco | none | 0.738 | 1.60 | 0.211 |
| Heating | standard_in_loco | given | 0.843 | 2.72 | 0.359 |
| Cooling | standard_grouped | lgbm | 0.900 | 2.92 | 0.041 |
| Cooling | standard_grouped | none | 0.899 | 2.77 | 0.039 |
| Cooling | new_site_loco | lgbm | 0.914 | 10.58 | 0.148 |
| Cooling | new_site_loco | none | 0.892 | 10.52 | 0.148 |
| Cooling | standard_in_loco | lgbm | 0.360 | 2.88 | 0.040 |
| Cooling | standard_in_loco | none | 0.410 | 2.77 | 0.039 |
| EUI | standard_grouped | lgbm | 0.900 | 11.94 | 0.083 |
| EUI | standard_grouped | none | 0.899 | 11.26 | 0.078 |
| EUI | new_site_loco | lgbm | 0.891 | 56.91 | 0.394 |
| EUI | new_site_loco | none | 0.887 | 30.50 | 0.211 |
| EUI | standard_in_loco | lgbm | 0.481 | 12.12 | 0.084 |
| EUI | standard_in_loco | none | 0.550 | 11.25 | 0.078 |
| Peak heat | standard_grouped | lgbm | 0.904 | 1.77 | 0.172 |
| Peak heat | standard_grouped | none | 0.899 | 1.58 | 0.153 |
| Peak heat | new_site_loco | lgbm | 0.935 | 10.09 | 0.980 |
| Peak heat | new_site_loco | none | 0.868 | 7.24 | 0.703 |
| Peak heat | standard_in_loco | lgbm | 0.235 | 1.96 | 0.190 |
| Peak heat | standard_in_loco | none | 0.186 | 1.58 | 0.153 |
| Peak cool | standard_grouped | lgbm | 0.908 | 2.57 | 0.097 |
| Peak cool | standard_grouped | none | 0.899 | 2.44 | 0.093 |
| Peak cool | new_site_loco | lgbm | 0.888 | 11.89 | 0.451 |
| Peak cool | new_site_loco | none | 0.892 | 7.27 | 0.276 |
| Peak cool | standard_in_loco | lgbm | 0.414 | 2.49 | 0.095 |
| Peak cool | standard_in_loco | none | 0.460 | 2.44 | 0.093 |
| Core | standard_grouped | lgbm | 0.901 | 0.91 | 0.016 |
| Core | standard_grouped | none | 0.900 | 0.83 | 0.015 |
| Core | new_site_loco | lgbm | 0.844 | 12.54 | 0.222 |
| Core | new_site_loco | none | 0.880 | 3.85 | 0.068 |
| Core | standard_in_loco | lgbm | 0.286 | 0.93 | 0.016 |
| Core | standard_in_loco | none | 0.252 | 0.83 | 0.015 |
| South | standard_grouped | lgbm | 0.906 | 15.67 | 0.140 |
| South | standard_grouped | none | 0.901 | 14.31 | 0.128 |
| South | new_site_loco | lgbm | 0.911 | 61.76 | 0.552 |
| South | new_site_loco | none | 0.890 | 49.96 | 0.446 |
| South | standard_in_loco | lgbm | 0.448 | 16.21 | 0.145 |
| South | standard_in_loco | none | 0.475 | 14.26 | 0.127 |
| East | standard_grouped | lgbm | 0.911 | 21.96 | 0.165 |
| East | standard_grouped | none | 0.900 | 20.65 | 0.156 |
| East | new_site_loco | lgbm | 0.888 | 68.50 | 0.516 |
| East | new_site_loco | none | 0.881 | 61.00 | 0.459 |
| East | standard_in_loco | lgbm | 0.460 | 21.55 | 0.162 |
| East | standard_in_loco | none | 0.518 | 20.57 | 0.155 |
| North | standard_grouped | lgbm | 0.907 | 16.99 | 0.152 |
| North | standard_grouped | none | 0.899 | 15.97 | 0.143 |
| North | new_site_loco | lgbm | 0.894 | 74.58 | 0.669 |
| North | new_site_loco | none | 0.896 | 59.66 | 0.535 |
| North | standard_in_loco | lgbm | 0.296 | 16.79 | 0.151 |
| North | standard_in_loco | none | 0.332 | 15.96 | 0.143 |
| West | standard_grouped | lgbm | 0.903 | 14.68 | 0.152 |
| West | standard_grouped | none | 0.900 | 14.59 | 0.151 |
| West | new_site_loco | lgbm | 0.872 | 71.17 | 0.736 |
| West | new_site_loco | none | 0.895 | 55.09 | 0.570 |
| West | standard_in_loco | lgbm | 0.289 | 14.56 | 0.151 |
| West | standard_in_loco | none | 0.357 | 14.56 | 0.151 |

## 3. Selection rules

- Heating: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 6, 'given': 3}); G5 → conformal
- Cooling: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 1})
- EUI: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 2, 'none': 5})
- Peak heat: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 0})
- Peak cool: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 1})
- Core: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 4})
- South: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 0})
- East: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 1})
- North: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 1})
- West: G2 → lgbm (climates outside 0.85–0.95: {'lgbm': 0, 'none': 4})

## 4. Applicability domain — each training climate as if unseen

| Climate | Distance to nearest | Nearest | Out of range (vs other 9) | Level |
|---|---|---|---|---|
| atlanta | 1.79 | san_francisco | — | new |
| chennai | 1.49 | miami | cdd10,mean_dbt,occ_cool_dh | extrapolation |
| chicago | 0.82 | minneapolis | — | new |
| colombo | 1.59 | chennai | mean_rh,latitude,mean_wind,occ_enth_excess,occ_latent_excess,occ_band_dh,occ_heat_dh | extrapolation |
| denver | 2.14 | san_francisco | occ_latent_excess | extrapolation |
| miami | 1.49 | chennai | — | new |
| minneapolis | 0.82 | chicago | hdd18,annual_ghi,mean_dbt,latitude,occ_heat_dh,hdd15,hdd12 | extrapolation |
| new_york | 1.86 | atlanta | — | new |
| phoenix | 3.56 | miami | annual_ghi,mean_rh | extrapolation |
| san_francisco | 1.79 | atlanta | cdd10,mean_wind,occ_enth_excess,occ_band_dh,occ_cool_dh | extrapolation |
