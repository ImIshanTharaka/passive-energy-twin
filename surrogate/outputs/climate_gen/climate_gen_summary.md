# Sub-task 3.4 — summary (thesis tables)

All errors in original units (kWh/m²) after the inverse transform; LOCO = leave-one-climate-out.

## Before (3.3 provisional) vs after (3.4 final)

| Target | 3.3 pick | LOCO MAPE % | median \|bias\| % | 3.4 pick | grouped skill | LOCO MAPE % | median \|bias\| % | max \|bias\| % |
|---|---|---|---|---|---|---|---|---|
| Heating | Neural network (MLP) C-G direct | 25.7 | 11.2 | Gaussian process C-G+I log1p | 0.988 | 10.0 | 6.2 | 15.9 (Atlanta) |
| Cooling | Neural network (MLP) C-G direct | 3.3 | 2.7 | Neural network (MLP) C-G direct | 0.996 | 3.3 | 2.7 | 6.9 (Phoenix) |
| EUI | Gaussian process C-G direct | 5.6 | 3.6 | XGBoost C-G+I decomposed | 0.935 | 4.8 | 3.9 | 12.8 (Phoenix) |

## LOCO MAPE % by family and form (C-G+I)

| Family | Heating direct | Heating log1p | Heating ratio | Cooling direct | EUI direct | EUI decomposed |
|---|---|---|---|---|---|---|
| Random Forest | 39.3 (0.947) | 35.0 (0.943) | 18.2 (0.955) | 5.1 (0.915) | 11.9 (0.919) | 5.8 (0.886) |
| XGBoost | 27.1 (0.980) | 25.5 (0.983) | 14.4 (0.976) | 4.4 (0.971) | 12.0 (0.970) | 4.8 (0.935) |
| LightGBM | 31.2 (0.976) | 33.6 (0.973) | 21.9 (0.871) | 3.8 (0.967) | 13.6 (0.969) | 5.8 (0.923) |
| Gaussian process | 27.1 (0.958) | 10.0 (0.988) | 14.7 (0.992) | 13.5 (0.956) | 6.2 (0.953) | 7.4 (0.924) |
| Neural network (MLP) | 22.5 (0.997) | 33.3 (0.994) | 13.3 (0.996) | 3.8 (0.995) | 6.7 (0.990) | 5.9 (0.959) |

Cells: LOCO MAPE % (grouped design skill).

## Sensitivity (sensitivity_3_4, reported only): LightGBM ratio forms with scale matching

| Target | Form | grouped skill (unscaled → scaled) | LOCO MAPE % (unscaled → scaled) | median \|bias\| % |
|---|---|---|---|---|
| Heating | ratio_scaled | 0.871 → 0.983 | 21.9 → 14.7 | 8.6 |
| Peak heating | ratio_scaled | 0.946 → 0.941 | 21.6 → 21.9 | 22.0 |
| EUI | decomposed_scaled | 0.923 → 0.932 | 5.8 → 5.8 | 3.9 |

Figures: fig_before_after, fig_bias_per_climate, fig_error_vs_distance, fig_ranking_unseen (PNG + PDF).