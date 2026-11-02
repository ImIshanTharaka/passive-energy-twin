# S0 reproduction check (Sub-task 3.1 step 6)

**PASS** - untuned 2.8 RF re-run through the Phase 3 harness; design skill must match 2.8 within ± 0.002.

Run 2026-11-02T14:53:54 UTC, 24 s, dataset sha256 `b826d61b541f…`, scikit-learn 1.9.1.

| target | role | 2.8 skill | new skill | difference | result | pooled R² | Spearman (median) | top-10 % hit |
|---|---|---|---|---|---|---|---|---|
| heating_demand_kwh_m2 | primary | 0.704321 | 0.704321 | -0.000000 | PASS | 0.891 | 0.900 | 0.690 |
| cooling_demand_kwh_m2 | primary | 0.748890 | 0.748890 | +0.000000 | PASS | 0.886 | 0.919 | 0.727 |
| eui_kwh_m2 | primary | 0.679843 | 0.679843 | -0.000000 | PASS | 0.977 | 0.886 | 0.680 |
| peak_heating_w_m2 | secondary | 0.764403 | 0.764403 | -0.000000 | PASS | 0.930 | 0.920 | 0.729 |
| peak_cooling_w_m2 | secondary | 0.744728 | 0.744728 | -0.000000 | PASS | 0.776 | 0.902 | 0.690 |
| zone_cooling_core_kwh_m2 | secondary | 0.868029 | 0.868029 | -0.000000 | PASS | 0.965 | 0.946 | 0.783 |
| zone_cooling_south_kwh_m2 | secondary | 0.830055 | 0.830055 | -0.000000 | PASS | 0.881 | 0.955 | 0.743 |
| zone_cooling_east_kwh_m2 | secondary | 0.852220 | 0.852220 | +0.000000 | PASS | 0.888 | 0.959 | 0.723 |
| zone_cooling_north_kwh_m2 | secondary | 0.825538 | 0.825538 | -0.000000 | PASS | 0.886 | 0.953 | 0.723 |
| zone_cooling_west_kwh_m2 | secondary | 0.766184 | 0.766184 | -0.000000 | PASS | 0.876 | 0.928 | 0.697 |

Spearman and top-10 % hit rate are new Phase 3 metrics (within-climate ranking; heating targets over the 7 heating climates) - recorded here as the untuned baseline.
