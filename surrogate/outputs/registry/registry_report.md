# Deployed surrogate (registry)

Built 2026-11-20T08:00:57.577514+00:00 · config MD5 a5585a58e4414927e27429e409ebb695 · git 8981fc98e221839a02a170d9a5d0f38a6fc28b9d · variant **primary**

## What serves each output

| Output | Object | Model / features / form | File | SHA-256 |
|---|---|---|---|---|
| heating_demand_kwh_m2 | gpr_heating | gpr / C-G+I / log1p | `models/gpr_heating_slim.pkl` (0.42 MB) | fef62058a24f… |
| cooling_demand_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| eui_kwh_m2 | xgb_eui | xgb / C-G+I / decomposed | `models/xgb_eui.pkl` (5.63 MB) | 04df009f0bab… |
| peak_heating_w_m2 | mlp_physics | mlp / C-G+I / direct | `models/mlp_physics.pkl` (2.31 MB) | f884cadc9668… |
| peak_cooling_w_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| zone_cooling_core_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| zone_cooling_south_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| zone_cooling_east_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| zone_cooling_north_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |
| zone_cooling_west_kwh_m2 | mlp_cg | mlp / C-G / direct | `models/mlp_cg.pkl` (2.29 MB) | c31e0ef89e55… |

Intervals: `models/intervals.pkl` (9.5 MB, 460ad9899ac8…) — scaled split-conformal (3.5 G2), both types; quantiles = 3.5 selection.json (max |Δq| 0.0e+00).

Slim heating GPR: 4,000 rows compared with the full 3.6 file, max relative difference 0.0e+00, bit-identical True. Full file kept as archive (a6dfcf2dab6c…).

Climate levels: training / new (distance ≤ 3.559, all values in range) / extrapolation. Design check strict by default.

## Acceptance

Load 0.51 s (all hashes verified) · self-checks **PASS** · speed **PASS**

| Path | One design (median) | p95 | Batch | Designs/s |
|---|---|---|---|---|
| values only | 6.7 ms | 7.1 ms | 10,000 (10 calls) | 25,985 |
| values only | 6.7 ms | 7.1 ms | 100,000 (10 calls) | 31,507 |
| with ranges + flags | 14.1 ms | 14.7 ms | 10,000 (10 calls) | 6,954 |
| values only, just after one call with ranges (report only) | 6.9 ms | 7.1 ms | - | - |

Limits (same as the 3.7 speed check): one design < 50 ms; 100,000 designs ≥ 10,000 designs/s.

3.7 path for comparison: 5.9 ms, 32,737 designs/s (mixed climates in one call).
