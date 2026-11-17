# Surrogate speed (Sub-task 3.7)

2026-11-17T07:20:48.496573+00:00 · Apple M4 · 10 cores · torch threads 1 · commit `adf12e7c4cf9bdcd66adf3ead50bcf3a3091d7d1`

Model load: 0.76 s (+ interval models 0.02 s).

| Variant | One design, median (ms) | p95 (ms) | With 90 % intervals, median (ms) | Batch 10,000 (designs/s) | Batch 100,000 (designs/s) |
|---|---|---|---|---|---|
| primary | 5.9 | 6.1 | 11.9 | 32,898 | 32,737 |
| all_mlp | 2.6 | 2.6 | 8.3 | 196,029 | 195,535 |

Single-design time per fitted object (primary, median ms): gpr_heating 0.2, xgb_eui 3.3, mlp_cg 0.2, mlp_physics 0.7; feature engineering alone 1.5 ms.

Batch of 10,000 designs, seconds per fitted object (features ready): gpr_heating 0.21, xgb_eui 0.04, mlp_cg 0.02, mlp_physics 0.02.

Speed-up vs EnergyPlus: 2,819× for one design (16.7 s per run); 75,294× in batch (2.3 s per run in the parallel campaign).

**S8 PASS** (one design < 50 ms and batch ≥ 10,000 designs/s, primary picks).
