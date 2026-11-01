# Pilot validation report — `pilot`

Generated 2026-10-12 06:43 by `runner/pilot_report.py` from `/sessions/rcw-014f9bjnjw1dkt4u68d9uprk/mnt/simulation/sampling/samples_pilot.csv` (200 rows).

| check | status |
|---|---|
| R. Reliability | **PASS** |
| P. Physics by climate | **PASS** |
| D. Direction of effects | **PASS** |
| E. Edge cases | **PASS** |
| V. VT-only pair (Decision S2) | **PASS** |
| K. Peak timing (Decision T4) | **PASS** |
| M. Meter vs zone sensible; unmet hours; latent share | **INFO** |
| G. Ground coupling (ground-temperature diagnostic) | **PASS** |
| T. Runtime and disk | **INFO** |

## R. Reliability — PASS

- Runs: **200/200 ok**, failed 0, build errors 0; parser problems: 0
- Fatal: 0; runs with severe errors: **0**
- Clean runs (completed, no severe, only benign warnings): 200/200

## P. Physics by climate — PASS

| check | share of designs | required |
|---|---|---|
| Minneapolis has the highest heating | 100% | 100% |
| San Francisco has the lowest EUI | 100% | 100% |
| Colombo heating < 2 kWh/m² | 100% | 100% |
| Colombo cooling ≥ Phoenix (per design) | 74% | info |
| EUI within 50–300 kWh/m² | 100% | info |

Colombo heating: median 0.48, max 0.62 kWh/m². Median cooling Colombo 88.8 vs Phoenix 84.1 kWh/m².

Per-climate distribution (min / median / max):

| target | colombo | phoenix | san_francisco | minneapolis |
|---|---|---|---|---|
| heating | 0.3 / 0.5 / 0.6 | 0.6 / 1.0 / 4.2 | 0.2 / 1.3 / 8.5 | 3.5 / 15.5 / 52.5 |
| cooling | 68.6 / 88.8 / 196.4 | 62.8 / 84.1 / 215.9 | 37.5 / 58.9 / 138.8 | 30.7 / 54.6 / 113.6 |
| EUI | 201.5 / 227.8 / 349.1 | 120.2 / 143.3 / 274.3 | 66.7 / 75.8 / 156.4 | 104.7 / 124.2 / 199.7 |
| peak heat | 0.2 / 0.5 / 0.8 | 2.1 / 4.5 / 13.0 | 3.1 / 5.9 / 14.2 | 6.3 / 14.5 / 35.0 |
| peak cool | 20.6 / 26.0 / 61.7 | 22.0 / 29.9 / 70.0 | 16.2 / 21.7 / 63.1 | 17.6 / 23.7 / 62.3 |

EUI outside band: pilot_P24_colombo (349)

## D. Direction of effects — PASS

Spearman ρ on the LHS designs (n = 42 per climate). Bold = required strong effect (expected sign and |ρ| ≥ 0.3).

| target | input | expected | colombo | phoenix | san_francisco | minneapolis |
|---|---|---|---|---|---|---|
| heating | glazing_u | + | -0.04 | **+0.67** | **+0.69** | **+0.68** |
| cooling | shgc_x_wwr | + | **+0.64** | **+0.81** | **+0.82** | **+0.79** |
| peak cool | shgc_x_wwr | + | **+0.65** | **+0.54** | **+0.77** | **+0.70** |
| peak heat | glazing_u | + | -0.06 | **+0.32** | **+0.39** | **+0.51** |
| heating | wall_r_value | − | +0.18 | -0.11 | -0.15 | -0.16 |
| heating | roof_r_value | − | +0.35 | -0.07 | -0.11 | -0.13 |
| heating | infiltration_ach | + | -0.21 | +0.03 | +0.08 | +0.10 |
| cooling | mean_overhang | − | -0.04 | -0.05 | -0.11 | -0.09 |

Rows without bold entries are weak effects that a 42-design screen cannot resolve (SE of ρ ≈ 0.16); they need a one-at-a-time response test.

## E. Edge cases — PASS

Edge-case runs: 32; not clean: 0

EUI (kWh/m²):

| design | colombo | phoenix | san_francisco | minneapolis |
|---|---|---|---|---|
| E01 | 252.7 | 144.2 | 72.7 | 152.5 |
| E02 | 252.4 | 176.5 | 93.5 | 151.4 |
| E03 | 276.9 | 188.8 | 95.3 | 158.0 |
| E04 | 214.6 | 132.6 | 73.4 | 111.3 |
| E05 | 235.8 | 144.1 | 74.1 | 126.9 |
| E06 | 228.7 | 145.4 | 77.1 | 125.0 |
| E07 | 207.7 | 123.7 | 68.8 | 105.7 |
| E08 | 227.0 | 142.1 | 76.3 | 120.0 |

## V. VT-only pair (Decision S2) — PASS

- vtpair_P01vtA_colombo vs vtpair_P01vtB_colombo: 0 of 34 numeric columns differ (rel. > 1e-6)
- vtpair_P01vtA_minneapolis vs vtpair_P01vtB_minneapolis: 0 of 34 numeric columns differ (rel. > 1e-6)

## K. Peak timing (Decision T4) — PASS

Hour-of-day (hour ending) of the steady-state peaks. Excluded transition hours: 07, 08, 23 (Mon–Fri); 07, 08, 19 (Sat). A cluster at hour 09 would mean recovery lasts > 2 h.

| peak | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heat |  |  |  |  |  |  |  |  | 2 | 3 |  |  |  |  |  |  |  | 2 |  | 50 | 107 | 36 |  |  |

Heating steady peaks at hour 09: 1% of runs; ratio steady/hourly peak there median 0.18.
| cool |  |  |  |  |  |  |  |  | 1 | 37 | 3 | 5 |  | 8 | 12 | 19 | 115 |  |  |  |  |  |  |  |

## M. Meter vs zone sensible; unmet hours; latent share — INFO

Zone sensible minus EnergyTransfer meter, % of zone sensible (annual):

| climate | heating mean | heating max | cooling mean | cooling max |
|---|---|---|---|---|
| colombo | 7.4% | 8.7% | 0.32% | 0.45% |
| phoenix | 2.9% | 4.0% | 0.17% | 0.24% |
| san_francisco | 2.5% | 4.3% | 0.04% | 0.10% |
| minneapolis | 1.7% | 5.0% | 0.10% | 0.17% |

Max unmet hours: heating 0.00, cooling 0.00.
Latent share of cooling (median): colombo 18%, phoenix 9%, san_francisco 11%, minneapolis 11%

## G. Ground coupling (ground-temperature diagnostic) — PASS

Colombo heating vs floor insulation: Spearman ρ = **+0.08**; max Colombo heating 0.62 kWh/m².
Colombo's ground is ~27–29 °C; heating that falls with floor insulation there means the slab is losing heat to a ground colder than the zone - i.e. the ground temperature is wrong for this climate (FAIL when ρ < −0.5 and heating > 2 kWh/m²).

## T. Runtime and disk — INFO

- EnergyPlus time per run: median 29.4 s, p90 51.8 s, max 65.8 s
- IDF build time per run: median 0.7 s, max 2.2 s
- Disk: 253 MB for 200 runs = 1.26 MB/run
- Throughput: 4.0 s/run on 8 workers → 3400 runs ≈ **3.8 h**, ≈ 4.3 GB

## Figures

![fig1_targets_by_climate.png](report_figures/fig1_targets_by_climate.png)
![fig2_spearman_heatmap.png](report_figures/fig2_spearman_heatmap.png)
![fig3_peak_hours.png](report_figures/fig3_peak_hours.png)
![fig4_heating_vs_floor_insulation.png](report_figures/fig4_heating_vs_floor_insulation.png)
