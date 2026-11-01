# One-at-a-time parameter response test — PASS

Baseline = every parameter at mid-range (n_floors 3); each row changes ONE parameter min → max (orientation 0 → 90°). Values: (max − min) / baseline, %.

- Parameters that respond (> 0.1% on heating, cooling, EUI or a zone cooling): **21/21**
- Expected directions confirmed: **19/19**

| parameter | heat Minneapolis | cool Colombo | cool Minneapolis | EUI Colombo | EUI Minneapolis | max | expected direction |
|---|---|---|---|---|---|---|---|
| floor_area | -99.7 | -34.2 | -6.9 | -16.1 | -23.1 | 99.7 | — (responds only) |
| aspect_ratio | +13.0 | +7.3 | +2.7 | +3.5 | +4.3 | 13.0 | — (responds only) |
| n_floors | -34.4 | -9.3 | +13.5 | -4.8 | -6.7 | 34.4 | — (responds only) |
| floor_height | +43.1 | +13.4 | +6.0 | +6.8 | +9.0 | 43.1 | — (responds only) |
| wall_r_value | -34.4 | -3.7 | +2.2 | -2.0 | -4.4 | 34.4 | ✅ heat minn ↓ (-34.4%) |
| roof_r_value | -35.7 | -9.3 | +11.0 | -5.6 | -8.1 | 35.7 | ✅ heat minn ↓ (-35.7%) |
| floor_r_value | -4.0 | -0.0 | +0.8 | -0.0 | -0.4 | 4.0 | ✅ heat minn ↓ (-4.0%); ✅ cool colo ↓ (-0.0%) |
| wall_thermal_mass | -1.0 | -0.0 | -0.4 | +0.1 | -0.4 | 2.1 | — (responds only) |
| infiltration_ach | +14.8 | +0.9 | -10.5 | +1.3 | +4.3 | 14.8 | ✅ heat minn ↑ (+14.8%) |
| wwr_north | +37.0 | +16.0 | +7.5 | +7.3 | +9.1 | 91.5 | ✅ zone_cool_sens_north colo ↑ (+91.5%); ✅ zone_heat_sens_north minn ↑ (+127.4%) |
| wwr_east | +8.4 | +5.3 | +5.1 | +2.5 | +3.3 | 94.2 | ✅ zone_cool_sens_east colo ↑ (+78.7%) |
| wwr_south | +34.8 | +15.0 | +9.4 | +7.0 | +9.6 | 89.5 | ✅ zone_cool_sens_south colo ↑ (+89.5%) |
| wwr_west | +25.4 | +3.9 | -0.1 | +1.9 | +3.9 | 65.4 | ✅ zone_cool_sens_west colo ↑ (+65.4%) |
| glazing_u | +134.9 | +3.7 | -25.6 | +1.2 | +14.6 | 134.9 | ✅ heat minn ↑ (+134.9%) |
| glazing_shgc | -62.4 | +37.2 | +43.0 | +17.6 | +6.2 | 167.8 | ✅ cool colo ↑ (+37.2%); ✅ cool minn ↑ (+43.0%); ✅ heat minn ↓ (-62.4%) |
| overhang_depth_n | +3.6 | -4.7 | -4.0 | -2.1 | -1.0 | 39.0 | ✅ zone_cool_sens_north colo ↓ (-26.9%) |
| overhang_depth_e | +2.8 | -2.1 | -3.9 | -1.0 | -0.8 | 71.4 | ✅ zone_cool_sens_east colo ↓ (-31.6%) |
| overhang_depth_s | +3.5 | -4.3 | -4.2 | -2.0 | -1.1 | 36.5 | ✅ zone_cool_sens_south colo ↓ (-25.9%) |
| overhang_depth_w | +0.2 | -1.0 | -0.2 | -0.5 | -0.1 | 16.1 | ✅ zone_cool_sens_west colo ↓ (-16.1%) |
| sidefin_depth | +0.5 | -0.2 | -0.3 | -0.1 | -0.0 | 2.2 | ✅ cool colo ↓ (-0.2%) |
| orientation | +2.2 | +4.2 | +2.7 | +2.1 | +2.1 | 120.7 | — (responds only) |
