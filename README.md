# Web-Based Digital Twin Framework for Passive Building Energy Optimisation

MSc Computer Science research, University of Moratuwa — M.G.I. Tharaka (Ishan), supervised by Prof. Indika Perera.

One folder per framework component (Phase 3 plan, decision P-L1). Dependencies point one way only:
`simulation` <- `surrogate` <- `optimiser` <- `web` (P-L3).

| Folder | Phase | Status |
|---|---|---|
| `simulation/` | 2 — EnergyPlus data generation | frozen (own venv kept for Phase 2 commands) |
| `surrogate/` | 3 — ML surrogate models | in progress |
| `optimiser/` | 4 — surrogate-assisted optimisation | placeholder |
| `web/` | 5 — web tool (FastAPI, React, Docker) | placeholder |

## Shared development environment (P-L4)

`venv/` is one Python 3.13 environment for Phase 3 onwards. Create it once on the Mac:

```bash
cd ~/Documents/Msc/Research/Project
bash setup_venv.sh
```

The script installs the exact Phase 2 package versions and adds two path files to the venv,
so `from dataset.assemble import load_dataset`, `from eda import eda_lib` and
`import surrogate` work from any folder without changing anything inside `simulation/`.
Always run Python as `./venv/bin/python` (not a bare `python`).
