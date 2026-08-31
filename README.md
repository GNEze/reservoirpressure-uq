# Real-Time Reservoir Pressure UQ via Polynomial Chaos Expansion on Sparse Grids

Companion code for Eze, G. N., Wang, Y., "Efficient Uncertainty Quantification of Real-Time Reservoir Pressure Estimates via Polynomial Chaos Expansion on Sparse Grids." 

## Pipeline

Eight scripts, run in order via `python main.py`, or individually (each reads its inputs from `outputs/` and `data/`, so 02-08 need their predecessors to have run first).

| Script | Produces |
|---|---|
| `01_volve_data_exploration.py` | Well 15/9-F-14 H reservoir parameters |
| `02_digital_twin.py` | 365-day synthetic PDHMS/MPFM digital twin |
| `03_data_quality.py` | QC-cleaned pressure/rate streams |
| `04_state_classifier.py` | 4-state operating classification |
| `05_pavg_estimator.py` | PI and MBE deterministic Pavg estimates |
| `06_pce_uq_digital_twin.py` | PCE/Smolyak UQ on the digital twin (Fig. 1-4, 6-12) |
| `07_pce_uq_real_data.py` | PCE/Sparse-grid UQ on the real F-14 record (Fig. 13-14) |
| `08_bma_layer.py` | Drawdown-weighted Bayesian model averaging of PI/MBE (Fig. 15-16) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# or: .venv/bin/pip install -r requirements.txt       # macOS/Linux
```

`numpoly` (a chaospy dependency) builds from source. On Windows, install the MSVC Build Tools (C++ workload) first if `pip install` fails on it.

## Data

Place the raw Volve production record at `data/Volve production data.xlsx`. Not redistributed here. See `data/README.md` for the source.

## Reproducibility

Seeds are fixed (`seed=42` for sensor noise and Monte Carlo draws, `seed=7` for the sampled MPFM transport lag), so `06`-`08` reproduce the paper's exact results given the same input data. Performance is hardware-dependent and not part of the code: the paper reports a median of 1.09 ms/timestep (0.40 s for the 365-day digital-twin run) on an Intel Core Ultra 9 285H. `expected_outputs/` holds reference CSVs for diffing without a full re-run.

## License

PolyForm Noncommercial License 1.0.0. See `LICENSE`.

## Citation

See `CITATION.cff`.
