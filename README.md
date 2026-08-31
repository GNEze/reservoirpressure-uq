# Real-Time Reservoir Pressure UQ via Polynomial Chaos Expansion and Sparse Grids

Companion code for:

> Eze, G. N., Wang, Y. "A Workflow for Uncertainty Quantification of Real-Time Reservoir Pressure Estimates Via Polynomial Chaos Expansions and Sparse Grids." Submitted to *Geoenergy Science and Engineering*.

Non-intrusive PCE / Smolyak-sparse-grid uncertainty quantification for real-time average reservoir pressure (Pavg) estimates, validated against a 365-day digital twin of Volve well 15/9-F-14 H and applied to the real 3,056-day production record.

## Pipeline

Eight scripts run in sequence, each producing the inputs the next one needs:

| Script | Produces | Paper section |
|---|---|---|
| `01_volve_data_exploration.py` | Well 15/9-F-14 H reservoir parameters | §2.1, Table 3 |
| `02_digital_twin.py` | 365-day synthetic PDHMS/MPFM digital twin | §2.1 |
| `03_data_quality.py` | QC-cleaned pressure/rate streams | §2.1 (Module 1) |
| `04_state_classifier.py` | 4-state operating classification | §2.1 (Module 2) |
| `05_pavg_estimator.py` | PI and MBE deterministic Pavg estimates | Eq. 1, Eq. 2 (§2.1 Module 3) |
| `06_pce_uq_digital_twin.py` | PCE/Smolyak UQ on the digital twin | §3.5, §4.1–4.4 (Tables 2, 4–6, Fig. 1–3, 6–11) |
| `07_pce_uq_real_data.py` | PCE/Sparse-grid UQ on the real F-14 record | §4.5 (Fig. 12–13) |
| `08_bma_layer.py` | Drawdown-weighted Bayesian model averaging of PI/MBE | §4.6 (Table 7, Fig. 14) |

Run everything with:

```bash
python main.py
```

or run any script individually (each reads its inputs from `outputs/` and `data/`, so scripts 02–08 require their predecessors to have already run once).

`06_pce_uq_digital_twin.py` is the script most directly relevant to the paper's core contribution — the PCE/Smolyak methodology, the Gibbs-phenomenon post-hoc-clipping treatment, the analytical Sobol' decomposition, the convergence study, and the prior-stress-test non-circularity check are all self-contained there. `08_bma_layer.py` implements the model-form-uncertainty reconciliation layer of §4.6.

`make_figure4_workflow.py` is a supplementary, non-pipeline script that regenerates Figure 5 (the end-to-end workflow diagram) — it draws a static diagram and does not process any data, so it is not part of `main.py`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt        # Windows
# or: .venv/bin/pip install -r requirements.txt       # macOS/Linux
```

`requirements.txt` pins `numpy`, `chaospy`, and `pandas` to the exact versions the paper's reported results were generated with. Note: `chaospy`'s dependency `numpoly` ships source-only on PyPI (no prebuilt wheels) and needs a C/C++ compiler to build — on Windows, install the Microsoft Visual C++ Build Tools (C++ workload) first if `pip install` fails on `numpoly`.

## Data

Place the raw Volve production record at `data/Volve production data.xlsx`. It is Equinor's public Volve field open dataset and is not redistributed in this repository; see `data/README.md` for the source.

## Reproducibility notes

- All four uncertain inputs (ε_P, ε_q, J, c_t), the reciprocal reparameterization (η=1/J, ζ=1/c_t), the Smolyak level-3 sparse grid (165 nodes, d=4), and the post-hoc clipping treatment are implemented exactly as described in the manuscript's Methodology section.
- Random seeds are fixed throughout (`np.random.seed(42)` for the digital twin's sensor noise; `seed=7` for the sampled MPFM transport lag; `seed=42` for all Monte Carlo draws in the UQ scripts), so `06_pce_uq_digital_twin.py`, `07_pce_uq_real_data.py`, and `08_bma_layer.py` reproduce the paper's reported numbers exactly given the same input data.
- **Performance is not reproducible across machines and is not included as executable code.** The paper reports 1.04 ms/timestep (0.84 s total for the 365-day digital-twin run), measured on an Intel Core i7-1185G7 @ 3.00 GHz, 34 GB RAM, Windows 11 Pro, Python 3.13.5, NumPy 2.1.3, chaospy 4.3.21. Re-running this code on different hardware will produce a different, and equally valid, timing — the reported figure is a measurement, not a target the code enforces.
- `expected_outputs/` contains the small reference CSVs (parameter tables, UQ results, Sobol' indices, convergence study, standardized residuals, the prior-stress-test result) produced by a full run, so results can be diffed without necessarily re-running the whole pipeline.

## License

[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — see `LICENSE`. Free for any noncommercial purpose (academic, personal, nonprofit, government, and public research use are all explicitly covered); commercial use requires permission from the author.

## Citation

See `CITATION.cff` (GitHub renders this as a "Cite this repository" button on the repo page). Journal, volume, pages, and DOI will be filled in there upon acceptance/publication.
