# markov-deposition-lapdmouse

Minimum reproducible code for the manuscript:

> **Evaluation of analytical particle deposition models against experimental
> mouse lung data**

This repository contains only the particle-deposition pipeline and the code
needed to regenerate the paper's **main results** from the LAPDMouse dataset.
Exploratory analyses, supplementary sensitivity variants, and generated
outputs are intentionally left out. Simulation results and figures are not
committed to git.

## What it reproduces

Running the pipeline below regenerates every code-produced main-text figure and
the numeric results behind the paper's tables. Figure filenames match the
manuscript:

| Output file | Paper figure |
|---|---|
| `deposition_gt.png`, `deposition_gt_density.png` | Fig. `ex_gt` — experimental captured-particle maps |
| `deposition_transition.png` | Fig. `heyder-total-dep` — Heyder regional-deposition transition |
| `dep_per_gen_gt.png` | Fig. `dep_per_gen_gt` — experimental deposition per generation |
| `comparison_hq_{05,1,2}m.png` | Fig. `compare_on_data` — per-mechanism deposition vs generation |
| `dep_per_gen_{05,1,2}_variability.png` | Fig. `dep_per_gen` — model vs experiment, per generation, with inter-subject bands |
| `dep_per_lobe_{05,1,2}.png` | Fig. `dep_per_lobe` — lobe-wise deposition fractions |
| `heatmap_bh_{1,2}.png` | Fig. `heatmap` — lobe-wise signed error, Wilcoxon + Benjamini-Hochberg |
| `stk_impaction_vs_generation.png` | Fig. `stk_impaction` — Stokes number vs impaction probability |
| `morphometry_{absolute,normalized}_diameter.png` | Fig. `morphometry` — airway diameter vs Islam/Weibel references |
| `dep_per_gen_gt_by_strain.png` | Fig. S1 — strain-stratified experimental curves |

Numeric results (see `results_summary*` below):

- `results_summary_total_capture.csv` — total captured fraction (%) per particle
  size × impaction kernel;
- `results_summary_penetration_t12.csv` — distal penetration `T12 = P(generation ≥ 12 | captured)`;
- `results_summary_lobe_stats.csv` — lobe-wise signed errors with Wilcoxon
  signed-rank + Benjamini-Hochberg significance (the values drawn on the heatmaps);
- `results_summary.txt` — human-readable digest of all three.

**Not reproduced by code:** the hand-drawn mechanism schematics
(impaction / sedimentation / diffusion) and the 3-D airway-tree renderings are
authored assets, not pipeline outputs.

## Contents

```text
scripts/
  model_deposition.py          Core Markov-chain deposition model
  compare_impaction_models.py  Simulation CLI (flow split, ventilation, TLC scaling)
  sensitivity_helpers.py       Flow-split, lobe, strain, and geometry-scaling helpers
  figure_helpers.py            Figure + numeric-results generation
data/
  morphometry_references.py    Islam (2017) and Weibel (1963) reference airway sizes
  islam_2017_generation_averages.csv
  heyder_deposition_transition.csv
notebooks/
  generate_figures.ipynb       Runs the whole figure/results set from a result folder
tests/
  test_minimum_repro.py        Smoke test on a synthetic fixture
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The reproduction uses the external LAPDMouse dataset. Download it from the
official LAPDMouse archive and arrange one folder per mouse:

```text
/path/to/lapdMouseDB/
  m01/
    m01_Info.md
    m01_AirwayTreeTable.csv
    m01_AirwaySegmentsDeposition.csv
    m01_Ventilation_Pre.csv
    ...
  m02/
  ...
```

## Reproduce the main results

The main-text figures use the **piecewise (hybrid) flow split with TLC-to-tidal
geometry scaling**: generations 1–3 are held at the imaged (TLC) dimensions and
generation 4 onward is scaled to the tidal breathing state.

### 1. Run the deposition pipeline

```bash
python scripts/compare_impaction_models.py \
  --data_path /path/to/lapdMouseDB \
  --results_path results_main \
  --use_mouse_ventilation \
  --particle_sizes 5e-7 1e-6 2e-6 \
  --impaction_models chan_lipp zhang yeh_schum \
  --flow_split hybrid \
  --hybrid_compliant_generation 4 \
  --tlc_to_tidal_scaling
```

The strain-mean tidal volumes needed for TLC-to-tidal scaling are computed
directly from the LAPDMouse metadata and written to
`results_main/strain_ventilation_table.csv`; no external table is required.

### 2. Generate figures and numeric results

Open `notebooks/generate_figures.ipynb` and run it (the config cell already
points at `results_main` and writes to `outputs/`):

```python
RESULTS_PATH = REPO_ROOT / 'results_main'
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_SUFFIX = ''
```

Or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/generate_figures.ipynb
```

All figures and `results_summary*` files are written to `outputs/`.

## Smoke test

The fixture is synthetic and only checks code health, not scientific results:

```bash
pytest
```

## License

MIT
