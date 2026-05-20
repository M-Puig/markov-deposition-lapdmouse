# markov-deposition-lapdmouse

Minimum reproducible code for the manuscript:

> Evaluation of analytical particle deposition models against experimental mouse lung data

The repository keeps only the code needed to rerun the LAPDMouse deposition model and regenerate the publication figure set. Generated results and figures are intentionally excluded from git.

## Contents

```text
scripts/
  model_deposition.py          Core Markov-chain deposition model
  compare_impaction_models.py  Simulation CLI
  sensitivity_helpers.py       Flow-split, lobe, strain, and scaling helpers
  figure_helpers.py            Shared figure-generation code
data/
  morphometry_references.py
  islam_2017_generation_averages.csv
  heyder_deposition_transition.csv
notebooks/
  generate_figures.ipynb       One configurable notebook for all figure variants
```

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The full reproduction uses the external LAPDMouse dataset. Download it from the official LAPDMouse archive and arrange it as:

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

## Run Simulations

Baseline area-flow split:

```bash
python scripts/compare_impaction_models.py \
  --data_path /path/to/lapdMouseDB \
  --results_path results_area \
  --use_mouse_ventilation \
  --particle_sizes 5e-7 1e-6 2e-6 \
  --impaction_models chan_lipp zhang yeh_schum \
  --flow_split area
```

Area split with TLC-to-tidal scaling from generation 4:

```bash
python scripts/compare_impaction_models.py \
  --data_path /path/to/lapdMouseDB \
  --results_path results_area_scaled_gen4 \
  --use_mouse_ventilation \
  --particle_sizes 5e-7 1e-6 2e-6 \
  --impaction_models chan_lipp zhang yeh_schum \
  --flow_split area \
  --tlc_to_tidal_scaling
```

Volume split with TLC-to-tidal scaling from generation 4:

```bash
python scripts/compare_impaction_models.py \
  --data_path /path/to/lapdMouseDB \
  --results_path results_vsplit_scaled_gen4 \
  --use_mouse_ventilation \
  --particle_sizes 5e-7 1e-6 2e-6 \
  --impaction_models chan_lipp zhang yeh_schum \
  --flow_split volume \
  --tlc_to_tidal_scaling
```

For TLC-to-tidal scaling, `--vt_strain_table` is optional. If the file is absent, the script computes strain-mean tidal volumes from the supplied LAPDMouse metadata and writes `strain_ventilation_table.csv` into the result folder.

## Regenerate Figures

Open `notebooks/generate_figures.ipynb` and edit only the config cell:

```python
RESULTS_PATH = "../results_area"
OUTPUT_DIR = "../outputs"
OUTPUT_SUFFIX = ""
```

For variants, keep the notebook code unchanged and switch only:

```python
RESULTS_PATH = "../results_area_scaled_gen4"
OUTPUT_SUFFIX = "area_scaled_gen4"
```

or:

```python
RESULTS_PATH = "../results_vsplit_scaled_gen4"
OUTPUT_SUFFIX = "vsplit_scaled_gen4"
```

The notebook writes figures such as:

```text
comparison_hq_05m.png
comparison_hq_1m.png
comparison_hq_2m.png
dep_per_gen_05_variability.png
dep_per_lobe_1_area_scaled_gen4.png
stat_heatmap_bh_2_vsplit_scaled_gen4.png
deposition_gt.png
deposition_gt_density.png
deposition_transition.png
dep_per_gen_gt.png
dep_per_gen_gt_by_strain.png
morphometry_absolute_diameter.png
morphometry_normalized_diameter.png
stk_impaction_vs_generation.png
```

## Smoke Tests

The fixture is synthetic and only checks code health, not scientific results.

```bash
pytest
```

## License

MIT
