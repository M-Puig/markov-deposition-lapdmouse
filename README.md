# markov-deposition-lapdmouse

Code accompanying the article:

> **Evaluation of analytical particle deposition models against experimental mouse lung data**  
> Martin Puig, Nicolas Molinari, Eric Matzner-Løber

## Overview

This repository implements a probabilistic Markov chain model for particle deposition in mouse airway trees and evaluates the sensitivity of its predictions to the choice of impaction formula (Chan-Lippmann, Yeh-Schum, Zhang variants) against the [LAPDMouse dataset](https://lapdmouse.iibi.uiowa.edu/), which provides spatially resolved, per-airway particle deposition counts across 34 mice for 0.5, 1, and 2 µm particles.

## Repository structure

```
markov-deposition-paper/
├── scripts/
│   ├── model_deposition.py          # Core Markov chain deposition model
│   └── compare_impaction_models.py  # Compare impaction model variants
├── data/
│   └── morphometry_references.py    # Reference morphometric data (Weibel, Islam et al.)
├── notebooks/
│   └── tutorial.ipynb               # Step-by-step tutorial
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `tqdm`, `jupyter`.

## Data

The scripts require the LAPDMouse dataset. Download it from https://lapdmouse.iibi.uiowa.edu/ and organize it as follows:

```
data_path/
├── m01/
│   ├── m01_Info.md
│   ├── m01_AirwayTreeTable.csv
│   ├── m01_AirwaySegmentsDeposition.csv
│   ├── m01_Ventilation_Pre.csv
│   ├── m01_Ventilation_Post1.csv
│   └── m01_Ventilation_Post2.csv
├── m02/
│   └── ...
└── ...
```

## Usage

### Compare impaction models

```bash
cd scripts

python compare_impaction_models.py \
    --data_path /path/to/lapdmouse/data \
    --results_path /path/to/output \
    --particle_sizes 5e-7 1e-6 2e-6 \
    --impaction_models chan_lipp yeh_schum zhang \
    --use_mouse_ventilation
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_path` | *(required)* | Path to the LAPDMouse data directory |
| `--results_path` | *(required)* | Output directory for results and figures |
| `--particle_sizes` | `1e-6` | Particle diameters in meters (space-separated) |
| `--impaction_models` | `chan_lipp zhang yeh_schum` | Impaction formula(s) to compare |
| `--use_mouse_ventilation` | False | Use per-mouse breathing parameters (RR, Vt, I:E) |
| `--Q_intake` | `2.08e-6` | Intake flow rate in m³/s (ignored if `--use_mouse_ventilation`) |
| `--mice_ids` | all | Subset of mouse IDs to process |
| `--use_outlet_area` | False | Use outlet area in flow calculations |

### Use the model directly

```python
from pathlib import Path
from scripts.model_deposition import (
    load_trees, propagate_flow, compute_probabilities,
    propagate_probabilities, compute_generation
)

data = load_trees(Path("/path/to/data"), mice_list=["m01"], Q_intake=2.08e-6)
tree = data["m01"]["tree_table"]

tree = propagate_flow(tree)
tree = compute_generation(tree)
tree = compute_probabilities(tree, particle_size=1e-6, impaction_model="chan_lipp")
tree = propagate_probabilities(tree)
```

See `notebooks/tutorial.ipynb` for a full walkthrough.

## Impaction models

Three variants of the impaction efficiency formula are implemented:

- **Chan-Lippmann** (`chan_lipp`): Chan & Lippmann (1980)
- **Yeh-Schum** (`yeh_schum`): Yeh & Schum (1980)
- **Zhang** (`zhang`): Zhang et al. (1997)

## License

MIT
