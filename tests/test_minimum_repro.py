from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from compare_impaction_models import run_for_mouse
from figure_helpers import run_all_figures
from model_deposition import add_child, compute_generation, load_trees
from sensitivity_helpers import compute_distal_volumes, scale_tree_to_tidal


FIXTURE = REPO / "tests" / "fixtures" / "mini_lapdmouse"


def test_model_smoke_on_synthetic_fixture(tmp_path):
    data = load_trees(FIXTURE, mice_list=["m01"], use_mouse_ventilation=True)
    mouse_data = data["m01"]

    tree, combined = run_for_mouse(
        "m01",
        mouse_data,
        particle_sizes=[1e-6],
        impaction_models=["chan_lipp", "yeh_schum", "zhang"],
    )

    assert len(tree) == 8
    assert {"generation", "flow", "chan_lipp_deposition_1.00e-06"}.issubset(combined.columns)
    dep = combined["chan_lipp_deposition_1.00e-06"].to_numpy(dtype=float)
    assert np.isfinite(dep).all()
    assert ((dep >= 0) & (dep <= 1)).all()

    results = tmp_path / "results" / "m01"
    results.mkdir(parents=True)
    tree.to_csv(results / "m01_tree_with_flow.csv", index=False)
    combined.to_csv(results / "m01_all_models_deposition_results.csv", index=False)
    mouse_data["deposition"].to_csv(results / "m01_ground_truth.csv", index=False)
    (results / "m01_info.txt").write_text(mouse_data["info"])

    out = tmp_path / "outputs"
    run_all_figures(tmp_path / "results", out)
    assert (out / "comparison_hq_1m.png").exists()
    assert (out / "dep_per_lobe_1.png").exists()
    assert (out / "morphometry_absolute_diameter.png").exists()


def test_tidal_scaling_only_changes_generation_four_plus():
    data = load_trees(FIXTURE, mice_list=["m01"], use_mouse_ventilation=True)
    tree = data["m01"]["tree_table"].copy()
    add_child(tree)
    compute_generation(tree, parent_label=1)

    scaled = scale_tree_to_tidal(tree, 0.5)
    proximal = tree["generation"] < 4
    distal = tree["generation"] >= 4

    pd.testing.assert_series_equal(scaled.loc[proximal, "radius"], tree.loc[proximal, "radius"])
    assert np.allclose(scaled.loc[distal, "radius"], tree.loc[distal, "radius"] * 0.5)

    volumes = compute_distal_volumes(scaled)
    assert set(volumes) == set(scaled["label"].astype(int))
    assert all(v > 0 for v in volumes.values())
