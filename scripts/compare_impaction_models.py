import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import datetime
import json

from model_deposition import (
    load_trees, add_child, propagate_flow, compute_probabilities,
    propagate_probabilities, compute_generation, RHO_WATER,
    parse_breathing_parameters,
)
from sensitivity_helpers import (
    compute_distal_volumes, make_q_flow_volume, make_q_flow_hybrid,
    parse_strain, compute_tidal_scale_factor, scale_tree_to_tidal,
    collect_breathing_table, strain_breathing_means,
)


def run_for_mouse(mouse_id, mouse_data, particle_sizes, impaction_models,
                  *, flow_split_fn=None, use_outlet=False, rho_p=RHO_WATER):
    """
    Core per-mouse simulation: builds child links, propagates flow, runs each
    impaction model at each particle size, and returns a combined wide-format
    DataFrame (same schema as `{mouse_id}_all_models_deposition_results.csv`).

    Side-effect free aside from mutating `mouse_data["deposition"]` to merge
    geometry columns (matches existing main() behavior, kept for compatibility).

    Parameters
    ----------
    mouse_id : str
    mouse_data : dict
        Output of `load_trees`. Must contain 'tree_table', 'deposition',
        'Q_intake'.
    particle_sizes : iterable of float
        Particle diameters in meters.
    impaction_models : iterable of str
        Subset of {'chan_lipp', 'zhang', 'yeh_schum'}.
    flow_split_fn : callable or None
        Forwarded to `propagate_flow`. If None, area-based q_flow is used.
    use_outlet : bool
        Forwarded to `propagate_flow` (only meaningful when flow_split_fn=None).
    rho_p : float
        Particle density (kg/m^3); forwarded into `compute_probabilities`.

    Returns
    -------
    tree_table : pandas.DataFrame
        Tree with computed flow and generation columns.
    combined_results : pandas.DataFrame
        Per-segment deposition for every (model, particle_size) combination.
    """
    tree_table = mouse_data["tree_table"].copy()
    add_child(tree_table)

    mouse_Q_intake = mouse_data.get("Q_intake")
    tree_table.loc[tree_table['label'] == 1, 'Q'] = mouse_Q_intake
    propagate_flow(tree_table, parent_label=1, use_outlet=use_outlet, flow_split_fn=flow_split_fn)
    compute_generation(tree_table, parent_label=1)

    # Idempotent merge: only attach tree geometry columns the first time.
    if 'generation' not in mouse_data["deposition"].columns:
        mouse_data["deposition"] = mouse_data["deposition"].merge(
            tree_table[['label', 'parent', 'name', 'child_1', 'child_2', 'child_3', 'generation']],
            on='label')

    particle_sizes = list(particle_sizes)
    impaction_models = list(impaction_models)

    model_results = {}
    for impaction_model in impaction_models:
        deposition_results = pd.DataFrame()
        deposition_results['label'] = tree_table['label']
        deposition_results['parent'] = tree_table['parent']
        deposition_results['name'] = tree_table['name']
        deposition_results['child_1'] = tree_table['child_1']
        deposition_results['child_2'] = tree_table['child_2']
        deposition_results['child_3'] = tree_table['child_3']
        deposition_results['generation'] = tree_table['generation']
        deposition_results['radius'] = tree_table['radius']
        deposition_results['length'] = tree_table['length']
        deposition_results['flow'] = tree_table['Q']
        deposition_results['theta'] = tree_table['theta']
        deposition_results['beta'] = tree_table['beta']

        for particle_diameter in particle_sizes:
            particle_tree = tree_table.copy()
            particle_tree["p_sedimentation"] = np.nan
            particle_tree["p_impaction"] = np.nan
            particle_tree["p_diffusion"] = np.nan
            particle_tree["p_escape"] = np.nan
            particle_tree['p_deposition_aggreg'] = np.nan
            particle_tree['p_escape_aggreg'] = np.nan

            compute_probabilities(particle_tree, particle_diameter, model=impaction_model, rho_p=rho_p)

            particle_tree.loc[particle_tree['label'] == 1, 'p_deposition_aggreg'] = \
                1 - particle_tree.loc[particle_tree['label'] == 1, 'p_escape']
            particle_tree.loc[particle_tree['label'] == 1, 'p_escape_aggreg'] = \
                particle_tree.loc[particle_tree['label'] == 1, 'p_escape']

            propagate_probabilities(particle_tree, parent_label=1)

            column_name = f"deposition_{particle_diameter:.2e}"
            deposition_results[column_name] = particle_tree['p_deposition_aggreg']

            sed_column = f"sedimentation_{particle_diameter:.2e}"
            imp_column = f"impaction_{particle_diameter:.2e}"
            diff_column = f"diffusion_{particle_diameter:.2e}"
            deposition_results[sed_column] = particle_tree['p_sedimentation']
            deposition_results[imp_column] = particle_tree['p_impaction']
            deposition_results[diff_column] = particle_tree['p_diffusion']

        model_results[impaction_model] = deposition_results

    # Combine: same column layout as the legacy CLI output
    first_model = impaction_models[0]
    combined_results = model_results[first_model][[
        'label', 'parent', 'name', 'child_1', 'child_2', 'child_3',
        'generation', 'radius', 'length', 'flow', 'theta', 'beta'
    ]].copy()
    for impaction_model, model_df in model_results.items():
        for particle_diameter in particle_sizes:
            col_name = f"deposition_{particle_diameter:.2e}"
            combined_col = f"{impaction_model}_{col_name}"
            combined_results[combined_col] = model_df[col_name]
            mech_col = f"impaction_{particle_diameter:.2e}"
            combined_results[f"{impaction_model}_{mech_col}"] = model_df[mech_col]
            if impaction_model == first_model:
                for mech in ['sedimentation', 'diffusion']:
                    mech_col_only = f"{mech}_{particle_diameter:.2e}"
                    combined_results[mech_col_only] = model_df[mech_col_only]

    return tree_table, combined_results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare impaction models for particle deposition in mouse airway trees")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the directory containing mouse data")
    parser.add_argument("--results_path", type=str, required=True, help="Path to save comparison results")
    parser.add_argument("--Q_intake", type=float, default=2.08e-6, 
                        help="Intake flow rate in m³/s (default: 2.08e-6). Ignored if --use_mouse_ventilation is set.")
    parser.add_argument("--use_mouse_ventilation", action='store_true',
                        help="Use per-mouse breathing parameters (RR, Vt, I:E) to compute inspiratory flow rate")
    parser.add_argument("--particle_sizes", type=float, nargs='+', default=[1e-6], help="List of particle sizes in meters")
    parser.add_argument("--mice_ids", type=str, nargs='+', default=None, help="List of mouse IDs to process (default: all)")
    parser.add_argument("--impaction_models", type=str, nargs='+', default=['chan_lipp', 'zhang', 'yeh_schum'],
                        help="List of impaction models to compare (default: ['chan_lipp', 'zhang', 'yeh_schum'])")
    parser.add_argument("--use_outlet_area", action='store_true',
                        help="Whether to use outlet area in flow calculations")
    parser.add_argument("--flow_split", type=str, default='area', choices=['area', 'volume', 'hybrid'],
                        help="Bifurcation flow-split rule: 'area' (Q ∝ daughter cross-section, "
                             "default) or 'volume' (Q ∝ daughter subtree airway volume, "
                             "Asgharian-style), or 'hybrid' (area below the compliance threshold, "
                             "volume at/above it).")
    parser.add_argument("--hybrid_compliant_generation", type=int, default=4,
                        help="For --flow_split hybrid, daughter generation at which airways are "
                             "treated as compliant and use the volume split rule (default: 4).")
    parser.add_argument("--tlc_to_tidal_scaling", action='store_true',
                        help="Apply Hofmann 2011 cube-root scaling: scale radius/length by "
                             "s = ((FRC + V_t/2) / V_TLC)^(1/3) for generation >= 4 before "
                             "flow propagation. FRC is strain-specific (Rojas-Ruiz 2023).")
    parser.add_argument("--vt_strain_table", type=str,
                        default='outputs/sim2/sim2_strain_table.csv',
                        help="CSV with strain-mean V_t for m20/m30 fallback. Only consulted "
                             "when --tlc_to_tidal_scaling is set.")
    parser.add_argument("--frc_override", type=str, nargs='+', default=None,
                        help="FRC sensitivity overrides as STRAIN=VAL pairs (mL). "
                             "Bypasses STRAIN_FRC_ML for the named strains. "
                             "Example: --frc_override 'B6C3F1=0.25' 'CD-1=0.40'. "
                             "Only consulted when --tlc_to_tidal_scaling is set.")
    return parser.parse_args()


def save_parameters(args, results_path):
    """Save simulation parameters to a JSON file"""
    params = {
        'data_path': str(args.data_path),
        'results_path': str(args.results_path),
        'Q_intake_m3s': args.Q_intake,
        'Q_intake_mL_s': args.Q_intake * 1e6,
        'use_mouse_ventilation': args.use_mouse_ventilation,
        'particle_sizes_m': args.particle_sizes,
        'particle_sizes_um': [ps * 1e6 for ps in args.particle_sizes],
        'mice_ids': args.mice_ids if args.mice_ids else 'all',
        'impaction_models': args.impaction_models,
        'use_outlet_area': args.use_outlet_area,
        'flow_split': args.flow_split,
        'hybrid_compliant_generation': (
            args.hybrid_compliant_generation if args.flow_split == 'hybrid' else None
        ),
        'tlc_to_tidal_scaling': args.tlc_to_tidal_scaling,
        'vt_strain_table': args.vt_strain_table if args.tlc_to_tidal_scaling else None,
        'frc_override': args.frc_override if args.tlc_to_tidal_scaling else None,
        'simulation_date': datetime.datetime.now().strftime('%Y-%m-%d'),
        'simulation_time': datetime.datetime.now().strftime('%H:%M:%S')
    }
    
    params_file = results_path / 'simulation_parameters.json'
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Parameters saved to {params_file}")
    return params


def main():
    args = parse_arguments()
    data_path = Path(args.data_path)
    results_path = Path(args.results_path)
    results_path.mkdir(exist_ok=True, parents=True)
    particle_sizes = np.array(args.particle_sizes)
    
    # Save simulation parameters
    save_parameters(args, results_path)
    
    # Load mouse data with optional per-mouse ventilation
    if args.use_mouse_ventilation:
        print("Using per-mouse breathing parameters to compute inspiratory flow...")
    mice_data = load_trees(
        data_path, 
        mice_list=args.mice_ids, 
        Q_intake=args.Q_intake,
        use_mouse_ventilation=args.use_mouse_ventilation
    )
    if len(mice_data) == 0:
        print("Error: No mice data found. Please check the data path and mouse IDs.")
        return

    # Hofmann 2011 cube-root scaling: load strain-mean Vt fallback once.
    vt_strain_means = None
    frc_overrides = {}
    scale_log = []
    if args.tlc_to_tidal_scaling:
        import sys as _sys
        _sys.setrecursionlimit(5000)
        vt_table_path = Path(args.vt_strain_table)
        if vt_table_path.exists():
            vt_table = pd.read_csv(vt_table_path)
        else:
            print(f"Vt strain table not found at {vt_table_path}; computing it from loaded LAPDMouse metadata.")
            vt_table = strain_breathing_means(collect_breathing_table(data_path, mice_data.keys()))
            vt_table.to_csv(results_path / 'strain_ventilation_table.csv', index=False, float_format='%.6g')
            print(f"  wrote {results_path / 'strain_ventilation_table.csv'}")
        vt_strain_means = dict(zip(vt_table['strain'], vt_table['mean_Vt']))
        print(f"TLC->tidal scaling enabled. Strain-mean Vt (mL): {vt_strain_means}")
        if args.frc_override:
            for spec in args.frc_override:
                if '=' not in spec:
                    raise ValueError(f"--frc_override expects STRAIN=VAL, got {spec!r}")
                k, v = spec.split('=', 1)
                frc_overrides[k.strip()] = float(v)
            print(f"  FRC overrides (mL): {frc_overrides}")

    for mouse_id, mouse_data in tqdm(mice_data.items(), desc="Processing mice"):
        print(f"\nProcessing mouse: {mouse_id}")
        mouse_results_path = results_path / mouse_id
        mouse_results_path.mkdir(exist_ok=True)

        mouse_Q_intake = mouse_data.get("Q_intake", args.Q_intake)
        print(f"Propagating flow (Q={mouse_Q_intake*1e6:.2f} mL/s, split={args.flow_split})...")

        if args.tlc_to_tidal_scaling:
            tt = mouse_data['tree_table']
            add_child(tt)
            compute_generation(tt, parent_label=1)
            strain = parse_strain(mouse_data['info'])
            bp = parse_breathing_parameters(mouse_data['info'])
            vt_override = vt_strain_means[strain] if bp.get('Vt_ml') is None else None
            frc_override = frc_overrides.get(strain)
            info = compute_tidal_scale_factor(
                mouse_data['info'],
                vt_ml_override=vt_override,
                frc_ml_override=frc_override,
            )
            mouse_data['tree_table'] = scale_tree_to_tidal(tt, info['s'])
            scale_log.append({'mouse_id': mouse_id, **info})
            print(f"  scaling: strain={info['strain']} V_TLC={info['v_tlc_ml']:.4f} mL "
                  f"V_t={info['vt_ml']:.4f} mL ({info['vt_source']}) "
                  f"FRC={info['frc_ml']:.3f} mL ({info['frc_source']})  s={info['s']:.4f}")

        if args.flow_split == 'volume':
            v_distal = compute_distal_volumes(mouse_data['tree_table'])
            flow_split_fn = make_q_flow_volume(v_distal)
        elif args.flow_split == 'hybrid':
            if 'generation' not in mouse_data['tree_table'].columns:
                add_child(mouse_data['tree_table'])
                compute_generation(mouse_data['tree_table'], parent_label=1)
            v_distal = compute_distal_volumes(mouse_data['tree_table'])
            flow_split_fn = make_q_flow_hybrid(
                mouse_data['tree_table'],
                v_distal,
                compliant_generation=args.hybrid_compliant_generation,
            )
        else:
            flow_split_fn = None

        tree_table, combined_results = run_for_mouse(
            mouse_id, mouse_data, particle_sizes, args.impaction_models,
            flow_split_fn=flow_split_fn, use_outlet=args.use_outlet_area
        )

        # Save the tree with flow information and ground truth
        tree_table.to_csv(mouse_results_path / f"{mouse_id}_tree_with_flow.csv", index=False)
        mouse_data["deposition"].to_csv(mouse_results_path / f"{mouse_id}_ground_truth.csv", index=False)

        # Save metadata
        info_file = open(mouse_results_path / f"{mouse_id}_info.txt", "w")
        info_file.write(mouse_data["info"])
        info_file.close()

        # Compare models: plot deposition by generation for each model and particle size
        for particle_diameter in particle_sizes:
            plt.figure(figsize=(10, 6))
            for impaction_model in args.impaction_models:
                print(f"Plotting results for {impaction_model} model and particle size {particle_diameter:.2e} m")
                col = f"{impaction_model}_deposition_{particle_diameter:.2e}"
                gen_dep = combined_results.groupby('generation')[col].sum()
                plt.plot(gen_dep.index, gen_dep.values, label=f"{impaction_model}")
            generation_ground_truth = mouse_data["deposition"].groupby('generation').sum()
            plt.plot(generation_ground_truth.index, generation_ground_truth['probability'], 'k--', label='Ground Truth')
            plt.xlabel('Airway Generation')
            plt.ylabel('Deposition Probability')
            plt.title(f'Impaction Model Comparison - Mouse {mouse_id} - Particle {particle_diameter*1e9:.2f} nm')
            plt.legend()
            plt.grid(True)
            plt.savefig(mouse_results_path / f"{mouse_id}_impaction_model_comparison_{particle_diameter:.2e}.png", dpi=300)
            plt.close()

        combined_results.to_csv(mouse_results_path / f"{mouse_id}_all_models_deposition_results.csv", index=False)
        print(f"Completed comparison for mouse {mouse_id}")

    if args.tlc_to_tidal_scaling and scale_log:
        scale_df = pd.DataFrame(scale_log)[
            ['mouse_id', 'strain', 'frc_ml', 'frc_source',
             'vt_ml', 'vt_source', 'v_tlc_ml', 's']
        ]
        scale_df.to_csv(results_path / 'tidal_scaling_factors.csv',
                        index=False, float_format='%.6f')
        print(f"\nWrote {results_path / 'tidal_scaling_factors.csv'} "
              f"({len(scale_df)} rows, mean s={scale_df['s'].mean():.4f})")

    print("\nImpaction model comparison completed for all mice!")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
