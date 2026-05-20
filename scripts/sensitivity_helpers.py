"""
Shared utilities for the paper-revision sensitivity simulations
(Sim 1 volume-based flow split, Sim 2 strain-mean breathing, Sim 3 particle density).

All public functions are pure and don't write to disk. Sim runners orchestrate IO.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model_deposition import parse_breathing_parameters, compute_inspiratory_flow

# Lobe ordering used in every output CSV
LOBES = ['LMB', 'CrRMB', 'MiRMB', 'AcRMB', 'CaRMB']

# Brief's display spelling for strains. Maps from raw Info.md strain string
# (case/punctuation as found) to the display label used in output CSVs.
STRAIN_DISPLAY = {
    'B6C3F1': 'B6C3F1',
    'BALB/c': 'BALB/C',
    'BALB/C': 'BALB/C',
    'C57BL/6': 'C57Bl/6',
    'C57Bl/6': 'C57Bl/6',
    'C57BL/6J': 'C57Bl/6',
    'CD-1': 'CD-1',
    'CD1': 'CD-1',
}

# Info.md lobe-table label -> brief lobe code
GT_INFO_LOBE_TO_CODE = {
    'left': 'LMB',
    'cranial': 'CrRMB',
    'middle': 'MiRMB',
    'caudal': 'CaRMB',
    'accessory': 'AcRMB',
}

# Translate the internal impaction model key to the brief's required spelling
MODEL_DISPLAY = {
    'chan_lipp': 'chan_lippmann',
    'yeh_schum': 'yeh_schum',
    'zhang': 'zhang',
}


# ---------------------------------------------------------------------------
# Subtree airway volumes (Sim 1)
# ---------------------------------------------------------------------------

def compute_distal_volumes(df_tree: pd.DataFrame) -> dict[int, float]:
    """
    Post-order subtree airway volume per node:
        V_distal[node] = pi * radius^2 * length      (segment volume)
                       + sum(V_distal[children])

    Iterative — does not recurse, so it handles deep trees safely.

    Assumes df_tree has columns: 'label', 'parent', 'radius', 'length' and
    that radius/length are already in meters (load_trees converts mm -> m).

    Returns
    -------
    dict[int, float]
        Map from segment label to V_distal in m^3.
    """
    labels = df_tree['label'].astype(int).to_numpy()
    parents = df_tree['parent'].astype(int).to_numpy()
    radii = df_tree['radius'].to_numpy(dtype=float)
    lengths = df_tree['length'].to_numpy(dtype=float)

    v_seg = np.pi * radii * radii * lengths

    # Build children adjacency
    children: dict[int, list[int]] = {int(l): [] for l in labels}
    for lbl, par in zip(labels, parents):
        if par != 0:
            children[int(par)].append(int(lbl))

    # Iterative post-order: process a node only after all its children are done.
    v_distal: dict[int, float] = {}
    indeg = {int(l): len(children[int(l)]) for l in labels}
    # ready = leaves (no children)
    ready = [int(l) for l in labels if indeg[int(l)] == 0]
    label_to_idx = {int(l): i for i, l in enumerate(labels)}

    while ready:
        node = ready.pop()
        total = v_seg[label_to_idx[node]] + sum(v_distal[c] for c in children[node])
        v_distal[node] = float(total)
        par = int(parents[label_to_idx[node]])
        if par != 0:
            indeg[par] -= 1
            if indeg[par] == 0:
                ready.append(par)

    if len(v_distal) != len(labels):
        raise RuntimeError(
            f"compute_distal_volumes: processed {len(v_distal)} of {len(labels)} "
            "nodes — tree may have a cycle or orphan."
        )
    return v_distal


def make_q_flow_volume(v_distal: dict[int, float]):
    """
    Build a flow-split callable suitable for `propagate_flow(flow_split_fn=...)`.

    The callable receives (Q_parent, child_labels_array) and returns Q for each
    child proportional to the cumulative segmented-airway volume of that child's
    subtree (Asgharian-style volume split).
    """
    def q_flow_volume(Q_parent, child_labels):
        weights = np.array([v_distal[int(c)] for c in child_labels], dtype=float)
        total = weights.sum()
        if total <= 0.0:
            # Fallback to equal split if all weights vanish (shouldn't happen
            # in practice because every segment has positive r and length).
            n = len(weights)
            return np.full(n, Q_parent / n)
        return Q_parent * weights / total

    return q_flow_volume


def make_q_flow_hybrid(
    df_tree: pd.DataFrame,
    v_distal: dict[int, float],
    *,
    compliant_generation: int = 4,
):
    """
    Build a flow-split callable that uses area weights proximally and distal
    airway-volume weights once the daughter airways are compliant.

    The current flow callback receives child labels, not the parent label, so
    the switch is defined by daughter generation. With the TLC-to-tidal scaling
    convention used here, `compliant_generation=4` means generations 1-3 use
    cross-sectional area and splits into generation 4+ use subtree volume.
    """
    labels = df_tree['label'].astype(int).to_numpy()
    radius_by_label = dict(zip(labels, df_tree['radius'].to_numpy(dtype=float)))
    generation_by_label = dict(zip(labels, df_tree['generation'].to_numpy(dtype=float)))

    def q_flow_hybrid(Q_parent, child_labels):
        child_labels = [int(c) for c in child_labels]
        child_generations = [generation_by_label.get(c, np.nan) for c in child_labels]
        use_volume = all(
            np.isfinite(g) and g >= compliant_generation
            for g in child_generations
        )
        if use_volume:
            weights = np.array([v_distal[c] for c in child_labels], dtype=float)
        else:
            radii = np.array([radius_by_label[c] for c in child_labels], dtype=float)
            weights = np.pi * radii * radii

        total = weights.sum()
        if total <= 0.0:
            return np.full(len(child_labels), Q_parent / len(child_labels))
        return Q_parent * weights / total

    return q_flow_hybrid


# ---------------------------------------------------------------------------
# Lobe membership and aggregation
# ---------------------------------------------------------------------------

def assign_lobe_membership(df_tree: pd.DataFrame) -> pd.Series:
    """
    Walk the tree top-down. Each node inherits its parent's lobe assignment;
    if its own `name` is in LOBES, it starts a new lobe rooted at itself.

    Trachea + RMB-stem (and any segment outside named subtrees) -> NaN.

    Returns a Series of dtype object (lobe code or NaN), aligned with df_tree.
    """
    labels = df_tree['label'].astype(int).to_numpy()
    parents = df_tree['parent'].astype(int).to_numpy()
    names = df_tree['name'].astype(object).to_numpy()
    label_to_idx = {int(l): i for i, l in enumerate(labels)}

    # Build children adjacency for BFS
    children: dict[int, list[int]] = {int(l): [] for l in labels}
    for lbl, par in zip(labels, parents):
        if par != 0:
            children[int(par)].append(int(lbl))

    lobe = [None] * len(labels)
    # BFS from roots (parent == 0)
    queue: list[int] = [int(l) for l, p in zip(labels, parents) if p == 0]
    visited: set[int] = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        idx = label_to_idx[node]
        par = int(parents[idx])
        own = names[idx]
        if isinstance(own, str) and own in LOBES:
            lobe[idx] = own
        elif par != 0:
            lobe[idx] = lobe[label_to_idx[par]]
        # else: root -> remain None
        for ch in children[node]:
            queue.append(ch)

    return pd.Series(lobe, index=df_tree.index, dtype=object)


def _size_to_um(particle_size_m: float) -> float:
    return round(particle_size_m * 1e6, 4)


def aggregate_per_lobe(
    combined_results: pd.DataFrame,
    df_tree: pd.DataFrame,
    particle_sizes: Iterable[float],
    impaction_models: Iterable[str],
    *,
    mouse_id: str,
    strain: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a long-format DataFrame:
        mouse_id, strain, particle_um, model, lobe, deposition_fraction

    Drops segments outside the 5 named lobes (Trachea + RMB stem + any
    NaN-name segment) before normalizing to 1.0 across the 5 lobes.

    Returns the DataFrame plus a dropped-fraction dict keyed by
    (particle_size_m, model_internal_key) -> fraction-dropped (0..1).
    """
    lobe = assign_lobe_membership(df_tree)
    rows = []
    dropped = {}
    # Use 'label' alignment between df_tree and combined_results
    if not combined_results['label'].equals(df_tree['label']):
        # Re-sort combined_results to df_tree order for safety
        combined_results = combined_results.set_index('label').loc[df_tree['label']].reset_index()
    lobe_arr = lobe.to_numpy()
    for size in particle_sizes:
        for model in impaction_models:
            col = f"{model}_deposition_{size:.2e}"
            dep = combined_results[col].to_numpy(dtype=float)
            in_lobe = np.array([isinstance(x, str) for x in lobe_arr])
            # nansum: a few mice have disconnected segments with NaN deposition
            # (no flow propagated). Treat them as zero contribution.
            total_in = float(np.nansum(dep[in_lobe]))
            total_all = float(np.nansum(dep))
            dropped[(size, model)] = (total_all - total_in) / total_all if total_all > 0 else 0.0
            for lobe_code in LOBES:
                mask = (lobe_arr == lobe_code)
                lobe_sum = float(np.nansum(dep[mask]))
                fraction = lobe_sum / total_in if total_in > 0 else 0.0
                rows.append({
                    'mouse_id': mouse_id,
                    'strain': strain,
                    'particle_um': _size_to_um(size),
                    'model': MODEL_DISPLAY[model],
                    'lobe': lobe_code,
                    'deposition_fraction': fraction,
                })
    df = pd.DataFrame(rows)
    return df, dropped


def aggregate_per_generation(
    combined_results: pd.DataFrame,
    df_tree: pd.DataFrame,
    particle_sizes: Iterable[float],
    impaction_models: Iterable[str],
    *,
    mouse_id: str,
    strain: str,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame:
        mouse_id, strain, particle_um, model, generation, deposition_fraction

    All segments contribute to per-generation sums; rows are normalized so the
    fractions per (mouse, particle_size, model) sum to 1.0 over all generations.
    """
    if not combined_results['label'].equals(df_tree['label']):
        combined_results = combined_results.set_index('label').loc[df_tree['label']].reset_index()
    # Disconnected segments may have NaN generation; mask them out so
    # downstream int-cast and grouping skip them entirely.
    gen_raw = df_tree['generation'].to_numpy()
    valid_gen = ~np.isnan(gen_raw)
    gen_arr = gen_raw[valid_gen].astype(int)
    rows = []
    for size in particle_sizes:
        for model in impaction_models:
            col = f"{model}_deposition_{size:.2e}"
            dep_full = combined_results[col].to_numpy(dtype=float)
            dep = dep_full[valid_gen]
            total = float(np.nansum(dep))
            if total <= 0:
                continue
            for g in np.unique(gen_arr):
                gen_sum = float(np.nansum(dep[gen_arr == g]))
                rows.append({
                    'mouse_id': mouse_id,
                    'strain': strain,
                    'particle_um': _size_to_um(size),
                    'model': MODEL_DISPLAY[model],
                    'generation': int(g),
                    'deposition_fraction': gen_sum / total,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Info.md parsing
# ---------------------------------------------------------------------------

_STRAIN_RE = re.compile(r"^\s*\*\s*Strain\s*:\s*(.+?)\s*$", re.MULTILINE)
_LOBE_TABLE_RE = re.compile(
    r"^\s*(left|cranial|middle|caudal|accessory)\s*\|\s*([0-9.]+)",
    re.MULTILINE,
)


def parse_strain(info_text: str) -> str:
    """
    Extract strain from `* Strain: ...` line and normalize to brief's display
    spelling. Raises if not found.
    """
    m = _STRAIN_RE.search(info_text)
    if not m:
        raise ValueError("could not find '* Strain:' line in info text")
    raw = m.group(1).strip()
    if raw not in STRAIN_DISPLAY:
        raise ValueError(f"unknown strain '{raw}' — extend STRAIN_DISPLAY")
    return STRAIN_DISPLAY[raw]


def parse_lobe_volumes_mm3(info_text: str) -> dict[str, float]:
    """
    Parse the per-lobe Volume table at the bottom of Info.md:
        | left      | 389.46 | ...
        | cranial   | 193.90 | ...
        ...
    Returns {'LMB': 389.46, 'CrRMB': 193.90, ...} (in mm^3).
    """
    out = {}
    for m in _LOBE_TABLE_RE.finditer(info_text):
        word, vol = m.group(1), float(m.group(2))
        out[GT_INFO_LOBE_TO_CODE[word]] = vol
    if set(out.keys()) != set(LOBES):
        raise ValueError(
            f"Info.md lobe table parsed incomplete: {set(out.keys())} vs {set(LOBES)}"
        )
    return out


def gt_lobe_mapping_from_info(
    info_text: str,
    lobes_deposition_df: pd.DataFrame,
) -> dict[str, int]:
    """
    Match each lobe code (LMB, CrRMB, ...) to the integer label in
    {mouse}_LobesDeposition.csv by matching the `volume` column to the
    per-lobe volumes parsed from Info.md.

    Returns {'LMB': 1, 'CrRMB': 2, ...} (mapping is per-mouse but typically
    stable across the cohort).
    """
    info_vols = parse_lobe_volumes_mm3(info_text)  # mm^3
    csv_vols = lobes_deposition_df.set_index('label')['volume'].to_dict()
    mapping = {}
    used: set[int] = set()
    for code, target in info_vols.items():
        # Pick the unmatched label whose volume is closest
        best_lbl = min(
            (lbl for lbl in csv_vols if lbl not in used),
            key=lambda lbl: abs(csv_vols[lbl] - target),
        )
        if abs(csv_vols[best_lbl] - target) > 1.0:
            raise ValueError(
                f"GT lobe match failed for {code}: target {target} mm^3, "
                f"closest CSV label {best_lbl} has volume {csv_vols[best_lbl]}"
            )
        mapping[code] = int(best_lbl)
        used.add(int(best_lbl))
    return mapping


def gt_lobe_deposition_fractions(
    info_text: str,
    lobes_deposition_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Return the LAPDMouse measured deposition fraction per lobe, normalized so
    the 5 lobes sum to 1.0.

    Uses (mean * volume) per lobe as the per-lobe deposition mass (consistent
    with how the segment-level GT is normalized in load_trees).
    """
    mapping = gt_lobe_mapping_from_info(info_text, lobes_deposition_df)
    lookup = lobes_deposition_df.set_index('label')
    masses = {
        code: float(lookup.loc[lbl, 'mean']) * float(lookup.loc[lbl, 'volume'])
        for code, lbl in mapping.items()
    }
    total = sum(masses.values())
    if total <= 0:
        raise ValueError("GT lobe masses sum to zero")
    return {code: masses[code] / total for code in LOBES}


# ---------------------------------------------------------------------------
# Strain-mean breathing (Sim 2)
# ---------------------------------------------------------------------------

def collect_breathing_table(data_path: Path, mice_ids: Iterable[str]) -> pd.DataFrame:
    """
    For every mouse in mice_ids, parse Pre-Aerosol breathing params and strain
    from {mouse}_Info.md. Returns a DataFrame with columns:
        mouse_id, strain, RR_bpm, Vt_ml, IE_ratio, Q_insp_m3s
    Mice with missing Pre-Aerosol data have NaN in the breathing columns.
    """
    rows = []
    for mid in mice_ids:
        info_path = Path(data_path) / mid / f"{mid}_Info.md"
        info = info_path.read_text()
        strain = parse_strain(info)
        params = parse_breathing_parameters(info)
        if params['RR_bpm'] is not None:
            q_insp = compute_inspiratory_flow(
                RR_bpm=params['RR_bpm'],
                Vt_ml=params['Vt_ml'],
                IE_ratio=params['IE_ratio'],
            )
        else:
            q_insp = np.nan
        rows.append({
            'mouse_id': mid,
            'strain': strain,
            'RR_bpm': params['RR_bpm'] if params['RR_bpm'] is not None else np.nan,
            'Vt_ml': params['Vt_ml'] if params['Vt_ml'] is not None else np.nan,
            'IE_ratio': params['IE_ratio'] if params['IE_ratio'] is not None else np.nan,
            'Q_insp_m3s': q_insp,
        })
    return pd.DataFrame(rows)


def strain_breathing_means(df_breathing: pd.DataFrame) -> pd.DataFrame:
    """
    Group by strain and compute mean/SD of RR/Vt/IE/Q_insp across mice with
    valid Pre-Aerosol data. Q_insp is stored internally in m^3/s, with mL/s
    convenience columns added for human-readable tables:

        strain, n_mice, mean_RR, sd_RR, mean_Vt, sd_Vt, mean_IE, sd_IE,
        mean_Qinsp_m3s, sd_Qinsp_m3s, mean_Qinsp_ml_s, sd_Qinsp_ml_s
    """
    valid = df_breathing.dropna(subset=['RR_bpm', 'Vt_ml', 'IE_ratio', 'Q_insp_m3s'])
    rows = []
    for strain, group in valid.groupby('strain'):
        mean_q_m3s = float(group['Q_insp_m3s'].mean())
        sd_q_m3s = float(group['Q_insp_m3s'].std(ddof=1)) if len(group) > 1 else float('nan')
        rows.append({
            'strain': strain,
            'n_mice': int(len(group)),
            'mean_RR': float(group['RR_bpm'].mean()),
            'sd_RR': float(group['RR_bpm'].std(ddof=1)) if len(group) > 1 else float('nan'),
            'mean_Vt': float(group['Vt_ml'].mean()),
            'sd_Vt': float(group['Vt_ml'].std(ddof=1)) if len(group) > 1 else float('nan'),
            'mean_IE': float(group['IE_ratio'].mean()),
            'sd_IE': float(group['IE_ratio'].std(ddof=1)) if len(group) > 1 else float('nan'),
            'mean_Qinsp_m3s': mean_q_m3s,
            'sd_Qinsp_m3s': sd_q_m3s,
            'mean_Qinsp_ml_s': mean_q_m3s * 1e6,
            'sd_Qinsp_ml_s': sd_q_m3s * 1e6,
        })
    # Order rows in the brief's strain order
    order = ['B6C3F1', 'BALB/C', 'C57Bl/6', 'CD-1']
    df = pd.DataFrame(rows)
    df['__order'] = df['strain'].map({s: i for i, s in enumerate(order)})
    df = df.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# TLC -> tidal-volume airway scaling (Hofmann 2011)
# ---------------------------------------------------------------------------

# Strain-specific functional residual capacity (mL).
# BALB/C and C57Bl/6: Rojas-Ruiz et al. 2023, Table 1 (direct).
# B6C3F1: midpoint of C57BL/6 (0.25) and C3H/HeJ (~0.35); F1 cross of these strains.
# CD-1: BALB/c proxy; CD-1 is larger so this is conservative.
STRAIN_FRC_ML = {
    'BALB/C':  0.31,
    'C57Bl/6': 0.25,
    'B6C3F1':  0.30,
    'CD-1':    0.31,
}

_LUNG_VOL_RE = re.compile(
    r"^\s*\*\s*Lung volume\s*:\s*([0-9.]+)\s*\(\s*mm\^3\s*\)",
    re.MULTILINE | re.IGNORECASE,
)


def parse_lung_volume_mm3(info_text: str) -> float:
    """
    Parse the imaged total lung volume from Info.md:
        * Lung volume: 1128.40 (mm^3)
    Returns the value in mm^3. Raises ValueError if missing.
    """
    m = _LUNG_VOL_RE.search(info_text)
    if not m:
        raise ValueError("could not find '* Lung volume:' line in info text")
    return float(m.group(1))


def compute_tidal_scale_factor(
    info_text: str,
    *,
    vt_ml_override: float | None = None,
    frc_ml_override: float | None = None,
) -> dict:
    """
    Hofmann 2011 cube-root rescaling factor:
        s = ((FRC + V_t / 2) / V_TLC) ^ (1/3)

    Reads strain (-> FRC), V_TLC (Lung volume), and Pre-Aerosol V_t from
    Info.md. For mice with missing Pre-Aerosol V_t (m20, m30), the caller
    must pass `vt_ml_override` (typically the strain-mean V_t). For FRC
    sensitivity sweeps on strains without direct measurements (B6C3F1, CD-1),
    pass `frc_ml_override` to bypass STRAIN_FRC_ML for that mouse.

    Returns a dict with:
        strain      -- normalized strain display string
        frc_ml      -- FRC used (mL)
        frc_source  -- 'rojas_ruiz_2023' (default lookup) or 'override'
        vt_ml       -- V_t used (mL)
        vt_source   -- 'pre_aerosol' or 'strain_mean'
        v_tlc_ml    -- imaged Lung volume (mL)
        s           -- scaling factor
    """
    strain = parse_strain(info_text)
    if frc_ml_override is not None:
        frc_ml = float(frc_ml_override)
        frc_source = 'override'
    else:
        if strain not in STRAIN_FRC_ML:
            raise ValueError(f"no FRC entry for strain '{strain}'; extend STRAIN_FRC_ML")
        frc_ml = STRAIN_FRC_ML[strain]
        frc_source = 'rojas_ruiz_2023'
    v_tlc_ml = parse_lung_volume_mm3(info_text) / 1000.0  # mm^3 -> mL

    if vt_ml_override is not None:
        vt_ml = float(vt_ml_override)
        vt_source = 'strain_mean'
    else:
        params = parse_breathing_parameters(info_text)
        vt = params.get('Vt_ml')
        if vt is None:
            raise ValueError(
                "Pre-Aerosol Vt missing from info_text; supply vt_ml_override "
                "(e.g. strain-mean Vt for m20, m30)."
            )
        vt_ml = float(vt)
        vt_source = 'pre_aerosol'

    s = ((frc_ml + vt_ml / 2.0) / v_tlc_ml) ** (1.0 / 3.0)
    return {
        'strain': strain,
        'frc_ml': frc_ml,
        'frc_source': frc_source,
        'vt_ml': vt_ml,
        'vt_source': vt_source,
        'v_tlc_ml': v_tlc_ml,
        's': float(s),
    }


def scale_tree_to_tidal(df_tree: pd.DataFrame, s: float) -> pd.DataFrame:
    """
    Apply isotropic Hofmann 2011 scaling to the airway tree:
    multiply `radius` and `length` by `s` for every segment with
    generation >= 4.

    Note on indexing: this codebase uses 1-indexed generations
    (Trachea = gen 1, LMB/RMB = gen 2, inside-lobe = gen 3+). The
    Hofmann 2011 paper and reviewer assignment use 0-indexed generations
    (Trachea = gen 0). Holding generations 0, 1, and 2 fixed therefore
    corresponds to holding this codebase's generations 1, 2, and 3 fixed
    and scaling generation 4+.

    Direction vectors and centroids are NOT scaled — isotropic scaling
    preserves angles between unit direction vectors, so theta and beta
    (computed in load_trees) remain valid.

    Caller must populate `df_tree['generation']` first (via
    `compute_generation`). Caller is responsible for recomputing
    `V_distal` afterward (via `compute_distal_volumes`) before propagating
    flow under the volume rule.

    Returns a copy with the scaled radii/lengths; does not mutate input.
    """
    if 'generation' not in df_tree.columns:
        raise ValueError(
            "scale_tree_to_tidal needs 'generation' column populated; "
            "call compute_generation first."
        )
    out = df_tree.copy()
    gen = out['generation'].to_numpy()
    # Treat NaN generations as proximal (don't scale) — disconnected
    # segments shouldn't move under scaling.
    mask = np.where(np.isnan(gen.astype(float)), False, gen.astype(float) >= 4)
    out.loc[mask, 'radius'] = out.loc[mask, 'radius'].astype(float) * s
    out.loc[mask, 'length'] = out.loc[mask, 'length'].astype(float) * s
    return out
