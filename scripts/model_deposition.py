import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

_STRAIN_RE = re.compile(r"^\s*\*\s*Strain\s*:\s*(.+?)\s*$", re.MULTILINE)
_ORIENT_RE = re.compile(
    r"^\s*\*\s*Approximate lung orientation\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _strain_from_info(info_text):
    """Extract the raw `* Strain: ...` value from Info.md, or None if absent."""
    m = _STRAIN_RE.search(info_text)
    return m.group(1).strip() if m else None


def _orientation_from_info(info_text):
    """Extract the raw `* Approximate lung orientation: ...` value, or None."""
    m = _ORIENT_RE.search(info_text)
    return m.group(1).strip() if m else None


def _orientation_to_las_rotation(orient):
    """Return a 3x3 rotation that takes vectors from the declared frame into
    the canonical LAS frame used by `compute_theta` (gravity = [0,-1,0]).

    Recognises the three conventions present in the LAPDMouse cohort:
      - 'LAS'                       -> identity
      - 'RPS'                       -> 180 deg around Z (flip X and Y signs)
      - 'LAS (rotated ~45 deg CCW)' -> -45 deg around Z

    Unknown labels return identity (callers should warn).
    """
    if orient is None:
        return np.eye(3)
    raw = orient.strip()
    upper = raw.upper()
    if upper == 'LAS':
        return np.eye(3)
    if upper == 'RPS':
        return np.diag([-1.0, -1.0, 1.0])
    if 'rotated' in raw.lower() and '45' in raw:
        c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0., 0., 1.0]])
    return np.eye(3)

MU_AIR = 1.8e-5 # air viscosity, Pa s
RHO_AIR = 1.2 # air density, kg/m^3
RHO_WATER = 1000 # water density, kg/m^3
TEMPERATURE = 310 # K
BOLTZMANN_CONSTANT = 1.38064852e-23 # m^2 kg / s^2 / K
GRAVITY = 9.81 # m/s^2

# Default breathing parameters (average across mice)
DEFAULT_RR = 228  # breaths per minute
DEFAULT_VT = 0.26e-6  # tidal volume in m³ (0.26 mL)
DEFAULT_IE = 0.9  # I:E ratio


def parse_breathing_parameters(info_text):
    """
    Parse breathing parameters from mouse info.txt content.
    
    Extracts RR (respiratory rate), Vt (tidal volume), VE (minute ventilation), 
    and I:E ratio from the "Pre Aerosol" row of the ventilation table.
    
    Parameters:
    - info_text: str, content of the info.txt file
    
    Returns:
    - dict with keys: 'RR_bpm', 'Vt_ml', 'VE_ml_min', 'IE_ratio'
      Returns None values if parsing fails
    """
    result = {'RR_bpm': None, 'Vt_ml': None, 'VE_ml_min': None, 'IE_ratio': None}
    
    for line in info_text.split('\n'):
        if 'Pre Aerosol' in line:
            # Parse: |Pre Aerosol        | 166      | 0.20    | 33.0        | 0.91 |
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 5:
                try:
                    rr = float(parts[1])
                    vt = float(parts[2])
                    ve = float(parts[3])
                    ie = float(parts[4])
                    # Check for NaN values (e.g., from "NaN" strings in the file)
                    if not (np.isnan(rr) or np.isnan(vt) or np.isnan(ve) or np.isnan(ie)):
                        result['RR_bpm'] = rr
                        result['Vt_ml'] = vt
                        result['VE_ml_min'] = ve
                        result['IE_ratio'] = ie
                except (ValueError, IndexError):
                    pass
            break
    
    return result


def compute_inspiratory_flow(RR_bpm=None, Vt_ml=None, IE_ratio=None):
    """
    Compute the average inspiratory flow rate from breathing parameters.
    
    Logic:
    1. I:E ratio gives inspiration fraction of breathing cycle
    2. RR gives cycle duration
    3. Inspiration time = cycle_time * IE/(1+IE)
    4. Inspiratory flow = Vt / inspiration_time
    
    Parameters:
    - RR_bpm: respiratory rate in breaths per minute (default: 228)
    - Vt_ml: tidal volume in mL (default: 0.26)
    - IE_ratio: inspiratory to expiratory time ratio (default: 0.9)
    
    Returns:
    - Q_insp: inspiratory flow rate in m³/s
    """
    # Use defaults if not provided
    if RR_bpm is None:
        RR_bpm = DEFAULT_RR
    if Vt_ml is None:
        Vt_ml = DEFAULT_VT * 1e6  # Convert default from m³ to mL
    if IE_ratio is None:
        IE_ratio = DEFAULT_IE
    
    # Breathing cycle duration (seconds)
    cycle_time = 60.0 / RR_bpm
    
    # Inspiration fraction of cycle: IE/(1+IE)
    insp_fraction = IE_ratio / (1.0 + IE_ratio)
    
    # Inspiration time (seconds)
    insp_time = cycle_time * insp_fraction
    
    # Tidal volume in m³
    Vt_m3 = Vt_ml * 1e-6
    
    # Inspiratory flow rate (m³/s)
    Q_insp = Vt_m3 / insp_time
    
    return Q_insp


def get_mouse_inspiratory_flow(info_text, use_defaults_on_failure=True):
    """
    Get the inspiratory flow rate for a mouse from its info.txt content.
    
    Parameters:
    - info_text: str, content of the info.txt file
    - use_defaults_on_failure: if True, use default values when parsing fails
    
    Returns:
    - Q_insp: inspiratory flow rate in m³/s
    - breathing_params: dict with parsed breathing parameters
    """
    params = parse_breathing_parameters(info_text)
    
    if params['RR_bpm'] is None and not use_defaults_on_failure:
        raise ValueError("Could not parse breathing parameters from info file")
    
    Q_insp = compute_inspiratory_flow(
        RR_bpm=params['RR_bpm'],
        Vt_ml=params['Vt_ml'],
        IE_ratio=params['IE_ratio']
    )
    
    return Q_insp, params


# Cunningham correction factor
A1 = 1.257
A2 = 0.4
A3 = 0.55
MFP_AIR = 6.6e-8 # mean free path, m

def Cunningham__correction(d_p):
    return 1.0 + (2*MFP_AIR/d_p)*(A1 + A2*np.exp(-2*A3*d_p/MFP_AIR))

# Stokes drag force, N
# v: particle speed, m/s
# r: particle radius, m
def stokes_number(v_f, rho_p, d_p, Cc, MU_AIR, D):
    return (v_f * rho_p * (d_p**2) * Cc) / (18 * MU_AIR * D)

# Reynolds number (particle-based) - uses particle diameter
# v: flow velocity, m/s
# d_p: particle diameter, m
def reynolds_number(v, d_p):
    return RHO_AIR * v * d_p / MU_AIR

# Alias for clarity
reynolds_particle = reynolds_number

# Reynolds number (duct-based) - uses duct diameter
# This is the correct Re for Zhang impaction formula
# v: flow velocity, m/s
# D: duct diameter, m
def reynolds_duct(v, D):
    return RHO_AIR * v * D / MU_AIR

# particle and flow speed, meters per second
def v_flow(Q,D):
    return Q / (np.pi * D**2 / 4)

def settling_velocity(Q, D, rho_f, mu_f, d_p, rho_p):
    Cc = Cunningham__correction(d_p)
    v_s = Cc * d_p**2 * rho_p * GRAVITY / (18 * MU_AIR)
    return v_s

def q_flow(Q_parent, r_children, outlet_area=0):
    children_area = [np.pi * elt**2 for elt in r_children]
    area_sum = sum(children_area + [outlet_area])
    return [Q_parent * elt / area_sum for elt in children_area]

# angle between the section and the horizontal plane
def compute_theta(x, y, z):
    gravity_direction = [0,-1,0]
    section_direction = [x,y,z]
    dot_product = np.dot(gravity_direction, section_direction)
    angle = np.arccos(dot_product)
    return np.pi/2-angle


def bifurcation_angle(vec1, vec2):
    """
    Compute the bifurcation angle between two 3D vectors.

    Parameters:
    - vec1: list or array-like, first direction vector [x1, y1, z1]
    - vec2: list or array-like, second direction vector [x2, y2, z2]

    Returns:
    - angle_rad: angle in radians
    - angle_deg: angle in degrees
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Compute the dot product
    dot_product = np.dot(v1_norm, v2_norm)
    
    # Clamp the dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Compute the angle in radians and degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_rad, angle_deg


def compute_bifurcation_angle(df_tree, label):
    """
    Compute the bifurcation angle between a given segment and its parent segment.
    This function uses the dot product to calculate the angle between the direction vectors of the parent and child segments.
    Parameters:
    - df_tree: DataFrame containing the airway tree data
    - label: label of the segment for which to compute the angle
    Returns:
    - angle: bifurcation angle in radians
    """
    # Get the segment with the given label
    segment = df_tree[df_tree['label'] == label]
    if segment.empty:
        raise ValueError(f"Segment with label {label} not found in the tree.")
    
    # Get the parent label
    parent_label = segment['parent'].values[0]
    if parent_label == 0:
        return 0.0  # No parent, angle is zero
    
    # Get the parent segment
    parent_segment = df_tree[df_tree['label'] == parent_label]
    
    # Calculate the angle using the dot product
    direction_child = np.array([segment['directionX'].values[0], segment['directionY'].values[0], segment['directionZ'].values[0]])
    direction_parent = np.array([parent_segment['directionX'].values[0], parent_segment['directionY'].values[0], parent_segment['directionZ'].values[0]])
    
    return bifurcation_angle(direction_parent, direction_child)[0]  # Return angle in radians


def compute_beta(df_tree):
    """
    Compute and store the bifurcation angle (beta, in radians) for every segment.
    Root segment gets beta=0. Result is written into df_tree['beta'] in place.
    """
    df_tree['beta'] = [compute_bifurcation_angle(df_tree, lbl) for lbl in df_tree['label'].values]


# sedimentation
def p_sedimentation(Q, theta, L, D, d_p, rho_p=RHO_WATER):
    v_s = settling_velocity(Q, D, RHO_AIR, MU_AIR, d_p, rho_p)
    kappa = 0.75 * (v_s / (Q / (np.pi * (D / 2)**2))) * (L / D) * np.cos(theta)
    # Clamp to [0, 1-eps] so arcsin/sqrt stay in domain; values >= 1 give proba = 1.0
    k = np.clip(kappa, 0.0, 1.0 - 1e-10)
    t1 = 2 * k * np.sqrt(1 - k**(2/3))
    t2 = -k**(1/3) * np.sqrt(1 - k**(2/3))
    t3 = np.arcsin(k**(1/3))
    proba = (2 / np.pi) * (t1 + t2 + t3)
    # Where kappa >= 1, complete sedimentation
    return np.where(kappa >= 1, 1.0, np.maximum(0.0, proba))

# Impaction
def p_impaction(Q=None, D=None, d_p=None, theta=None, beta=None, model="chan_lipp", fit_params=None, rho_p=RHO_WATER):
    """
    Calculate impaction probability based on the specified model.

    Parameters:
    - Q: flow rate (m^3/s)
    - D: tube diameter (m)
    - d_p: particle diameter (m)
    - theta: branching angle in radians (unused, kept for API compatibility)
    - beta: bifurcation angle in radians (required for zhang and yeh_schum)
    - model: 'chan_lipp', 'zhang', or 'yeh_schum'
    - fit_params: optional fit parameters passed to the model
    - rho_p: particle density (kg/m^3); default RHO_WATER (1000)

    Returns:
    - impaction probability (0 to 1)
    """
    v_fluid = v_flow(Q, D)
    Cc = Cunningham__correction(d_p)
    stk = stokes_number(v_fluid, rho_p, d_p, Cc, MU_AIR, D)
    Re_d = reynolds_duct(v_fluid, D)
    return p_impaction_from_stk(stk, model=model, beta=beta, Re=Re_d, fit_params=fit_params)

def p_impaction_from_stk(stk, model="chan_lipp", beta=None, Re=None, fit_params=None):
    """
    Calculate impaction probability from pre-computed Stokes number.

    Parameters:
    - stk: Stokes number (dimensionless)
    - model: 'chan_lipp', 'zhang', or 'yeh_schum'
    - beta: bifurcation angle in radians (required for zhang and yeh_schum)
    - Re: duct Reynolds number (required for zhang)
    - fit_params: optional fit parameters passed to the model

    Returns:
    - impaction probability (0 to 1)
    """
    if model == "chan_lipp":
        return p_impaction_chan_lipp(stk=stk)
    elif model == "zhang":
        if beta is None or Re is None:
            raise ValueError("For Zhang's formula, beta and Re must be provided.")
        return p_impaction_zhang(stk, Re, beta, fit_params=fit_params)
    elif model == "yeh_schum":
        if beta is None:
            raise ValueError("For Yeh and Schum's formula, beta must be provided.")
        return p_impaction_yeh_schum(stk, beta, fit_params=fit_params)
    else:
        raise ValueError(f"Invalid model '{model}'. Use 'chan_lipp', 'zhang', or 'yeh_schum'.")

# impaction
# stk: stokes number
def p_impaction_chan_lipp(stk):
    return np.minimum(1.606*stk+0.0023, 0.999)


def p_impaction_yeh_schum(stk, beta, fit_params=None):
    """
    Calculate impaction efficiency using Yeh and Schum (1980) formula.

    Parameters:
    - stk: Stokes number (dimensionless), scalar or array
    - beta: empirical constant (dimensionless), scalar or array
    - fit_params: optional tuple of fit parameters (a, b, c)

    Returns:
    - eta: impaction efficiency (0 to 1)
    """
    a, b, c = fit_params if fit_params is not None else (1.0, 1.0, 1.0)
    x = a * stk * beta
    # Clip to arccos domain; when x > 1 this yields theta=0, Pi=c*b=1 (fully impacted)
    theta = np.arccos(np.minimum(x, 1.0))
    Pi = c * (b - (2 / np.pi) * theta + (1 / np.pi) * np.sin(2 * theta))
    return np.clip(Pi, 0.0, 1.0)


def p_impaction_zhang(Stk, Re, beta, fit_params=None):
    """
    Calculate impaction efficiency using Zhang et al. (2001) formula.
    Supports both scalar and vector inputs.

    Parameters:
    - Stk: Stokes number (dimensionless)
    - Re: Reynolds number (dimensionless)
    - beta: impaction angle in radians
    - fit_params: optional tuple of fit parameters (a, b, c)
    Returns:
    - eta: impaction efficiency (0 to 1)
    """
    if fit_params is not None:
        a, b, c = fit_params
        # Clip exponent to prevent overflow (exp(700) ~ 1e304, near float64 max)
        raw_exponent = b * 55.7 * Stk**(a * 0.954)
        exponent = np.clip(raw_exponent, -700, 700)
        if np.any(raw_exponent != exponent):
            import warnings
            warnings.warn(
                f"Zhang model: exponent clipped to prevent overflow "
                f"(params a={a:.3f}, b={b:.3f})",
                RuntimeWarning
            )
        eta = c * 0.000654 * np.exp(exponent) * Re**(1/3) * np.sin(beta)
        return np.clip(eta, 0.0, 0.999)

    # Standard Zhang model (no fit params) — fully vectorized
    eta_low = 0.000654 * np.exp(55.7 * Stk**0.954) * Re**(1/3) * np.sin(beta)
    eta_high = (0.19 - 0.193 * np.exp(-9.5 * Stk**1.565)) * Re**(1/3) * np.sin(beta)
    eta = np.where(Stk < 0.04, eta_low, eta_high)
    return np.clip(eta, 0.0, 0.999)



# diffusion
# Q: flow rate, m^3/s
# D: tube diameter, m
# L: tube length, m
# r: particle radius, m
def p_diffusion(Q,D,L,d_p):
    Cc = Cunningham__correction(d_p)
    v_f = v_flow(Q,D)
    delta = BOLTZMANN_CONSTANT*TEMPERATURE*Cc*L/(3*np.pi*MU_AIR*d_p*v_f*(D/2)**2)
    
    proba = 1.0 - 0.819*np.exp(-14.63*delta) - 0.0976*np.exp(-89.22*delta) - 0.0325*np.exp(-228*delta) - 0.0509*np.exp(-125.9*(delta)**(2/3))
    proba = np.where(delta >= 0.16853, 1.0, proba)
    
    return np.minimum(proba, 0.999)

def propagate_flow(df_tree, parent_label, use_outlet=False, flow_split_fn=None):
    """
    Recursively propagate flow from `parent_label` to its descendants.

    flow_split_fn: optional callable (Q_parent, child_labels_array) -> Q_children_array.
        Receives child labels in the same order as the corresponding df rows; must
        return Q values in that same order. If None, the existing area-based
        q_flow is used (with outlet_area included when use_outlet=True).
    """
    # get the parent node
    parent_node = df_tree[df_tree['label'] == parent_label]
    # get the children
    children = parent_node[['child_1','child_2','child_3']].values[0]
    # get the flow
    Q_parent = parent_node['Q'].values[0]
    # get the number of children
    n_children = 0
    for child in children:
        if not np.isnan(child):
            n_children += 1

    # if there are children, propagate the flow
    if n_children > 0:
        children = children[:n_children]
        # get the children nodes (DataFrame ordering, used for both read and write)
        children_nodes = df_tree[df_tree['label'].isin(children)]
        if flow_split_fn is None:
            # existing area-based behavior
            r_children = children_nodes['radius'].values
            Q_children = q_flow(Q_parent, r_children, parent_node["outlet_area"].values[0] if use_outlet else 0)
        else:
            child_labels = children_nodes['label'].values
            Q_children = flow_split_fn(Q_parent, child_labels)
        # assign the flow to the children
        df_tree.loc[df_tree['label'].isin(children), 'Q'] = Q_children
        # propagate the flow to the children
        for child in children:
            propagate_flow(df_tree, child, use_outlet=use_outlet, flow_split_fn=flow_split_fn)

# aggregate the probabilities
def propagate_probabilities(df_tree, parent_label):

    # get the parent node
    parent_node = df_tree[df_tree['label'] == parent_label]
    
    if parent_node.empty:
        return

    # get the children
    children = parent_node[['child_1','child_2','child_3']].values[0]
    # get the number of children
    n_children = 0
    for child in children:
        if not np.isnan(child):
            n_children += 1
    # if there are children, propagate the probabilities
    if n_children > 0:
        children = children[:n_children]
        # aggregate the probabilities
        for child in children:
            q_ratio = df_tree.loc[df_tree['label']==child, 'Q'].to_numpy()/parent_node['Q'].to_numpy()
            df_tree.loc[df_tree['label']==child, 'p_escape_aggreg'] = q_ratio*parent_node['p_escape_aggreg'].to_numpy()*df_tree.loc[df_tree['label']==child, 'p_escape']
            df_tree.loc[df_tree['label']==child, 'p_deposition_aggreg'] = q_ratio*parent_node['p_escape_aggreg'].to_numpy()*(1-df_tree.loc[df_tree['label']==child, 'p_escape'])
        # propagate the probabilities to the children
        for child in children:
            propagate_probabilities(df_tree, child)

def compute_probabilities(df_tree, particle_diameter, model="yeh_schum", rho_p=RHO_WATER, **kwargs):
    """
    Compute per-segment deposition probabilities for sedimentation, impaction,
    diffusion, and the combined escape probability.

    Impaction uses parent-segment flow/geometry because impaction occurs at the
    bifurcation inlet, not within the segment itself.

    rho_p: particle density (kg/m^3); default RHO_WATER (1000).
    """
    parametric_factor = kwargs.get('parametric_factor', 0.0)
    fit_params = kwargs.get('fit_params', None)

    D = 2 * df_tree['radius'].values
    Q = df_tree['Q'].values
    theta = df_tree['theta'].values
    L = df_tree['length'].values

    # --- sedimentation & diffusion: fully vectorized ---
    df_tree['p_sedimentation'] = p_sedimentation(Q, theta, L, D, particle_diameter, rho_p=rho_p)
    df_tree['p_diffusion'] = p_diffusion(Q, D, L, particle_diameter)

    # --- impaction: vectorized using parent Q/D ---
    # beta is pre-computed in load_trees; fall back to on-the-fly if missing
    if 'beta' in df_tree.columns:
        biff_angles = df_tree['beta'].values
    else:
        biff_angles = np.array([compute_bifurcation_angle(df_tree, lbl) for lbl in df_tree['label'].values])

    # Build parent Q/D arrays via positional index lookup (use segment's own as dummy for root)
    label_to_idx = {lbl: i for i, lbl in enumerate(df_tree['label'].values)}
    parent_pos = np.array([label_to_idx.get(p, i) for i, p in enumerate(df_tree['parent'].values)])
    Q_parent = Q[parent_pos]
    D_parent = D[parent_pos]
    non_root = df_tree['parent'].values != 0

    p_imp = p_impaction(Q_parent, D_parent, particle_diameter, beta=biff_angles, model=model, fit_params=fit_params, rho_p=rho_p)
    df_tree['p_impaction'] = np.where(non_root, p_imp, 0.0)

    # --- parametric factor ---
    df_tree['p_parametric'] = np.clip(parametric_factor * biff_angles, 0.0, 1.0)

    # --- escape probability ---
    df_tree['p_escape'] = (
        (1 - df_tree['p_sedimentation']) *
        (1 - df_tree['p_impaction']) *
        (1 - df_tree['p_diffusion']) *
        (1 - df_tree['p_parametric'])
    )

def add_child(df_tree):
    """Populate child_1, child_2, child_3 columns from the parent column."""
    df_tree['child_1'] = np.nan
    df_tree['child_2'] = np.nan
    df_tree['child_3'] = np.nan

    children = df_tree[df_tree['parent'] != 0][['label', 'parent']].copy()
    children['child_rank'] = children.groupby('parent').cumcount()
    children = children[children['child_rank'] < 3]  # max 3 children, like the original

    child_cols = ['child_1', 'child_2', 'child_3']
    for _, row in children.iterrows():
        parent_loc = df_tree.index[df_tree['label'] == row['parent']]
        df_tree.loc[parent_loc, child_cols[int(row['child_rank'])]] = int(row['label'])

# Q_intake: flow rate at the intake, m^3/s
def load_trees(file_path, mice_list=None, Q_intake=20*1e-6, use_mouse_ventilation=False):
    """
    Load mouse airway tree data.
    
    Parameters:
    - file_path: Path to the data directory
    - mice_list: list of mouse IDs to load (None = all)
    - Q_intake: default intake flow rate in m³/s (used if use_mouse_ventilation=False)
    - use_mouse_ventilation: if True, compute Q_intake from each mouse's breathing parameters
    
    Returns:
    - data_dict: dictionary with mouse data including tree_table, deposition, etc.
    """
    if mice_list is None:
        mice_list = []
        for mousePath in file_path.glob("*"):
            mice_list.append(mousePath.name)
    
    data_dict = {}
    # load the data
    for mousePath in file_path.glob("*"):
        mouseFolder = mousePath.name
        if mouseFolder in mice_list:
            data_dict[mouseFolder] = {}
            
            # Load info file
            with open(mousePath / (mouseFolder+"_Info.md")) as infoFile:
                data_dict[mouseFolder]["info"] = infoFile.read()
            
            # Compute per-mouse Q_intake from breathing parameters if requested
            mouse_Q_intake = Q_intake  # Default
            if use_mouse_ventilation:
                try:
                    mouse_Q_intake, breathing_params = get_mouse_inspiratory_flow(
                        data_dict[mouseFolder]["info"], 
                        use_defaults_on_failure=True
                    )
                    data_dict[mouseFolder]["breathing_params"] = breathing_params
                    data_dict[mouseFolder]["Q_intake"] = mouse_Q_intake
                    
                    # If Pre-Aerosol parsing yielded no data, defer the print
                    # and fix Q_intake via strain-mean fallback after the loop.
                    if breathing_params['RR_bpm'] is None:
                        pass
                    else:
                        print(f"  {mouseFolder}: Q_intake = {mouse_Q_intake*1e6:.2f} mL/s "
                              f"(RR={breathing_params['RR_bpm']:.0f}, Vt={breathing_params['Vt_ml']:.2f} mL, "
                              f"I:E={breathing_params['IE_ratio']:.2f})")
                except Exception as e:
                    print(f"  {mouseFolder}: Using default Q_intake ({Q_intake*1e6:.2f} mL/s) - {e}")
                    data_dict[mouseFolder]["breathing_params"] = None
                    data_dict[mouseFolder]["Q_intake"] = Q_intake
            else:
                data_dict[mouseFolder]["Q_intake"] = Q_intake
                data_dict[mouseFolder]["breathing_params"] = None
            
            # Load ventilation pre file
            vent_pre_file = mousePath / (mouseFolder+"_Ventilation_Pre.csv")
            if vent_pre_file.exists():
                data_dict[mouseFolder]["ventilation_pre"] = pd.read_csv(vent_pre_file)
            else:
                data_dict[mouseFolder]["ventilation_pre"] = None
                print(f"Warning: Ventilation_Pre file not found for {mouseFolder}")
            
            # Load ventilation post1 file
            vent_post1_file = mousePath / (mouseFolder+"_Ventilation_Post1.csv")
            if vent_post1_file.exists():
                data_dict[mouseFolder]["ventilation_post1"] = pd.read_csv(vent_post1_file)
            else:
                data_dict[mouseFolder]["ventilation_post1"] = None
                print(f"Warning: Ventilation_Post1 file not found for {mouseFolder}")
            
            # Load ventilation post2 file
            vent_post2_file = mousePath / (mouseFolder+"_Ventilation_Post2.csv")
            if vent_post2_file.exists():
                data_dict[mouseFolder]["ventilation_post2"] = pd.read_csv(vent_post2_file)
            else:
                data_dict[mouseFolder]["ventilation_post2"] = None
                print(f"Warning: Ventilation_Post2 file not found for {mouseFolder}")
            
            # Load tree table file
            data_dict[mouseFolder]["tree_table"] = pd.read_csv(mousePath / (mouseFolder+"_AirwayTreeTable.csv"))
            
            # Load deposition file
            data_dict[mouseFolder]["deposition"] = pd.read_csv(mousePath / (mouseFolder+"_AirwaySegmentsDeposition.csv"))

            #load outlet summary file
            outlet_summary_file = mousePath / (mouseFolder+"_OutletSummary.csv")
            if outlet_summary_file.exists():
                data_dict[mouseFolder]["outlet_summary"] = pd.read_csv(outlet_summary_file)
            else:
                data_dict[mouseFolder]["outlet_summary"] = None
                print(f"Warning: OutletSummary file not found for {mouseFolder}")
            
            # add debit Q to the tree table
            data_dict[mouseFolder]["tree_table"]['Q'] = np.nan
            data_dict[mouseFolder]["tree_table"].loc[data_dict[mouseFolder]["tree_table"]['label'] == 1, "Q"] = mouse_Q_intake # m^3/s
            #df_treetable.loc[df_treetable['label'] == 1, "Q"] = 0.55*1e-6 # m^3/s
            data_dict[mouseFolder]["tree_table"]['radius'] *= 0.001 #mm to m
            data_dict[mouseFolder]["tree_table"]['length'] *= 0.001 #mm to m

            # Reorient direction & centroid columns into LAS so that the
            # gravity = [0,-1,0] assumption inside compute_theta holds for
            # every mouse, regardless of the imaging frame declared in
            # Info.md (LAS, RPS, or LAS rotated ~45 deg CCW in this cohort).
            orient = _orientation_from_info(data_dict[mouseFolder]["info"])
            data_dict[mouseFolder]["lung_orientation_raw"] = orient
            R = _orientation_to_las_rotation(orient)
            if not np.allclose(R, np.eye(3)):
                tt = data_dict[mouseFolder]["tree_table"]
                dirs = tt[['directionX', 'directionY', 'directionZ']].to_numpy()
                tt[['directionX', 'directionY', 'directionZ']] = dirs @ R.T
                cents = tt[['centroidX', 'centroidY', 'centroidZ']].to_numpy()
                tt[['centroidX', 'centroidY', 'centroidZ']] = cents @ R.T
                print(f"  {mouseFolder}: reoriented {orient!r} -> LAS")
            elif orient is not None and orient.strip().upper() != 'LAS':
                print(f"  {mouseFolder}: unrecognised orientation {orient!r}; "
                      f"left as-is (gravity assumption may be wrong)")

            data_dict[mouseFolder]["tree_table"]['theta'] = data_dict[mouseFolder]["tree_table"].apply(lambda row: compute_theta(row['directionX'], row['directionY'], row['directionZ']), axis=1)
            compute_beta(data_dict[mouseFolder]["tree_table"])

            data_dict[mouseFolder]["tree_table"]["outlet_area"] = 0.0
            if data_dict[mouseFolder]["outlet_summary"] is not None:
                for index, row in data_dict[mouseFolder]["outlet_summary"].iterrows():
                    label = row['segmentId']
                    area = row['total_outlet_area']*1e-6 # from mm^2 to m^2
                    data_dict[mouseFolder]["tree_table"].loc[data_dict[mouseFolder]["tree_table"]['label']==label, "outlet_area"] = area

            total_dep = (data_dict[mouseFolder]["deposition"]["mean"]*data_dict[mouseFolder]["deposition"]["area"]).sum()
            data_dict[mouseFolder]["deposition"]["probability"] = data_dict[mouseFolder]["deposition"]["mean"]*data_dict[mouseFolder]["deposition"]["area"] / total_dep

    # Strain-mean Q_insp fallback for mice with no Pre-Aerosol data.
    if use_mouse_ventilation:
        by_strain = {}
        for mid, md in data_dict.items():
            bp = md.get('breathing_params')
            strain = _strain_from_info(md.get('info', ''))
            if strain is None or bp is None or bp.get('RR_bpm') is None:
                continue
            by_strain.setdefault(strain, []).append(md['Q_intake'])
        strain_q = {s: float(np.mean(vs)) for s, vs in by_strain.items()}
        all_valid_q = [q for qs in by_strain.values() for q in qs]
        loaded_mean_q = float(np.mean(all_valid_q)) if all_valid_q else None
        for mid, md in data_dict.items():
            bp = md.get('breathing_params')
            if bp is not None and bp.get('RR_bpm') is not None:
                continue
            strain = _strain_from_info(md.get('info', ''))
            q = md['Q_intake']
            if strain in strain_q:
                q = strain_q[strain]
                src = f"strain-mean ({strain})"
            elif loaded_mean_q is not None:
                q = loaded_mean_q
                src = f"loaded-cohort-mean (no {strain} siblings)"
            else:
                src = "module default (no valid breathing data in loaded cohort)"
            md['Q_intake'] = q
            md['tree_table'].loc[md['tree_table']['label'] == 1, 'Q'] = q
            print(f"  {mid} ({strain}): no Pre-Aerosol -> {src} Q={q*1e6:.2f} mL/s")

    return data_dict

def compute_generation(df, parent_label=None):
    df["generation"] = np.nan
    # get the parent node
    for index in df.index:
        parent = df.loc[index,'parent']
        if parent==0: # Trachea has as parent value 0; we assign trachea generation '1'
            df.loc[index, "generation"] = 1
        else:
            df.loc[index, "generation"] = df.loc[df["label"] == parent, "generation"].values[0] + 1
    return df['generation']


def visualizeCompartments(data, value=None, cmap=None, alpha=1.0, save_path=None):
    if not cmap: cmap=matplotlib.colormaps.get_cmap('gray')
    plt.figure(figsize=(8,10))
    #value_max = data[value].max()
    min = value.min()#/value_max
    #make sure there is no 0 proba
    if min==0:
        value[value==0] = value[value!=0].min()
        min = value.min()
    max = value.max()#/value_max
    #norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
    print("min ", min)
    print("max ", max)
    norm = matplotlib.colors.LogNorm(vmin=min, vmax=max)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    for index, row in data.iterrows():
        if 'volume' in row:
            r=pow(3.0/4.0*row['volume']/np.pi,1.0/3.0)
        elif 'area' in row:
            r=pow(1.0/4.0*row['area']/np.pi,1.0/2.0)
        else:
            r=row["radius"]*1000
        c = 'lightgray'
        if value is not None:
            if isinstance(cmap, dict):
                if (index in value) and (value[index] in cmap): c = cmap[value[index]]
            else:
                c = mapper.to_rgba(value[index])
        a = alpha*0.5 if c=='lightgray' else alpha
        circle = matplotlib.patches.Circle((row['centroidX'],row['centroidZ']),
            radius=r,color=c,alpha=a,zorder=-row['centroidY'])
        plt.gca().add_patch(circle)
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
