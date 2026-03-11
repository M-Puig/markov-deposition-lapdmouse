"""
Reference morphometric data from classic studies for comparison with LAPDMouse.

Sources:
- Weibel (1963): Human symmetric lung model, "Morphometry of the Human Lung"
- Islam et al. (2017): BALB/c mouse lung, Anatomical Record 300:2046-2057
- Oldham & Robinson (2007): BALB/c mouse lung averages
- Raabe et al. (1976): Rat lung morphometry (if OCR data becomes available)

Note: All diameters and lengths are in millimeters (mm) for consistency.
Generation 0 = trachea for human Weibel model, generation 1 = trachea for mouse models.
"""

import numpy as np
import pandas as pd


# =============================================================================
# WEIBEL (1963) HUMAN SYMMETRIC MODEL - Model A
# Source: Weibel, E.R. (1963). Morphometry of the Human Lung. Springer-Verlag.
# This is the idealized symmetric dichotomous model with 24 generations.
# =============================================================================

WEIBEL_HUMAN_1963 = pd.DataFrame({
    'generation': list(range(24)),
    'diameter_mm': [
        18.0,    # 0: Trachea
        12.2,    # 1: Main bronchi
        8.3,     # 2
        5.6,     # 3
        4.5,     # 4
        3.5,     # 5
        2.8,     # 6
        2.3,     # 7
        1.86,    # 8
        1.54,    # 9
        1.30,    # 10
        1.09,    # 11
        0.95,    # 12
        0.82,    # 13
        0.74,    # 14
        0.66,    # 15
        0.60,    # 16: Terminal bronchioles
        0.54,    # 17: Respiratory bronchioles 1
        0.50,    # 18: Respiratory bronchioles 2
        0.47,    # 19: Respiratory bronchioles 3
        0.45,    # 20: Alveolar ducts 1
        0.43,    # 21: Alveolar ducts 2
        0.41,    # 22: Alveolar ducts 3
        0.41,    # 23: Alveolar sacs
    ],
    'length_mm': [
        120.0,   # 0
        47.6,    # 1
        19.0,    # 2
        7.6,     # 3
        12.7,    # 4
        10.7,    # 5
        9.0,     # 6
        7.6,     # 7
        6.4,     # 8
        5.4,     # 9
        4.6,     # 10
        3.9,     # 11
        3.3,     # 12
        2.7,     # 13
        2.3,     # 14
        2.0,     # 15
        1.65,    # 16
        1.41,    # 17
        1.17,    # 18
        0.99,    # 19
        0.83,    # 20
        0.70,    # 21
        0.59,    # 22
        0.50,    # 23
    ],
    'n_airways': [2**g for g in range(24)],
    'source': ['Weibel1963'] * 24,
    'species': ['human'] * 24,
})

# Compute radius for consistency
WEIBEL_HUMAN_1963['radius_mm'] = WEIBEL_HUMAN_1963['diameter_mm'] / 2
WEIBEL_HUMAN_1963['L_D_ratio'] = WEIBEL_HUMAN_1963['length_mm'] / WEIBEL_HUMAN_1963['diameter_mm']


# =============================================================================
# ISLAM ET AL. (2017) BALB/c MOUSE - Automated CT measurements
# Source: Islam A, Oldham MJ, Wexler AS. Comparison of manual and automated 
# measurements of tracheobronchial airway geometry in three Balb/c mice.
# Anatomical Record, 2017, 300:2046-2057.
# 
# Data extracted from Supporting Information Tables S1-S3 for 3 BALB/c mice.
# Values are generation-averaged from automated CT measurements across all 3 lungs.
# Total segments: 188 (Lung 1: 62, Lung 2: 63, Lung 3: 63)
# =============================================================================

ISLAM_BALBC_2017 = pd.DataFrame({
    'generation': [1, 2, 3, 4, 5, 6],  # Note: Gen 1 = trachea in their notation
    'diameter_mm': [1.309, 1.189, 1.041, 0.794, 0.541, 0.375],
    'length_mm': [2.207, 3.881, 2.173, 1.283, 0.692, 0.509],
    'diameter_std': [0.158, 0.21, 0.249, 0.303, 0.269, 0.226],
    'length_std': [0.799, 1.533, 0.823, 0.59, 0.409, 0.26],
    'n_segments': [3, 6, 12, 24, 48, 95],
    'source': ['Islam2017'] * 6,
    'species': ['mouse_BALBC'] * 6,
})

ISLAM_BALBC_2017['radius_mm'] = ISLAM_BALBC_2017['diameter_mm'] / 2
ISLAM_BALBC_2017['L_D_ratio'] = ISLAM_BALBC_2017['length_mm'] / ISLAM_BALBC_2017['diameter_mm']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_diameter_ratio(df, diameter_col='diameter_mm'):
    """Compute the diameter ratio D_{G+1}/D_G per generation."""
    diameters = df[diameter_col].values
    ratios = diameters[1:] / diameters[:-1]
    return ratios


def compute_scaling_exponent(df, diameter_col='diameter_mm', generation_col='generation'):
    """
    Fit the diameter scaling: D = D_0 * r_D^G
    Returns (D_0, r_D) from exponential fit.
    """
    from scipy.optimize import curve_fit
    
    def exp_model(G, D0, r_D):
        return D0 * (r_D ** G)
    
    G = df[generation_col].values
    D = df[diameter_col].values
    
    popt, _ = curve_fit(exp_model, G, D, p0=[D[0], 0.8], maxfev=5000)
    return popt  # (D_0, r_D)


def normalize_by_trachea(df, diameter_col='diameter_mm', length_col='length_mm'):
    """
    Normalize diameters and lengths by tracheal values for cross-species comparison.
    """
    df = df.copy()
    D_trachea = df[diameter_col].iloc[0]
    L_trachea = df[length_col].iloc[0]
    
    df['diameter_normalized'] = df[diameter_col] / D_trachea
    df['length_normalized'] = df[length_col] / L_trachea
    
    return df


def get_all_mouse_references():
    """Return a combined DataFrame of all mouse reference data."""
    return ISLAM_BALBC_2017.copy()


def get_human_references():
    """Return human reference data."""
    return WEIBEL_HUMAN_1963.copy()