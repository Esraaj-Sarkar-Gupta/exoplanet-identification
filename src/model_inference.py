"""
Docstring for model_inference.py

    1) Loads raw NASA dataset inputs (KOI, TESS, K2)
    2) Computes physics-informed features
    3) Loads the trained SVM model and associated scaler from disk
    4) Applies the transformation
    5) Returns predictions

    This file can run predictions for any new raw exoplanet data. This is
    also part of the testing pipeline, where we will run our raw test data through.

Author: Esraaj Sarkar Gupta
"""

import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union

# ---- Secure Path Resolution ---- #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phys_only" / "svm" / "model.joblib"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "phys_only" / "svm" / "scaler.joblib"


# ---- Physics ---- #

# -- Physical Constants (SI Units) -- #
G_CONST = 6.6743e-11      # Gravitational constant in m^3 / (kg * s^2)
R_SUN = 6.957e8           # Radius of the Sun in meters
M_SUN = 1.9884e30         # Mass of the Sun in kilograms
AU_UNIT = 1.496e11        # Meters to Astronomical Units
R_EARTH = 6.371e6         # Radius of the Earth in meters
HOURS_PER_DAY = 24        # Number of hours in a day


# --- Conversion Functions ---
def slogg_to_si(slogg: float) -> float:
    """
    Converts log10(gravity) in cgs units (cm/s^2) to SI units (m/s^2).
    """
    g_cgs = 10 ** slogg       # Convert log10 back to linear cm/s^2
    g_si = g_cgs * 0.01       # Convert cm/s^2 to m/s^2
    return g_si

def srad_to_si(srad: float) -> float:
    """
    Converts stellar radius from Solar Radii to meters.
    """
    return srad * R_SUN

# -- Conversion Function -- #
def period_to_si(period_days: float) -> float:
    """
    Converts orbital period from days to seconds.
    """
    return period_days * 86400.0

def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all the derived and physics-aware features from raw data.
    Uses 'koi_' nomenclature, with safe handling for missing K2 variables.
    """
    
    # Work on a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # -- Derived Features -- #

    """
    1) Stellar Mass
    """
    stellar_mass = []

    for g, r in zip(df["koi_slogg"], df["koi_srad"]):
        # Convert to workable units
        g_si = slogg_to_si(g)
        r_si = srad_to_si(r)

        # Physics
        mass_kg = (g_si * (r_si ** 2)) / G_CONST

        # Convert back to solar masses
        mass_solar = mass_kg / M_SUN
        stellar_mass.append(mass_solar)

    df["phys_smass"] = np.array(stellar_mass)

    """
    2) Estimating the length of the semi-major axis
    """
    def a_kepler(M_s, P):
        numerator   = P**2 * G_CONST * M_s
        denominator = 4 * np.pi**2
        return np.cbrt(numerator / denominator)

    semi_major_axis = []

    for M_s, P in zip(df["phys_smass"], df["koi_period"]):
        # Convert to workable units
        M_s_si = M_s * M_SUN
        P_si = period_to_si(P)

        # Physics
        a = a_kepler(M_s_si, P_si)
        a_AU = a / AU_UNIT
        semi_major_axis.append(a_AU)

    df["phys_sma"] = np.array(semi_major_axis)

    # ---- Physics-Aware Features ---- #
    
    """
    1) Transit Depth Consistency
    """
    R_earth_to_sun = 0.009158
    df['theo_depth'] = ((df['koi_prad'] * R_earth_to_sun) / df['koi_srad'])**2 * 1e6

    # Safe residue: Leaves as NaN if raw depth is missing
    df['phys_depth_residual'] = np.where(
        df['koi_depth'].notna() & df['theo_depth'].notna(),
        (df['koi_depth'] - df['theo_depth']).abs(),
        np.nan
    )

    """
    2) Duration Consistency Anchoring
    """
    Rstar_by_AU_ratio = R_SUN / AU_UNIT
    df["theo_duration"] = (2 * df["koi_srad"] * df["koi_period"]) / (2 * np.pi * df["phys_sma"]) * Rstar_by_AU_ratio

    # Safe residue: Leaves as NaN if raw duration is missing
    df["phys_duration_residual"] = np.where(
        df["koi_duration"].notna() & df["theo_duration"].notna(),
        (df["koi_duration"] - df["theo_duration"]).abs(),
        np.nan
    )

    """
    3) Impact Parameter Consistency
    """
    AU_by_Rstar_ratio = AU_UNIT / R_SUN
    Rearth_by_Rstar_ratio = R_EARTH / R_SUN

    df["theo_radius_ratio"] = df["koi_prad"] / df["koi_srad"] * Rearth_by_Rstar_ratio 
    df["theo_distance_ratio"] = df["phys_sma"] / df["koi_srad"] * AU_by_Rstar_ratio

    # Safe impact parameter: Leaves as NaN if duration is missing
    df["phys_impact_parameter_squared"] = np.where(
        df["koi_duration"].notna(),
        (1 + df["theo_radius_ratio"])**2 - \
        (df["koi_duration"] * np.pi * df["theo_distance_ratio"] / (df["koi_period"] * HOURS_PER_DAY))**2,
        np.nan
    )

    """
    4) Thermal Consistency Anchoring
    """
    EARTH_TEQ = 255     # K
    df["phys_thermal"] = (df["koi_insol"] / (df["koi_teq"] / EARTH_TEQ)**4)
    df["phys_thermal_residual"] = 1 - df["phys_thermal"]

    return df


def get_predictions(
        raw_data: Union[pd.DataFrame, np.ndarray], 
        model_path: Path,
        scaler_path: Path,
        features : list[str],
        return_probs : bool = False
):
    """
    Takes raw dataframe, calculates physics features, loads the trained model and scaler, 
    transforms data, and returns predictions.
    """
    
    df_engineered = engineer_physics_features(raw_data)
    
    X_features = df_engineered[features]

    # Load from Path
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing model or scaler file. Ensure train_XXX.py was run first. Details: {e}")

    X_scaled = scaler.transform(X_features)
    predictions = model.predict(X_scaled)

    if return_probs:
        # [:, 1] grabs the probability of Class 1.0
        return predictions, model.predict_proba(X_scaled)[:, 1]
    
    return predictions, None