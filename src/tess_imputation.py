"""
Docstring for data_cleaning.knn_impute_tess

Data imputation for the TESS dataset.

Applies K-Nearest Neighbors (KNN) imputation to fill missing values 
in the TESS dataset after adapting it to the standard KOI nomenclature.
Saves the result to a new CSV file.

Author: Esraaj Sarkar Gupta
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ---- Secure Path Resolution ---- #
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR       
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

# ---- Column Mapping ---- #
tess_to_koi_map = {
    "Period (days)"                 : "koi_period",
    "Duration (hours)"              : "koi_duration",
    "Depth (ppm)"                   : "koi_depth",
    "Planet SNR"                    : "koi_model_snr",
    "Planet Radius (R_Earth)"       : "koi_prad",
    "Planet Equil Temp (K)"         : "koi_teq",
    "Planet Insolation (Earth Flux)": "koi_insol",
    "Stellar Eff Temp (K)"          : "koi_steff",
    "Stellar log(g) (cm/s^2)"       : "koi_slogg",
    "Stellar Radius (R_Sun)"        : "koi_srad",
    "TFOPWG Disposition"            : "koi_disposition"
}

# The base features needed to compute physical anchors later
features_to_impute = [
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_prad",
    "koi_teq", "koi_insol", "koi_steff", "koi_slogg", "koi_srad"
]

def adapt_to_koi_standard(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_adapted = df_raw.copy()
    df_adapted.rename(columns=tess_to_koi_map, inplace=True)
        
    # Standardize the Target column
    if "koi_disposition" in df_adapted.columns and "Target" not in df_adapted.columns:
        confirmed_labels = ['CONFIRMED', 'KP', 'CP'] 
        df_adapted['Target'] = df_adapted['koi_disposition'].apply(
            lambda x: 1.0 if str(x).upper() in confirmed_labels else 0.0
        )
        
    return df_adapted

def main():
    # -- Define Paths -- #
    tess_input_path = PROJECT_ROOT / "data" / "toi" / "tois.csv"
    tess_output_path = PROJECT_ROOT / "data" / "toi" / "tois_imputed.csv"
    
    print(f"Loading raw TESS data from: {tess_input_path}...")
    try:
        # comment='#' ensures Exoplanet Archive headers are bypassed
        df_raw = pd.read_csv(tess_input_path, comment='#')
    except FileNotFoundError:
        print(f"Error: Could not find data at {tess_input_path}")
        return

    # -- Adapt Dataset -- #
    print("Adapting columns to KOI standards...")
    df_adapted = adapt_to_koi_standard(df_raw)
    
    # -- Clean Target Data -- #
    # We drop rows where 'Target' is missing (unlabeled data)
    initial_len = len(df_adapted)
    df_adapted.dropna(subset=['Target'], inplace=True)
    print(f"Dropped {initial_len - len(df_adapted)} rows missing target labels.")
    
    # Ensure all required columns exist
    for col in features_to_impute:
        if col not in df_adapted.columns:
            df_adapted[col] = np.nan

    # Extract features to impute
    X_missing = df_adapted[features_to_impute].copy()
    
    # -- Scale the features -- #
    print("Scaling features for distance calculations...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_missing)
    
    # -- Apply KNN Imputation -- #
    print("Running KNN Imputation (k=5)...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed_scaled = imputer.fit_transform(X_scaled)
    
    # -- Inverse transform to return data to original physical units -- #
    X_imputed = scaler.inverse_transform(X_imputed_scaled)
    
    # -- Reintegrate the imputed data into the main dataframe -- #
    df_adapted[features_to_impute] = X_imputed
    
    # Save to new CSV
    tess_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_adapted.to_csv(tess_output_path, index=False)
    print(f"\nSuccess! Imputed dataset saved to: {tess_output_path}")

if __name__ == "__main__":
    main()