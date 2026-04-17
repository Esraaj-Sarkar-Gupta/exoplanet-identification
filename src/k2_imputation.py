"""
Data imputation for the K2 dataset.add

K2 is missing data in a large number of essential coloumns. Most of the dataset
is otherwised dropped during training and testing.

Applies K-Nearest Neighbors (KNN) imputation to fill missing values 
in the K2 dataset after adapting it to the standard KOI nomenclature.
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
k2_to_koi_map = {
    "pl_orbper"     : "koi_period",
    "pl_trandur"    : "koi_duration",
    "pl_trandep"    : "koi_depth",      
    "pl_rade"       : "koi_prad",
    "pl_eqt"        : "koi_teq",
    "pl_insol"      : "koi_insol",
    "st_teff"       : "koi_steff",
    "st_logg"       : "koi_slogg",
    "st_rad"        : "koi_srad",
    "disposition"   : "koi_disposition"
}

# The base features needed to compute your physical anchors later
features_to_impute = [
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_prad",
    "koi_teq", "koi_insol", "koi_steff", "koi_slogg", "koi_srad"
]

def adapt_to_koi_standard(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_adapted = df_raw.copy()
    df_adapted.rename(columns=k2_to_koi_map, inplace=True)
    
    # Convert K2 depth from Percent to PPM [1% = 10,000 ppm]
    if "koi_depth" in df_adapted.columns:
        df_adapted["koi_depth"] = df_adapted["koi_depth"] * 10000
        
    # Rigorous SNR calculation for K2 using the mean of asymmetrical errors
    if "pl_trandeperr1" in df_adapted.columns and "pl_trandeperr2" in df_adapted.columns and "koi_depth" in df_adapted.columns:
        mean_error_ppm = ((df_adapted["pl_trandeperr1"].abs() + df_adapted["pl_trandeperr2"].abs()) / 2) * 10000
        df_adapted["koi_model_snr"] = np.where(
            mean_error_ppm > 0, df_adapted["koi_depth"] / mean_error_ppm, np.nan
        )
    elif "pl_trandeperr1" in df_adapted.columns and "koi_depth" in df_adapted.columns:
        df_adapted["koi_model_snr"] = np.where(
            df_adapted["pl_trandeperr1"] > 0, df_adapted["koi_depth"] / (df_adapted["pl_trandeperr1"] * 10000), np.nan
        )
        
    # Standardize the Target column
    if "koi_disposition" in df_adapted.columns and "Target" not in df_adapted.columns:
        confirmed_labels = ['CONFIRMED', 'KP', 'CP'] 
        df_adapted['Target'] = df_adapted['koi_disposition'].apply(
            lambda x: 1.0 if str(x).upper() in confirmed_labels else 0.0
        )
        
    return df_adapted

def main():
    # -- Define Paths -- #
    k2_input_path = PROJECT_ROOT / "data" / "k2" / "k2pandc.csv"
    k2_output_path = PROJECT_ROOT / "data" / "k2" / "k2pandc_imputed.csv"
    
    print(f"Loading raw K2 data from: {k2_input_path}...")
    try:
        df_raw = pd.read_csv(k2_input_path, comment='#')
    except FileNotFoundError:
        print(f"Error: Could not find data at {k2_input_path}")
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
    
    #Save to new CSV
    k2_output_path.parent.mkdir(parents=True, exist_ok=True)
    df_adapted.to_csv(k2_output_path, index=False)
    print(f"\nSuccess! Imputed dataset saved to: {k2_output_path}")

if __name__ == "__main__":
    main()