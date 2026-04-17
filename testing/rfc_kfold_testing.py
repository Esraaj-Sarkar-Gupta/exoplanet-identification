"""
Docstring for testing.rfc_kfold_testing

Unified Random Forest Classifier k-fold testing pipeline.
Handles the primary KOI dataset as well as external datasets (TESS, K2) 
by adapting their nomenclature and evaluating them via internal cross-validation.

Author: Esraaj Sarkar Gupta
"""

# ---- Imports ---- #
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    precision_recall_curve, 
    auc
)

# ---- Secure Path Resolution ---- #
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR       
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# Import physics engine from src/model_inference.py
from model_inference import engineer_physics_features

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

# ---- Features Sets ---- #
features_4 = [
    "koi_period", "koi_model_snr", "koi_prad", "koi_steff", "koi_srad",
    "phys_depth_residual", "phys_duration_residual", 
    "phys_impact_parameter_squared", "phys_thermal_residual"
]

features_3 = [
    "koi_period", "koi_model_snr", "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_srad", "phys_depth_residual", "phys_duration_residual", 
    "phys_impact_parameter_squared"
]

ablation_features = [
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_prad",
    "koi_teq", "koi_insol", "koi_steff", "koi_slogg", "koi_srad"
]

# ---- Adapter Function ---- #
def adapt_to_koi_standard(dataset_name: str, dataset_path: Path) -> pd.DataFrame:
    print(f"Loading {dataset_name} raw data from: {dataset_path.name}...")
    try:
        # comment='#' ensures Exoplanet Archive headers are bypassed
        df = pd.read_csv(dataset_path, comment='#')
    except FileNotFoundError:
        print(f"Error: Could not find data at {dataset_path}")
        return None

    df_adapted = df.copy()
    
    # Apply column renaming and specific conversions
    if dataset_name.upper() == "TESS":
        df_adapted.rename(columns=tess_to_koi_map, inplace=True)
        
    elif dataset_name.upper() == "K2":
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
            
    elif dataset_name.upper() == "KOI":
        pass # KOI data is already correctly formatted
        
    else:
        print(f"Warning: Dataset name '{dataset_name}' not recognized.")
        
    # Standardize the Target column if it doesn't already exist
    if "koi_disposition" in df_adapted.columns and "Target" not in df_adapted.columns:
        confirmed_labels = ['CONFIRMED', 'KP', 'CP'] 
        df_adapted['Target'] = df_adapted['koi_disposition'].apply(
            lambda x: 1.0 if str(x).upper() in confirmed_labels else 0.0
        )
        
    return df_adapted


# ---- Main Pipeline ---- #
def main(datasets: dict, features: list, k_splits=10):
    
    for data_name, data_path in datasets.items():
        print("\n" + "="*50)
        print(f"K-FOLD TESTING ON DATASET: {data_name.upper()}")
        print("="*50)

        # ---- Adapt Dataset ---- #
        df_adapted = adapt_to_koi_standard(data_name, data_path)
        if df_adapted is None:
            continue

        # ---- Clean Missing Data ---- #
        required_columns = [
            "koi_slogg", "koi_srad", "koi_period", 
            "koi_prad", "koi_depth", "koi_duration", "Target"
        ]
        
        initial_len = len(df_adapted)
        df_clean = df_adapted.dropna(subset=required_columns).copy()
        print(f"Dropped {initial_len - len(df_clean)} rows containing NaNs in essential columns.")
        
        if len(df_clean) == 0:
            print(f"Skipping {data_name}: No valid rows remain.")
            continue

        # ---- Engineer Physics Features ---- #
        print("Computing physics-aware features...")
        try:
            df_engineered = engineer_physics_features(df_clean)
        except KeyError as e:
            print(f"Physics Engine Error: Details: {e}")
            continue

        # ---- Features and Target ---- #
        X = df_engineered[features]
        y = df_engineered["Target"]

        # ---- Initialize K-Fold Validation ---- #
        skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=24)
        
        fold_accuracies = []
        fold_pr_aucs = []

        print(f"\nStarting {k_splits}-Fold Stratified Cross-Validation...")
        print("-" * 50)

        # ---- Cross-Validation Loop ---- #
        for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Dataset-Specific (Local) Standardization inside the fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  
            
            # Initialize the Random Forest model
            rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=24)
            
            # Train and Predict
            rf_model.fit(X_train_scaled, y_train)
            predictions = rf_model.predict(X_test_scaled)
            probas = rf_model.predict_proba(X_test_scaled)[:, 1] 
            
            # Evaluate Metrics
            acc = accuracy_score(y_test, predictions)
            fold_accuracies.append(acc)
            
            precision, recall, _ = precision_recall_curve(y_test, probas)
            pr_auc = auc(recall, precision)
            fold_pr_aucs.append(pr_auc)

        # ---- Final Metrics ---- #
        print(f"\nOVERALL K-FOLD RESULTS FOR {data_name.upper()} (k={k_splits})")
        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
        print(f"Mean PR-AUC:   {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")
        print("="*50)


if __name__ == "__main__":
    
    # Easily add or remove datasets here. The pipeline handles them all!
    datasets_to_test = {
        "KOI": Path(PROJECT_ROOT / "data" / "koi" / "cleaned" / "q1_q17_koi.csv"),
        #"TESS": Path(PROJECT_ROOT / "data" / "toi" / "tois.csv"),
        #"K2": Path(PROJECT_ROOT / "data" / "k2" / "k2pandc.csv")
    }

    print("\n--- RUNNING ABLATION FEATURES ---")
    main(datasets_to_test, ablation_features, 5)
    
    print("\n--- RUNNING FEATURES 4 ---")
    main(datasets_to_test, features_4, 5)
    
    print("\n--- RUNNING FEATURES 3 ---")
    main(datasets_to_test, features_3, 5)