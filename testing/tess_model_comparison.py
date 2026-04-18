"""
Docstring for testing.tess_model_comparison

Head-to-head comparison on the TESS dataset:
1. HistGradientBoostingClassifier on RAW data (handles NaNs natively).
2. RandomForestClassifier on KNN-IMPUTED data.

Author: Esraaj Sarkar Gupta
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_curve, auc

# ---- Secure Path Resolution ---- #
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR       
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# Import physics engine
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

feature_sets = {
    "Ablation Features": ablation_features,
    "Features 3": features_3,
    "Features 4": features_4
}

# ---- Helper Functions ---- #
def adapt_raw_tess(dataset_path: Path) -> pd.DataFrame:
    """Loads and adapts raw TESS data with original mappings."""
    df = pd.read_csv(dataset_path, comment='#')
    df.rename(columns=tess_to_koi_map, inplace=True)
        
    if "koi_disposition" in df.columns:
        confirmed_labels = ['CONFIRMED', 'KP', 'CP'] 
        df['Target'] = df['koi_disposition'].apply(lambda x: 1.0 if str(x).upper() in confirmed_labels else 0.0)
        
    # Drop ONLY rows where the target label is missing. Keep all feature NaNs!
    df.dropna(subset=['Target'], inplace=True)
    return df

def load_imputed_tess(dataset_path: Path) -> pd.DataFrame:
    """Loads the previously imputed TESS data (already mapped to KOI standards)."""
    df = pd.read_csv(dataset_path)
    # Target is already processed in the imputed file, but just to be safe:
    df.dropna(subset=['Target'], inplace=True)
    return df

def run_kfold_evaluation(X: pd.DataFrame, y: pd.Series, model, model_name: str, k_splits=10):
    """Runs standard cross validation and prints results."""
    skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=24)
    fold_accuracies = []
    fold_pr_aucs = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  
        
        # Train and Predict
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probas = model.predict_proba(X_test_scaled)[:, 1] 
        
        # Metrics
        acc = accuracy_score(y_test, predictions)
        precision, recall, _ = precision_recall_curve(y_test, probas)
        pr_auc = auc(recall, precision)
        
        fold_accuracies.append(acc)
        fold_pr_aucs.append(pr_auc)

    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean PR-AUC:   {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")

# ---- Main Execution ---- #
if __name__ == "__main__":
    
    # Define Paths
    tess_raw_path = PROJECT_ROOT / "data" / "toi" / "tois.csv"
    tess_imputed_path = PROJECT_ROOT / "data" / "toi" / "tois_imputed.csv"
    
    print("\n" + "="*60)
    print("PHASE 1: NATIVE NaN HANDLING (HistGradientBoosting on RAW TESS)")
    print("="*60)
    
    df_raw = adapt_raw_tess(tess_raw_path)
    print("Computing physics-aware features (NaNs will propagate naturally)...")
    df_raw_engineered = engineer_physics_features(df_raw)
    
    hist_gb_model = HistGradientBoostingClassifier(max_iter=1000, random_state=24)
    
    for set_name, features in feature_sets.items():
        print(f"\n--- RUNNING {set_name.upper()} (HistGB) ---")
        X = df_raw_engineered[features]
        y = df_raw_engineered["Target"]
        run_kfold_evaluation(X, y, hist_gb_model, "HistGB", k_splits=10)


    print("\n\n" + "="*60)
    print("PHASE 2: IMPUTED DATA (RandomForest on KNN-Imputed TESS)")
    print("="*60)
    
    df_imputed = load_imputed_tess(tess_imputed_path)
    print("Computing physics-aware features on imputed values...")
    df_imputed_engineered = engineer_physics_features(df_imputed)
    
    rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1, random_state=24)
    
    for set_name, features in feature_sets.items():
        print(f"\n--- RUNNING {set_name.upper()} (RandomForest) ---")
        X = df_imputed_engineered[features]
        y = df_imputed_engineered["Target"]
        run_kfold_evaluation(X, y, rf_model, "RandomForest", k_splits=10)
        
    print("\n" + "="*60)
    print("TESTING COMPLETE.")
    print("="*60)