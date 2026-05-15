import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from pytorch_tabnet.tab_model import TabNetClassifier

# ---- Secure Path Resolution & Imports ---- #
SCRIPT_DIR = Path(__file__).resolve().parent

# Dynamically climb up the directory tree until we find the "src" folder
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.append(str(PROJECT_ROOT / "src"))

from model_inference import engineer_physics_features

# ---- Mappings & Features ---- #
tess_to_koi_map = {
    "Period (days)": "koi_period", "Duration (hours)": "koi_duration",
    "Depth (ppm)": "koi_depth", "Planet SNR": "koi_model_snr",
    "Planet Radius (R_Earth)": "koi_prad", "Planet Equil Temp (K)": "koi_teq",
    "Planet Insolation (Earth Flux)": "koi_insol", "Stellar Eff Temp (K)": "koi_steff",
    "Stellar log(g) (cm/s^2)": "koi_slogg", "Stellar Radius (R_Sun)": "koi_srad",
    "TFOPWG Disposition": "koi_disposition"
}

features_driven = ["koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_prad", "koi_teq", "koi_insol", "koi_steff", "koi_slogg", "koi_srad"]
features_anchored = ["phys_depth_residual", "phys_duration_residual", "phys_impact_parameter_squared", "phys_thermal_residual"]
features_combined = ["koi_period", "koi_model_snr", "koi_prad", "koi_steff", "koi_srad", "phys_depth_residual", "phys_duration_residual", "phys_impact_parameter_squared", "phys_thermal_residual"]

feature_dict = {
    "Purely Data Driven (Ablation)": features_driven,
    "Only Anchored Features": features_anchored,
    "Best Combined (Features 4)": features_combined
}

def main():
    data_path = PROJECT_ROOT / "data" / "toi" / "tois_imputed.csv"
    print(f"Loading TESS data from: {data_path.name}...")
    
    df = pd.read_csv(data_path, comment='#')
    df.rename(columns=tess_to_koi_map, inplace=True)
    
    confirmed_labels = ['CONFIRMED', 'KP', 'CP'] 
    df['Target'] = df['koi_disposition'].apply(lambda x: 1.0 if str(x).upper() in confirmed_labels else 0.0)
    
    required_columns = ["koi_slogg", "koi_srad", "koi_period", "koi_prad", "koi_depth", "koi_duration", "Target"]
    df_clean = df.dropna(subset=required_columns).copy()
    
    print("Computing physics-aware features...")
    df_engineered = engineer_physics_features(df_clean)

    for set_name, features in feature_dict.items():
        print(f"\n{'='*50}\nRUNNING FEATURE SET: {set_name}\n{'='*50}")
        X = df_engineered[features].values 
        y = df_engineered["Target"].astype(int).values 

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=24)
        fold_accuracies, fold_pr_aucs = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  
            
            clf = TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, optimizer_params=dict(lr=2e-2), verbose=0)
            clf.fit(X_train_scaled, y_train, eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)], eval_name=['train', 'valid'], eval_metric=['auc'], max_epochs=100, patience=15, batch_size=256, virtual_batch_size=128)
            
            fold_accuracies.append(accuracy_score(y_test, clf.predict(X_test_scaled)))
            precision, recall, _ = precision_recall_curve(y_test, clf.predict_proba(X_test_scaled)[:, 1])
            fold_pr_aucs.append(auc(recall, precision))

        print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
        print(f"Mean PR-AUC:   {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")

if __name__ == "__main__":
    main()