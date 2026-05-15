"""
Unified Random Forest Classifier k-fold testing pipeline with Feature Importance Visualization.
Handles the primary KOI dataset as well as external datasets (TESS, K2).

Author: Esraaj Sarkar Gupta
"""

# ---- Imports ---- #
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
try:
    from model_inference import engineer_physics_features
except ImportError:
    print("Warning: Could not import model_inference. Ensure it is in the src/ directory.")

# -- Coloumn Mapping -- #
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
    "pl_orbper"   : "koi_period",
    "pl_trandur"  : "koi_duration",
    "pl_trandep"  : "koi_depth",
    "pl_tranmid"  : "koi_model_snr",
    "pl_rade"     : "koi_prad",
    "pl_eqt"      : "koi_teq",
    "pl_insol"    : "koi_insol",
    "st_teff"     : "koi_steff",
    "st_logg"     : "koi_slogg",
    "st_rad"      : "koi_srad",
    "k2c_disp"    : "koi_disposition"
}

# ---- Presentation Plotting Function ---- #
def plot_inbuilt_feature_importance(importance_df, dataset_name):
    """Plots a presentation-ready horizontal bar chart for RF feature importances."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    txt_color = "#222222"
    rf_color = "#009999" # Deep Turquoise / Cyan
    
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    sns.barplot(
        x='Importance_Mean', 
        y='Feature', 
        data=importance_df, 
        color=rf_color, 
        edgecolor='black',
        linewidth=2,
        alpha=0.9,
        ax=ax
    )

    ax.errorbar(
        x=importance_df['Importance_Mean'], 
        y=range(len(importance_df)), 
        xerr=importance_df['Importance_Std'], 
        fmt='none', 
        c='black', 
        capsize=6,
        elinewidth=3,
        markeredgewidth=2
    )

    plt.title(f'Random Forest Feature Importance: {dataset_name}', 
              fontsize=20, fontweight='black', color=txt_color, pad=20)
    plt.xlabel('Mean Gini Importance (Across 10 Folds)', 
               fontsize=15, fontweight='bold', color=txt_color)
    plt.ylabel('Physics & Data Features', 
               fontsize=15, fontweight='bold', color=txt_color)
    
    for i, row in importance_df.reset_index(drop=True).iterrows():
        ax.text(
            x=row['Importance_Mean'] + row['Importance_Std'] + 0.005, 
            y=i, 
            s=f"{row['Importance_Mean']:.3f}", 
            color=txt_color, 
            fontweight='bold', 
            fontsize=12, 
            va='center'
        )

    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15)

    plt.tight_layout()
    plt.show()

# ---- Main Testing Pipeline ---- #
def test_dataset(data_path: Path, data_name: str, features: list, k_splits=10):
    print(f"\n{'='*50}\nEvaluating Dataset: {data_name.upper()}\n{'='*50}")
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    # Load Data
    df = pd.read_csv(data_path, comment='#')
    
    # Map columns based on dataset origin
    if data_name.upper() == "TESS":
        df = df.rename(columns=tess_to_koi_map)
    elif data_name.upper() == "K2":
        df = df.rename(columns=k2_to_koi_map)

    # 2. Engineer Physics Features
    df = engineer_physics_features(df)
    
    # Standardize the disposition column names if needed based on the mapped dataset
    if "koi_disposition" in df.columns:
        # Drop NaNs for the selected features and the label
        df = df.dropna(subset=features + ["koi_disposition"])
        
        # NASA sets use different labels for confirmed planets (e.g. 'KP', 'CP' for TESS)
        confirmed_labels = ['CONFIRMED', 'CANDIDATE', 'PC', 'KP', 'CP']
        y = df["koi_disposition"].apply(lambda x: 1 if str(x).upper() in confirmed_labels else 0)
    else:
        print(f"Error: Target column 'koi_disposition' not found after mapping in {data_name}")
        return

    X = df[features]

    # 3. Setup K-Fold
    skf = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=24)
    rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=24)
    scaler = StandardScaler()
    
    fold_accuracies = []
    fold_pr_aucs = []
    fold_importances = [] 
    
    print(f"Running {k_splits}-Fold Cross Validation...")
    
    # 4. Execute K-Fold Loop
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale Data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        
        # Store Feature Importances
        fold_importances.append(rf_model.feature_importances_)

    # 5. Final Printout
    print(f"\nOVERALL K-FOLD RESULTS FOR {data_name.upper()} (k={k_splits})")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean PR-AUC:   {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")
    print("="*50)

    # 6Process and Plot Feature Importances
    mean_importances = np.mean(fold_importances, axis=0)
    std_importances = np.std(fold_importances, axis=0)

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance_Mean': mean_importances,
        'Importance_Std': std_importances
    }).sort_values(by='Importance_Mean', ascending=False)
    
    print(f"\nGenerating Feature Importance Plot for {data_name.upper()}...")
    plot_inbuilt_feature_importance(importance_df, data_name.upper())


if __name__ == "__main__":
    
    datasets_to_test = {
        "KOI": Path(PROJECT_ROOT / "data" / "koi" / "cleaned" / "q1_q17_koi.csv"),
        "TESS": Path(PROJECT_ROOT / "data" / "toi" / "tois_imputed.csv"),
        "K2": Path(PROJECT_ROOT / "data" / "k2" / "k2pandc_imputed.csv")
    }

    # Features 4 / Mixed Set
    features_to_use = [
        "koi_period", "koi_duration", "koi_depth", "koi_model_snr", 
        "koi_prad", "koi_teq", "koi_insol", "koi_steff", "koi_srad",
        "phys_duration_residual", "phys_depth_residual", 
        "phys_impact_parameter_squared", "theo_duration", "theo_radius_ratio"
    ]

    for name, path in datasets_to_test.items():
        test_dataset(path, name, features_to_use)