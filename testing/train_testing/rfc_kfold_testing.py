"""
Docstring for testing.rfc_kfold_testing

Random Forest Classifer k-fold testing pipeline.

Author: Esraaj Sarkar Gupta
"""

# ---- Imports ---- #
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

import sys
SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "src").exists():
    PROJECT_ROOT = SCRIPT_DIR       
else:
    PROJECT_ROOT = SCRIPT_DIR.parent 

SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

# Import physics engine from src/model_inference.py
from model_inference import engineer_physics_features

# Features to be passed the model

features_4 = [
        "koi_period",
        "koi_model_snr",
        "koi_prad",
        #"koi_teq",
        #"koi_insol",
        "koi_steff",
        "koi_srad",
        "phys_depth_residual",
        "phys_duration_residual",
        "phys_impact_parameter_squared",
        "phys_thermal_residual"
    ]

features_3 = [
        "koi_period",
        "koi_model_snr",
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_steff",
        "koi_srad",
        "phys_depth_residual",
        "phys_duration_residual",
        "phys_impact_parameter_squared",
        #"phys_thermal_residual"
    ]

features_all = [
        "koi_period",
        "koi_model_snr",
        "koi_prad",
        "koi_teq",
        "koi_insol",
        "koi_steff",
        "koi_srad",
        "phys_depth_residual",
        "phys_duration_residual",
        "phys_impact_parameter_squared",
        "phys_thermal_residual"
    ]

features_phys = [
    "phys_depth_residual",
    "phys_duration_residual",
    "phys_impact_parameter_squared",
    "phys_thermal_residual"
]


ablation_features = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_srad"
]

def main(features, k_splits = 10):
    # ---- Load Pre-Cleaned Data ---- #
    # Pointing to the exact directory and file you configured in your notebook
    data_path = PROJECT_ROOT / "data" / "koi" / "cleaned" / "q1_q17.csv"
    print(f"Loading pre-cleaned dataset: {data_path.name}...")
    
    try:
        df_clean = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find data at {data_path}. Ensure your notebook export cell ran successfully!")
        return

    # ---- Engineer Physics Features ---- #
    print("Computing physics-aware features...")
    df_engineered = engineer_physics_features(df_clean)

    # ---- Features and Target ---- #
    
    X = df_engineered[features]
    y = df_engineered["Target"]

    # ---- Initialize K-Fold Validation ---- #
    #k_splits = 5
    skf = StratifiedKFold(
        n_splits=k_splits,
        shuffle=True,
        random_state=24
        )
    
    fold_accuracies = []
    fold_pr_aucs = []

    print(f"\nStarting {k_splits}-Fold Stratified Cross-Validation...")
    print("="*50)

    # ---- Cross-Validation Loop ---- #
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        # Isolate the fold data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Scale the data dynamically to prevent data leakage into the test fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  
        
        # Initialize the Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=1000, 
            n_jobs=-1,          
            random_state=24     
        )
        
        # Train and Predict
        rf_model.fit(X_train_scaled, y_train)
        
        # Get hard predictions and positive class probabilities
        predictions = rf_model.predict(X_test_scaled)
        probas = rf_model.predict_proba(X_test_scaled)[:, 1] 
        
        # ---- Evaluate Metrics ---- #
        acc = accuracy_score(y_test, predictions)
        fold_accuracies.append(acc)
        
        # Calculate PR-AUC
        precision, recall, _ = precision_recall_curve(y_test, probas)
        pr_auc = auc(recall, precision)
        fold_pr_aucs.append(pr_auc)
        
        #print(f"> FOLD {fold}...")
        #print(f"Accuracy: {acc:.4f} | PR-AUC: {pr_auc:.4f}")
        #print("\nClassification Report:")
        #print(classification_report(y_test, predictions))

    # ---- Final Metrics ---- #
    print("\n" + "="*50)
    print(f"OVERALL K-FOLD RESULTS (k={k_splits})")
    print("="*50)
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean PR-AUC:   {np.mean(fold_pr_aucs):.4f} (+/- {np.std(fold_pr_aucs):.4f})")
    print("="*50)

if __name__ == "__main__":
    K = 100

    print(f"Ablation Test: KOI ONLY...")
    main(ablation_features,K)

    print(f"Physical anchors ONLY...")
    main(features_phys, K)

    print(f"Selection 3 -- First three physical anchors...")
    main(features_3, K)

    print(f"Selection 4 -- All four anchors. Fourth anchor in place of teq and insol...")
    main(features_4)

    print(f"All features listed here...")
    main(features_all)

