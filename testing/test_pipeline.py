"""
test_pipeline.py

Loads raw test data, passes it to the inference engine, 
and calculates performance metrics against the true labels.

Author: Esraaj Sarkar Gupta
"""

# ---- Imports ---- #
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    auc
)

# ---- Secure Path Resolution for Imports ---- #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))
from model_inference import get_predictions

# Main
def main(model_path : Path, scaler_path : Path, features : list, return_probs : bool):
    # ---- Secure Path Resolution ---- #
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_non_standardized_test.csv"

    # ---- Load Data ---- #
    print(f"Loading raw test data from: {DATA_PATH.name}...")
    try:
        df_test = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find data at {DATA_PATH}")
        return

    # ---- Clean Missing Data ---- #
    # We must drop NaNs in the raw columns used for physics calculations 
    # and the Target column so our predictions perfectly align with y_test.
    required_columns = [
        "koi_slogg", "koi_srad", "koi_period", 
        "koi_prad", "koi_depth", "koi_duration", "Target"
    ]
    
    initial_len = len(df_test)
    df_test_clean = df_test.dropna(subset=required_columns).copy()
    print(f"Dropped {initial_len - len(df_test_clean)} rows containing NaNs in essential columns.")

    # ---- Isolate Ground Truth ---- #
    y_test = df_test_clean["Target"]

    # ---- Run Inference ---- #
    print("Running raw data through the Inference Engine...")
    # Note: get__predictions handles all the physics engineering internally
    try:
        predictions, positive_class_probabilities = get_predictions(
            raw_data=df_test_clean,
            model_path = model_path,
            scaler_path = scaler_path,
            features = features,
            return_probs = return_probs
            )
    except ValueError as e:
        print(f"Inference Error: {e}")
        return
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        return

    # ---- Performance Metrics ---- #
    print("\n" + "="*40)
    print(f"{MODEL_PATH} TEST PERFORMANCE METRICS")
    print("="*40)
    
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # ---- PR AUC & Graphing ---- #
    if positive_class_probabilities is not None:
        # 1. Calculate Precision, Recall, and the AUC metric
        precision, recall, thresholds = precision_recall_curve(y_test, positive_class_probabilities)
        pr_auc = auc(recall, precision)
        
        print(f"PR-AUC Score: {pr_auc:.4f}\n")
        
        # Plot the Precision-Recall Curve
        print("Generating Precision-Recall Curve...")
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        
        plt.xlabel('Recall (True Positive Rate)', fontsize=12)
        plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
        plt.title('Precision-Recall Curve: Exoplanet Classification', fontsize=14)
        plt.legend(loc='lower left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save
        plot_path = DIR_PATH / "pr_curve.png"
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to: {plot_path}")

if __name__ == "__main__":
    DIR_PATH : Path     = Path(PROJECT_ROOT / "models" / "phys_plus_theo_plus_koi" / "SVM/" )
    MODEL_PATH : Path   = Path(DIR_PATH / "model.joblib")
    SCALER_PATH : Path  = Path(DIR_PATH / "scaler.joblib")

    features = [
    "phys_duration_residual",
    "phys_depth_residual",
    "phys_impact_parameter_squared",
    "koi_duration",
    "theo_radius_ratio",
    "koi_insol",
    "koi_teq",
    ]
    
    main(MODEL_PATH, SCALER_PATH, features, True)
