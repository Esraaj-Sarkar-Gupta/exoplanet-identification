"""
test_pipeline.py

Loads raw test data, passes it to the inference engine, 
and calculates performance metrics against the true labels.

Author: Esraaj Sarkar Gupta
"""

# ---- Imports ---- #
import sys
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---- Secure Path Resolution for Imports ---- #
# We must do this BEFORE importing model_inference
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))
from model_inference import get_svm_predictions

def main():
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
    print("Running raw data through the SVM Inference Engine...")
    # Note: get_svm_predictions handles all the physics engineering internally!
    try:
        predictions = get_svm_predictions(raw_data=df_test_clean)
    except ValueError as e:
        print(f"Inference Error: {e}")
        return
    except FileNotFoundError as e:
        print(f"File Error: {e}")
        return

    # ---- Performance Metrics ---- #
    print("\n" + "="*40)
    print("SVM TEST PERFORMANCE METRICS")
    print("="*40)
    
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()
