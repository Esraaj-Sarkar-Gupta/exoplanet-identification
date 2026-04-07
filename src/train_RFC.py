"""
Docstring for train_SVM.py

This file trains a Support Vector Machine model on the KOI dataset
with derived and physiscs-aware features. Trained models are saved in the
models/ directory.

Author: Esraaj Sarkar Gupta
""" 

# ---- Imports ---- #
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

from pathlib import Path

SAVE_DIR = Path("models/phys_only/RF")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = SAVE_DIR / 'model.joblib'

# ---- Load Data ---- #
data_path = Path("data/physics/physics_aware_features_KOI.csv")
df = pd.read_csv(data_path)

features = [
        "phys_duration_residual",
        "phys_depth_residual",
        "phys_impact_parameter_squared",
    ]

X = df[features]
y = df["Target"]

# ---- Data Standardization ---- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Train Random Forest Model ---- #
rf_model = RandomForestClassifier(
    n_estimators=100,   # Number of trees in the forest
    n_jobs=-1,          # Uses all available CPU cores for speed
    random_state=24     # My birthday is the 24th of February!
)
rf_model.fit(X_scaled, y)

joblib.dump(rf_model, FILE_PATH)
print("Random Forest model trained and saved at: ", FILE_PATH)

joblib.dump(scaler, SAVE_DIR / 'scaler.joblib')
print("Scaler saved at: ", SAVE_DIR / 'scaler.joblib')