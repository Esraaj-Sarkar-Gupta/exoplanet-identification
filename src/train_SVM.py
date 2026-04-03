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
from sklearn.svm import SVC
import joblib

from pathlib import Path

SAVE_DIR = Path("models/")
FILE_PATH = SAVE_DIR / 'svm_model.joblib'

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

# ---- Train SVM Model ---- #
svm_model = SVC(kernel='rbf', probability=True, random_state=24) # I am given the freedom to choose any number I find pretty
svm_model.fit(X_scaled, y)

joblib.dump(svm_model, FILE_PATH)
print("SVM model trained and saved at: ", FILE_PATH)

joblib.dump(scaler, SAVE_DIR / 'scaler.joblib')
print("Scaler saved at: ", SAVE_DIR / 'scaler.joblib')