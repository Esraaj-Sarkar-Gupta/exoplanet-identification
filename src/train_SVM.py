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

SAVE_DIR = Path("models/exp/SVM")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = SAVE_DIR / 'model.joblib'

# ---- Load Data ---- #
data_path = Path("data/koi/physics_train/physically_anchored_features_KOI.csv")
df = pd.read_csv(data_path)

features = [
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