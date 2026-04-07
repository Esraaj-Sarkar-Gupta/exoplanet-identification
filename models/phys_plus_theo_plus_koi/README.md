## A combination of physical anchors, derived parameters and native KOI features

features = [
    "phys_duration_residual",
    "phys_depth_residual",
    "phys_impact_parameter_squared",
    "koi_duration",
    "theo_radius_ratio",
    "koi_insol",
    "koi_teq",
]

### SVM
Accuracy: 0.8600

Classification Report:
              precision    recall  f1-score   support

         0.0       0.91      0.74      0.82       824
         1.0       0.83      0.95      0.89      1105

    accuracy                           0.86      1929
   macro avg       0.87      0.85      0.85      1929
weighted avg       0.87      0.86      0.86      1929

Confusion Matrix:
[[ 613  211]
 [  59 1046]]
PR-AUC Score: 0.8681

### RFC
rf_model = RandomForestClassifier(
    n_estimators=1000,  # Number of trees in the forest
    n_jobs=-1,          # Uses all available CPU cores for speed
    random_state=24     # My birthday is the 24th of February!
)

Accuracy: 0.9321

Classification Report:
              precision    recall  f1-score   support

         0.0       0.94      0.90      0.92       824
         1.0       0.92      0.96      0.94      1105

    accuracy                           0.93      1929
   macro avg       0.93      0.93      0.93      1929
weighted avg       0.93      0.93      0.93      1929

Confusion Matrix:
[[ 738   86]
 [  45 1060]]
PR-AUC Score: 0.9836