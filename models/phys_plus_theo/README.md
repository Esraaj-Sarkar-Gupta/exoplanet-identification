Added more features to combat the low accuracy from just the physically engineered features.

features = [
    "phys_duration_residual",
    "phys_depth_residual",
    "phys_impact_parameter_squared",
    "theo_duration",
    "theo_radius_ratio"
]

Accuracy: 0.8310

Classification Report:
              precision    recall  f1-score   support

         0.0       0.95      0.63      0.76       824
         1.0       0.78      0.98      0.87      1105

    accuracy                           0.83      1929
   macro avg       0.87      0.81      0.82      1929
weighted avg       0.86      0.83      0.82      1929

Confusion Matrix:
[[ 523  301]
 [  25 1080]]