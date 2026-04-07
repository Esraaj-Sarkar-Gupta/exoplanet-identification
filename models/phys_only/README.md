## These models were trained on only the 3 physics-aware engineered models.
`phys_duration_residual`
`phys_depth_residual`
`phys_impact_parameter_squared`

features = [
        "phys_duration_residual",
        "phys_depth_residual",
        "phys_impact_parameter_squared",
    ]

### SVM Results
We see that while precision is very high (0.98), recall is very low (0.22).

Accuracy: 0.6553

Classification Report:
              precision    recall  f1-score   support

         0.0       0.90      0.22      0.35       824
         1.0       0.63      0.98      0.77      1105

    accuracy                           0.66      1929
   macro avg       0.76      0.60      0.56      1929
weighted avg       0.74      0.66      0.59      1929

Confusion Matrix:
[[ 180  644]
 [  21 1084]]


This model is very good at finding true positives but struggles to identify false positives.
This could be attributed to the fact that all the features use can easy distinguish true
positives but do not have distinct indication for negatives. (See graphs in src/physics_aware_features.ipynb
for more context on this).

TODO:
	> Observe results from the random forest classifier.
	> Attempt to add features that can explain negatives.

### Random Forest Classifier Results

#### Run 1
rf_model = RandomForestClassifier(
    n_estimators=100,   # Number of trees in the forest
    n_jobs=-1,          # Uses all available CPU cores for speed
    random_state=24     # My birthday is the 24th of February!
)

Accuracy: 0.9067

Classification Report:
              precision    recall  f1-score   support

         0.0       0.93      0.85      0.89       824
         1.0       0.89      0.95      0.92      1105

    accuracy                           0.91      1929
   macro avg       0.91      0.90      0.90      1929
weighted avg       0.91      0.91      0.91      1929

Confusion Matrix:
[[ 697  127]
 [  53 1052]]

#### Run 2

rf_model = RandomForestClassifier(
    n_estimators=1000,   # Number of trees in the forest
    n_jobs=-1,          # Uses all available CPU cores for speed
    random_state=24     # My birthday is the 24th of February!
)

Accuracy: 0.9082

Classification Report:
              precision    recall  f1-score   support

         0.0       0.93      0.85      0.89       824
         1.0       0.89      0.95      0.92      1105

    accuracy                           0.91      1929
   macro avg       0.91      0.90      0.91      1929
weighted avg       0.91      0.91      0.91      1929

Confusion Matrix:
[[ 698  126]
 [  51 1054]]