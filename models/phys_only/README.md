## These models were trained on only the 3 physics-aware engineered models.
`phys_duration_residual`
`phys_depth_residual`
`phys_impact_parameter_squared`

### SVM Results
We see that while precision is very high (0.98), recall is very low (0.22).

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
