## Performance with Physics-Aware Engineered Features Only

These models were trained on only the 3 physics-aware engineered features.

### Selected Features

$$
\begin{array}{l}
\hline
\textbf{Feature Name} \\
\hline
\text{phys\_duration\_residual} \\
\text{phys\_depth\_residual} \\
\text{phys\_impact\_parameter\_squared} \\
\hline
\end{array}
$$

---

### SVM Results

We see that while precision is very high (0.98), recall is very low (0.22).

**Accuracy:** 0.6553

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.90 & 0.22 & 0.35 & 824 \\
1.0 & 0.63 & 0.98 & 0.77 & 1105 \\
\hline
\text{Accuracy} & & & 0.66 & 1929 \\
\text{Macro Avg} & 0.76 & 0.60 & 0.56 & 1929 \\
\text{Weighted Avg} & 0.74 & 0.66 & 0.59 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
180 & 644 \\
21 & 1084
\end{bmatrix}
$$

This model is very good at finding true positives but struggles to identify false positives. This could be attributed to the fact that all the features used can easily distinguish true positives but do not have distinct indication for negatives. (See graphs in `src/physics_aware_features.ipynb` for more context on this).

**TODO:**
* Observe results from the random forest classifier.
* Attempt to add features that can explain negatives.

---

### Random Forest Classifier Results

#### Run 1

**Model Parameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_estimators` | 100 | Number of trees in the forest |
| `n_jobs` | -1 | Uses all available CPU cores for speed |
| `random_state` | 24 | My birthday is the 24th of February! |

**Accuracy:** 0.9067

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.93 & 0.85 & 0.89 & 824 \\
1.0 & 0.89 & 0.95 & 0.92 & 1105 \\
\hline
\text{Accuracy} & & & 0.91 & 1929 \\
\text{Macro Avg} & 0.91 & 0.90 & 0.90 & 1929 \\
\text{Weighted Avg} & 0.91 & 0.91 & 0.91 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
697 & 127 \\
53 & 1052
\end{bmatrix}
$$

---

#### Run 2

**Model Parameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_estimators` | 1000 | Number of trees in the forest |
| `n_jobs` | -1 | Uses all available CPU cores for speed |
| `random_state` | 24 | My birthday is the 24th of February! |

**Accuracy:** 0.9082

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.93 & 0.85 & 0.89 & 824 \\
1.0 & 0.89 & 0.95 & 0.92 & 1105 \\
\hline
\text{Accuracy} & & & 0.91 & 1929 \\
\text{Macro Avg} & 0.91 & 0.90 & 0.91 & 1929 \\
\text{Weighted Avg} & 0.91 & 0.91 & 0.91 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
698 & 126 \\
51 & 1054
\end{bmatrix}
$$
