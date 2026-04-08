## Performance: Physical Anchors, Derived Parameters, and Native KOI

### Selected Features

$$
\begin{array}{l}
\hline
\textbf{Feature Name} \\
\hline
\text{phys\_duration\_residual} \\
\text{phys\_depth\_residual} \\
\text{phys\_impact\_parameter\_squared} \\
\text{koi\_duration} \\
\text{theo\_radius\_ratio} \\
\text{koi\_insol} \\
\text{koi\_teq} \\
\hline
\end{array}
$$

---

### SVM Results

**Accuracy:** 0.8600  
**PR-AUC Score:** 0.8681  

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.91 & 0.74 & 0.82 & 824 \\
1.0 & 0.83 & 0.95 & 0.89 & 1105 \\
\hline
\text{Accuracy} & & & 0.86 & 1929 \\
\text{Macro Avg} & 0.87 & 0.85 & 0.85 & 1929 \\
\text{Weighted Avg} & 0.87 & 0.86 & 0.86 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
613 & 211 \\
59 & 1046
\end{bmatrix}
$$

---

### Random Forest Classifier Results

**Model Parameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_estimators` | 1000 | Number of trees in the forest |
| `n_jobs` | -1 | Uses all available CPU cores for speed |
| `random_state` | 24 | My birthday is the 24th of February! |

**Accuracy:** 0.9321  
**PR-AUC Score:** 0.9836  

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.94 & 0.90 & 0.92 & 824 \\
1.0 & 0.92 & 0.96 & 0.94 & 1105 \\
\hline
\text{Accuracy} & & & 0.93 & 1929 \\
\text{Macro Avg} & 0.93 & 0.93 & 0.93 & 1929 \\
\text{Weighted Avg} & 0.93 & 0.93 & 0.93 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
738 & 86 \\
45 & 1060
\end{bmatrix}
$$
