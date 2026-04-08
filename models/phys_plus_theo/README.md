## Performance with Extended Physics/Theoretical Features

Added more features to combat the low accuracy from just the physically engineered features.

### Selected Features

$$
\begin{array}{l}
\hline
\textbf{Feature Name} \\
\hline
\text{phys\_duration\_residual} \\
\text{phys\_depth\_residual} \\
\text{phys\_impact\_parameter\_squared} \\
\text{theo\_duration} \\
\text{theo\_radius\_ratio} \\
\hline
\end{array}
$$

---

### SVM Results

**Accuracy:** 0.8310

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.95 & 0.63 & 0.76 & 824 \\
1.0 & 0.78 & 0.98 & 0.87 & 1105 \\
\hline
\text{Accuracy} & & & 0.83 & 1929 \\
\text{Macro Avg} & 0.87 & 0.81 & 0.82 & 1929 \\
\text{Weighted Avg} & 0.86 & 0.83 & 0.82 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
523 & 301 \\
25 & 1080
\end{bmatrix}
$$

---

### Random Forest Results

#### Run 1

**Model Parameters**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `n_estimators` | 100 | Number of trees in the forest |
| `n_jobs` | -1 | Uses all available CPU cores for speed |
| `random_state` | 24 | My birthday is the 24th of February! |

**Accuracy:** 0.9150

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.93 & 0.86 & 0.90 & 824 \\
1.0 & 0.90 & 0.95 & 0.93 & 1105 \\
\hline
\text{Accuracy} & & & 0.91 & 1929 \\
\text{Macro Avg} & 0.92 & 0.91 & 0.91 & 1929 \\
\text{Weighted Avg} & 0.92 & 0.91 & 0.91 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
712 & 112 \\
52 & 1053
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

**Accuracy:** 0.9155

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.93 & 0.87 & 0.90 & 824 \\
1.0 & 0.90 & 0.95 & 0.93 & 1105 \\
\hline
\text{Accuracy} & & & 0.92 & 1929 \\
\text{Macro Avg} & 0.92 & 0.91 & 0.91 & 1929 \\
\text{Weighted Avg} & 0.92 & 0.92 & 0.92 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
713 & 111 \\
52 & 1053
\end{bmatrix}
$$
