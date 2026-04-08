## Best Model (So Far): Physics-Informed RFC

### Selected Features

$$
\begin{array}{l}
\hline
\textbf{Feature Name} \\
\hline
\text{koi\_period} \\
\text{koi\_model\_snr} \\
\text{koi\_prad} \\
\text{koi\_teq} \\
\text{koi\_insol} \\
\text{koi\_steff} \\
\text{koi\_srad} \\
\text{phys\_depth\_residual} \\
\text{phys\_duration\_residual} \\
\text{phys\_impact\_parameter\_squared} \\
\hline
\end{array}
$$

### Model Parameters

| Parameter | Value |
| :--- | :--- |
| `n_estimators` | 1000 |
| `n_jobs` | -1 |
| `random_state` | 24 |

### RFC Results

**Accuracy:** 0.9440  
**PR-AUC Score:** 0.990355  

**Classification Report:**

$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.95 & 0.92 & 0.93 & 824 \\
1.0 & 0.94 & 0.97 & 0.95 & 1105 \\
\hline
\text{Accuracy} & & & 0.94 & 1929 \\
\text{Macro Avg} & 0.95 & 0.94 & 0.94 & 1929 \\
\text{Weighted Avg} & 0.94 & 0.94 & 0.94 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
754 & 70 \\
38 & 1067
\end{bmatrix}
$$

---

## Ablation Study: Purely using KOI Features (RFC)

### Selected Features

$$
\begin{array}{l}
\hline
\textbf{Feature Name} \\
\hline
\text{koi\_period} \\
\text{koi\_duration} \\
\text{koi\_depth} \\
\text{koi\_model\_snr} \\
\text{koi\_prad} \\
\text{koi\_teq} \\
\text{koi\_insol} \\
\text{koi\_steff} \\
\text{koi\_slogg} \\
\text{koi\_srad} \\
\hline
\end{array}
$$

### RFC Results

**Accuracy:** 0.9393  
**PR-AUC Score:** 0.988651  

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.95 & 0.91 & 0.93 & 824 \\
1.0 & 0.93 & 0.96 & 0.95 & 1105 \\
\hline
\text{Accuracy} & & & 0.94 & 1929 \\
\text{Macro Avg} & 0.94 & 0.94 & 0.94 & 1929 \\
\text{Weighted Avg} & 0.94 & 0.94 & 0.94 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
749 & 75 \\
42 & 1063
\end{bmatrix}
$$
