## Best Model (So Far): Physics-Informed RFC

### Selected Features
```python=
features = [
    "koi_period",
    "koi_model_snr",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_srad",
    "phys_depth_residual",
    "phys_duration_residual",
    "phys_impact_parameter_squared"
]
```

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

```python
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
```

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
