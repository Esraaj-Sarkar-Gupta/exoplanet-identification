import os

file_name = "model_baseline_results.md"

# Using a raw string (r"") so LaTeX backslashes are handled correctly
markdown_content = r"""## Baseline Performance: Native KOI Parameters

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
\text{koi\_srad} \\
\hline
\end{array}
$$
*(Note: koi_slogg was commented out in the configuration and excluded from this list)*

---

### SVM Results

**Accuracy:** 0.8694  
**PR-AUC Score:** 0.9228  

**Classification Report:**
$$
\begin{array}{lcccc}
\hline
\text{Class} & \text{Precision} & \text{Recall} & \text{F1-Score} & \text{Support} \\
\hline
0.0 & 0.92 & 0.76 & 0.83 & 824 \\
1.0 & 0.84 & 0.95 & 0.89 & 1105 \\
\hline
\text{Accuracy} & & & 0.87 & 1929 \\
\text{Macro Avg} & 0.88 & 0.86 & 0.86 & 1929 \\
\text{Weighted Avg} & 0.88 & 0.87 & 0.87 & 1929 \\
\hline
\end{array}
$$

**Confusion Matrix:**
$$
\begin{bmatrix}
627 & 197 \\
55 & 1050
\end{bmatrix}
$$

---

### RFC Results

**Accuracy:** 0.9393  
**PR-AUC Score:** 0.9887  

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
"""

# Write the content to a Markdown file
with open(file_name, "w", encoding="utf-8") as file:
    file.write(markdown_content)

print(f"Successfully generated '{file_name}' in the current directory.")