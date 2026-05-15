import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Prepare the Data
data = {
    'Feature Set': ['Data-Driven', 'Data-Driven', 'Anchors Purely', 'Anchors Purely', 'Mixed', 'Mixed'],
    'Model': ['SVM', 'RFC', 'SVM', 'RFC', 'SVM', 'RFC'],
    'Accuracy': [0.8694, 0.9393, 0.6553, 0.9082, 0.8310, 0.9155],
    'Class 0 Recall (Negatives)': [0.76, 0.91, 0.22, 0.85, 0.63, 0.87],
    'Class 1 Recall (Positives)': [0.95, 0.96, 0.98, 0.95, 0.98, 0.95]
}

df = pd.DataFrame(data)

# Confusion Matrices Data
cm_data = {
    'Data-Driven': {
        'SVM': np.array([[627, 197], [55, 1050]]),
        'RFC': np.array([[749, 75], [42, 1063]])
    },
    'Anchors Purely': {
        'SVM': np.array([[180, 644], [21, 1084]]),
        'RFC': np.array([[698, 126], [51, 1054]])
    },
    'Mixed': {
        'SVM': np.array([[523, 301], [25, 1080]]),
        'RFC': np.array([[713, 111], [52, 1053]])
    }
}

# Set Visual Style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# --- FIGURE 1: Overall Performance Comparison ---
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Accuracy
sns.barplot(data=df, x='Feature Set', y='Accuracy', hue='Model', ax=ax1, palette='viridis')
ax1.set_title('Model Accuracy Comparison across Feature Sets', fontsize=16, fontweight='bold')
ax1.set_ylim(0, 1.0)
ax1.legend(title='Model Type', loc='upper left')

# Add text labels on bars
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.2f}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', 
                 xytext = (0, 9), 
                 textcoords = 'offset points')

plt.tight_layout()
plt.show()

# --- FIGURE 2: Recall for Negatives (The "Struggle" Metric) ---
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Feature Set', y='Class 0 Recall (Negatives)', hue='Model', palette='magma')
plt.title('Class 0 Recall', fontsize=16, fontweight='bold')
plt.ylabel('Recall (Specificity)')
plt.ylim(0, 1.0)
plt.axhline(0.5, ls='--', color='red', alpha=0.5, label='Random Guess Threshold')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# --- FIGURE 3: Confusion Matrix Grid ---
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
fig.suptitle('Confusion Matrix Comparison', fontsize=20, fontweight='bold', y=0.95)

feature_sets = ['Data-Driven', 'Anchors Purely', 'Mixed']
models = ['SVM', 'RFC']

for i, f_set in enumerate(feature_sets):
    for j, model in enumerate(models):
        sns.heatmap(cm_data[f_set][model], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i, j])
        axes[i, j].set_title(f'{f_set} - {model}')
        axes[i, j].set_xlabel('Predicted Label')
        axes[i, j].set_ylabel('True Label')
        if i == 0 and j == 0: # Legend/Helpful tip for the first plot
             axes[i, j].set_xticklabels(['False', 'True'])
             axes[i, j].set_yticklabels(['False', 'True'])

plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()