import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Data Setup
data = {
    'Dataset': ['KOI', 'TESS', 'K2', 'KOI', 'TESS', 'K2'],
    'Feature Set': ['Ablation', 'Ablation', 'Ablation', 'Mixed', 'Mixed', 'Mixed'],
    'Accuracy': [0.9467, 0.8481, 0.7948, 0.9550, 0.8477, 0.7948],
    'Acc_Std': [0.0074, 0.0082, 0.0239, 0.0066, 0.0107, 0.0275],
    'PR_AUC': [0.9905, 0.5729, 0.9022, 0.9913, 0.5774, 0.8957],
    'AUC_Std': [0.0019, 0.0405, 0.0149, 0.0027, 0.0491, 0.0118]
}

df = pd.DataFrame(data)

# 2. Styling Configurations
sns.set_theme(style="whitegrid", font="sans-serif")
colors = {"Ablation": "#34495e", "Mixed": "#ff5733"} 
alpha_val = 0.75

# --- FIGURE 1: Zoomed PR-AUC Point Plot with Values ---
plt.figure(figsize=(7, 5)) # Smaller size for 'zoomed' effect
ax1 = plt.gca()

# Mapping for text placement
dataset_indices = {name: i for i, name in enumerate(df['Dataset'].unique())}

for feature in ['Ablation', 'Mixed']:
    subset = df[df['Feature Set'] == feature]
    
    # Plot points and error bars
    plt.errorbar(x=subset['Dataset'], 
                 y=subset['PR_AUC'], 
                 yerr=subset['AUC_Std'], 
                 fmt='o', markersize=15, capsize=6, 
                 label=feature, color=colors[feature], 
                 alpha=alpha_val, elinewidth=3)

    # Add large, color-coded text labels
    for i, row in subset.iterrows():
        # Offset Ablation to the left, Mixed to the right
        x_pos = dataset_indices[row['Dataset']]
        offset = -0.18 if feature == 'Ablation' else 0.18
        ha = 'right' if feature == 'Ablation' else 'left'
        
        plt.text(x=x_pos + offset, 
                 y=row['PR_AUC'], 
                 s=f"{row['PR_AUC']:.4f}", 
                 color=colors[feature], 
                 fontweight='bold', 
                 fontsize=12, 
                 va='center', 
                 ha=ha)

plt.title('PR-AUC Metric Stability', fontsize=14, fontweight='bold', pad=15)
plt.xlim(-0.6, 2.6) # Tighter x-limits to bring points closer
plt.ylabel('Mean PR-AUC', fontsize=11)
plt.legend(loc='lower left', fontsize=10)
plt.tight_layout()
plt.show()

# --- FIGURE 2: Compact Slope Graph for Accuracy ---
plt.figure(figsize=(6, 5)) # Compact square-ish format
ax2 = plt.gca()
line_colors = {'KOI': '#2ecc71', 'TESS': '#f1c40f', 'K2': '#e74c3c'}

for dataset in ['KOI', 'TESS', 'K2']:
    subset = df[df['Dataset'] == dataset]
    
    plt.plot(subset['Feature Set'], subset['Accuracy'], 
             marker='o', markersize=10, linewidth=4, 
             label=dataset, color=line_colors[dataset])
    
    # Annotate values (centered above points)
    for x, y in zip(subset['Feature Set'], subset['Accuracy']):
        plt.text(x, y + 0.006, f'{y:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.title('Accuracy Shift', fontsize=14, fontweight='bold', pad=12)
plt.ylim(0.75, 1.0)
plt.ylabel('Mean Accuracy', fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=9)
plt.tight_layout()
plt.show()