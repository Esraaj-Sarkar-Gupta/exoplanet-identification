import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Synthesized Data: Classical Best vs. TabNet
data = {
    'Dataset': ['KOI', 'KOI', 'KOI', 'KOI', 'TESS', 'TESS', 'TESS', 'TESS', 'K2', 'K2', 'K2', 'K2'],
    'Feature Set': ['Ablation', 'Ablation', 'Features 4', 'Features 4'] * 3,
    'Model Family': ['Classical ML', 'TabNet (DL)', 'Classical ML', 'TabNet (DL)'] * 3,
    'PR_AUC': [0.9905, 0.9430, 0.9918, 0.9178, 0.5543, 0.4824, 0.5591, 0.4749, 0.9375, 0.7710, 0.9046, 0.7518],
    'AUC_Std': [0.0019, 0.0065, 0.0023, 0.0199, 0.0493, 0.0423, 0.0404, 0.0300, 0.0096, 0.0329, 0.0184, 0.0212]
}

df = pd.DataFrame(data)

# 2. Presentation Styling Configurations
sns.set_theme(style="whitegrid", font_scale=1.1)
txt_color = "#222222" 

# Aesthetic: Cyan (Classical ML) vs Purple (TabNet)
colors_presentation = {
    "Classical ML": "#009999",  # Rich Deep Turquoise 
    "TabNet (DL)": "#8E44AD"    # Royal Purple
}
alpha_val = 0.85

def plot_comparative_panels(df):
    # Set up a 1x3 grid for the 3 datasets
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    datasets = ['KOI', 'TESS', 'K2']
    feature_map = {'Ablation': 0, 'Features 4': 1}
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset_data = df[df['Dataset'] == dataset]
        
        for strategy in subset_data['Model Family'].unique():
            strat_subset = subset_data[subset_data['Model Family'] == strategy]
            
            # Plot large points and bold error bars
            ax.errorbar(x=strat_subset['Feature Set'], 
                         y=strat_subset['PR_AUC'], 
                         yerr=strat_subset['AUC_Std'], 
                         fmt='o', markersize=18, capsize=12, 
                         label=strategy, color=colors_presentation[strategy], 
                         alpha=alpha_val, elinewidth=5, markeredgewidth=2.5,
                         markeredgecolor='white')

            # Apply large text labels
            for i, row in strat_subset.iterrows():
                x_pos = feature_map[row['Feature Set']]
                # Offset text: Classical to the left, DL to the right
                offset = -0.15 if strategy == "Classical ML" else 0.15
                ha = 'right' if strategy == "Classical ML" else 'left'
                
                ax.text(x=x_pos + offset, y=row['PR_AUC'], 
                         s=f"{row['PR_AUC']:.4f}", 
                         color=colors_presentation[strategy], 
                         fontweight='black', fontsize=14, va='center', ha=ha)

        ax.set_title(f"{dataset} Performance", fontsize=18, fontweight='bold', color=txt_color, pad=15)
        ax.set_xlim(-0.6, 1.6)
        
        # Format axes
        if idx == 0:
            ax.set_ylabel('Mean PR-AUC', fontsize=15, fontweight='bold', color=txt_color)
        else:
            ax.set_ylabel('')
            
        ax.set_xlabel('Feature Set', fontsize=14, fontweight='bold', color=txt_color)
        
        # Custom Y-limits to frame the dramatic drops accurately
        if dataset == "KOI": ax.set_ylim(0.88, 1.01)
        if dataset == "TESS": ax.set_ylim(0.40, 0.65)
        if dataset == "K2": ax.set_ylim(0.68, 0.98)
            
        if idx == 1: # Put legend in the middle plot
            leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                            ncol=2, fontsize=14, frameon=True, shadow=True, facecolor='white')
            leg.get_frame().set_edgecolor('black')
            leg.get_frame().set_linewidth(1.5)
        else:
            ax.get_legend().remove() if ax.get_legend() else None

    plt.suptitle("Classical Ensembles vs. Deep Learning (TabNet) on Tabular Data", 
                 fontsize=22, fontweight='bold', color=txt_color, y=1.05)
    plt.tight_layout()
    plt.show()

plot_comparative_panels(df)