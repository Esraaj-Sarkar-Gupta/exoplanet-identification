import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Data Setup from Tables 2 and 3 (Consistent with previous analysis)
data_tess = {
    'Model Strategy': ['HistGB (Native)', 'HistGB (Native)', 'RF (Imputed)', 'RF (Imputed)'],
    'Feature Set': ['Ablation', 'Features 3', 'Ablation', 'Features 3'],
    'PR_AUC': [0.5543, 0.5569, 0.5445, 0.5431],
    'AUC_Std': [0.0493, 0.0496, 0.0471, 0.0457]
}

data_k2 = {
    'Model Strategy': ['HistGB (Native)', 'HistGB (Native)', 'RF (Imputed)', 'RF (Imputed)'],
    'Feature Set': ['Ablation', 'Features 3', 'Ablation', 'Features 3'],
    'PR_AUC': [0.9375, 0.9297, 0.9022, 0.8962],
    'AUC_Std': [0.0096, 0.0104, 0.0167, 0.0168]
}

df_tess = pd.DataFrame(data_tess)
df_k2 = pd.DataFrame(data_k2)

# 2. Styling Configurations (Presentation Focus)
# Using 'darkgrid' can make cyan/purple pop more, but 'whitegrid' remains best for printing/paper integration.
sns.set_theme(style="whitegrid", font_scale=1.2) 
txt_color = "#222222" # Very dark gray for soft, confident text

# CYAN & PURPLE PALETTE: High-contrast, modern screen look
# We need a rich cyan that remains legible on white for text.
colors_presentation = {
    "HistGB (Native)": "#8E44AD", # Royal Purple
    "RF (Imputed)": "#009999"   # Rich, readable Deep Turquoise (Bold Cyan)
}
alpha_val = 0.85 # Slightly higher alpha to maintain vibrancy in the bold palette

def plot_presentation_metric(df, title):
    # Figure setup optimized for single-slide integration (7:5 aspect ratio)
    plt.figure(figsize=(7, 5)) 
    ax = plt.gca()
    
    # Text annotation placement logic map
    feature_map = {'Ablation': 0, 'Features 3': 1}
    
    for strategy in df['Model Strategy'].unique():
        subset = df[df['Model Strategy'] == strategy]
        
        # Plot large, "juicy" points and ultra-bold error bars
        plt.errorbar(x=subset['Feature Set'], 
                     y=subset['PR_AUC'], 
                     yerr=subset['AUC_Std'], 
                     fmt='o', 
                     markersize=21, # Even larger for ultimate screen visibility
                     capsize=14, 
                     label=strategy, 
                     color=colors_presentation[strategy], 
                     alpha=alpha_val, 
                     elinewidth=6, # Extremely bold lines
                     markeredgewidth=3,
                     markeredgecolor='white') # Add a white ring to pop points off grid lines

        # Apply large, color-matched text labels
        for i, row in subset.iterrows():
            x_pos = feature_map[row['Feature Set']]
            
            # Offset Native to the left, Imputed to the right for legibility on overlap
            offset = -0.16 if "Native" in strategy else 0.16
            ha = 'right' if "Native" in strategy else 'left'
            
            plt.text(x=x_pos + offset, 
                     y=row['PR_AUC'], 
                     s=f"{row['PR_AUC']:.4f}", 
                     color=colors_presentation[strategy], 
                     fontweight='black', # Extra bold text
                     fontsize=16, # Increased font size for text values
                     va='center', 
                     ha=ha)

    # Title and Axes Styling (Presentation Mode)
    plt.title(title, fontsize=22, fontweight='bold', pad=25, color=txt_color)
    plt.ylabel('Mean PR-AUC', fontsize=17, fontweight='bold', color=txt_color)
    plt.xlabel('Feature Set', fontsize=17, fontweight='bold', color=txt_color)
    
    # Compress axes limits for visual punch
    plt.xlim(-0.6, 1.6) 
    
    # Adjust y-limit for K2 to show the dramatic gap without excessive white space
    if "K2" in title: plt.ylim(0.86, 0.97) 

    # Legend Styling
    leg = plt.legend(loc='best', fontsize=13, frameon=True, shadow=True, facecolor='white')
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show()

# Generate and display both modern presentation plots
plot_presentation_metric(df_tess, "TESS")
plot_presentation_metric(df_k2, "K2")