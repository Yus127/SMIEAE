import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

input_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data.csv"  # Output from previous script
output_folder = "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/correlation_results_combined"

# Create output folder if it doesn't exist
Path(output_folder).mkdir(exist_ok=True)

# Load the combined data
print("LOADING COMBINED DATA")
try:
    df = pd.read_csv(input_file)
    print(f"Loaded: {input_file}")
    print(f"Shape: {df.shape}")
    print(f"Users: {df['userid'].nunique()}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Display basic information
print("\n" + "="*60)
print("DATASET OVERVIEW")
print(df.info())
print("\n")
print(df.describe())

# Select only numeric columns for correlation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove userid as it's just an identifier
if 'userid' in numeric_cols:
    numeric_cols.remove('userid')

print(f"\nNumeric columns for correlation: {len(numeric_cols)}")
print(f"Columns: {numeric_cols}")

# Calculate correlation matrix
print("\n" + "="*60)
print("CALCULATING CORRELATION MATRIX")
correlation_matrix = df[numeric_cols].corr()

# Save correlation matrix to CSV
correlation_matrix.to_csv(f"{output_folder}/correlation_matrix.csv")
print(f" Correlation matrix saved to {output_folder}/correlation_matrix.csv")

# Create full correlation heatmap (without numbers)
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, 
            annot=False,
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - All Variables (Including Stress & Anxiety)', fontsize=18, pad=20)
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap_full.png", dpi=300, bbox_inches='tight')
print(f" Full correlation heatmap saved")
plt.close()

# Find strong correlations (|r| > 0.5)
print("\n" + "="*60)
print("IDENTIFYING STRONG CORRELATIONS")
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.5:
            strong_correlations.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlation': corr_value
            })

# Sort by absolute correlation value
strong_corr_df = pd.DataFrame(strong_correlations)
if not strong_corr_df.empty:
    strong_corr_df = strong_corr_df.sort_values('Correlation', 
                                                  key=abs, 
                                                  ascending=False)
    strong_corr_df.to_csv(f"{output_folder}/strong_correlations.csv", index=False)
    
    print(f"Found {len(strong_corr_df)} strong correlations (|r| > 0.5)")
    print("\nTop 20 strongest correlations:")
    print(strong_corr_df.head(20).to_string(index=False))
else:
    print("No strong correlations (|r| > 0.5) found")

# Focus on stress and anxiety correlations
print("\n" + "="*60)
print("STRESS & ANXIETY CORRELATIONS")

stress_anxiety_vars = ['stress_level', 'anxiety_level']
available_psych_vars = [var for var in stress_anxiety_vars if var in numeric_cols]

if available_psych_vars:
    for psych_var in available_psych_vars:
        print(f"\n{psych_var.upper()} correlations:")
        print("-" * 50)
        
        # Get correlations with this variable
        var_correlations = correlation_matrix[psych_var].sort_values(ascending=False)
        
        # Remove self-correlation
        var_correlations = var_correlations[var_correlations.index != psych_var]
        
        # Show top positive correlations
        print(f"\nTop 10 POSITIVE correlations with {psych_var}:")
        print(var_correlations.head(10).to_string())
        
        # Show top negative correlations
        print(f"\nTop 10 NEGATIVE correlations with {psych_var}:")
        print(var_correlations.tail(10).to_string())
        
        # Save to CSV
        var_correlations.to_csv(f"{output_folder}/{psych_var}_correlations.csv")
    
    # Create focused heatmap for stress/anxiety and top correlated variables
    if len(available_psych_vars) >= 1:
        top_vars_per_psych = 15
        all_top_vars = set(available_psych_vars)
        
        for psych_var in available_psych_vars:
            var_corr = correlation_matrix[psych_var].abs().sort_values(ascending=False)
            top_for_var = var_corr.head(top_vars_per_psych + 1).index.tolist()
            all_top_vars.update(top_for_var)
        
        # Remove duplicates and limit
        focused_vars = list(all_top_vars)[:30]  # Limit to 30 variables for readability
        
        if len(focused_vars) > 2:
            focused_corr = correlation_matrix.loc[focused_vars, focused_vars]
            
            plt.figure(figsize=(16, 14))
            sns.heatmap(focused_corr, 
                        annot=False,
                        cmap='coolwarm', 
                        center=0,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": 0.8})
            plt.title('Focused Correlation: Stress/Anxiety & Top Related Variables', 
                     fontsize=16, pad=20)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/correlation_heatmap_stress_anxiety_focused.png", 
                       dpi=300, bbox_inches='tight')
            print(f"\n Focused stress/anxiety heatmap saved")
            plt.close()
else:
    print("Stress and/or anxiety variables not found in dataset")

# Create scatter plots for top stress/anxiety correlations
if available_psych_vars and not strong_corr_df.empty:
    print("\n" + "="*60)
    print("CREATING SCATTER PLOTS")
    
    for psych_var in available_psych_vars:
        # Get top 6 correlations (positive and negative)
        var_corr = correlation_matrix[psych_var].sort_values(key=abs, ascending=False)
        var_corr = var_corr[var_corr.index != psych_var]
        top_6 = var_corr.head(6).index.tolist()
        
        if len(top_6) > 0:
            # Create subplot grid
            n_plots = len(top_6)
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_plots > 1 else [axes]
            
            for idx, var in enumerate(top_6):
                ax = axes[idx]
                
                # Remove NaN values for plotting
                plot_data = df[[psych_var, var]].dropna()
                
                if len(plot_data) > 0:
                    ax.scatter(plot_data[var], plot_data[psych_var], alpha=0.5)
                    ax.set_xlabel(var, fontsize=10)
                    ax.set_ylabel(psych_var, fontsize=10)
                    
                    corr_val = correlation_matrix.loc[psych_var, var]
                    ax.set_title(f'r = {corr_val:.3f}', fontsize=12)
                    
                    # Add trend line
                    z = np.polyfit(plot_data[var], plot_data[psych_var], 1)
                    p = np.poly1d(z)
                    ax.plot(plot_data[var], p(plot_data[var]), "r--", alpha=0.8, linewidth=2)
                    
                    ax.grid(True, alpha=0.3)
            
            # Hide extra subplots
            for idx in range(len(top_6), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Top Correlations with {psych_var}', fontsize=16, y=1.00)
            plt.tight_layout()
            plt.savefig(f"{output_folder}/scatter_plots_{psych_var}.png", 
                       dpi=300, bbox_inches='tight')
            print(f" Scatter plots for {psych_var} saved")
            plt.close()

# Summary statistics by user
print("\n" + "="*60)
print("SUMMARY STATISTICS BY USER")
user_summary = df.groupby('userid')[numeric_cols].mean()
user_summary.to_csv(f"{output_folder}/user_summary_stats.csv")
print(f" User summary statistics saved")
print("\nFirst 5 users:")
print(user_summary.head())

# Check data availability for stress and anxiety
if available_psych_vars:
    print("\n" + "="*60)
    print("STRESS & ANXIETY DATA AVAILABILITY")
    for var in available_psych_vars:
        total = len(df)
        available = df[var].notna().sum()
        print(f"{var}: {available}/{total} records ({available/total*100:.1f}%)")
        
        # Per user
        user_coverage = df.groupby('userid')[var].apply(lambda x: x.notna().sum())
        print(f"  Average per user: {user_coverage.mean():.1f} records")
        print(f"  Range: {user_coverage.min()}-{user_coverage.max()} records")

print("\n" + "="*60)
print(f"All results saved to '{output_folder}' folder")
print("\nGenerated files:")
print("1. correlation_matrix.csv - Full correlation matrix")
print("2. correlation_heatmap_full.png - Full heatmap visualization")
print("3. strong_correlations.csv - Pairs with |r| > 0.5")
print("4. stress_level_correlations.csv - All stress correlations")
print("5. anxiety_level_correlations.csv - All anxiety correlations")
print("6. correlation_heatmap_stress_anxiety_focused.png - Focused heatmap")
print("7. scatter_plots_stress_level.png - Scatter plots for stress")
print("8. scatter_plots_anxiety_level.png - Scatter plots for anxiety")
print("9. user_summary_stats.csv - Average values per user")