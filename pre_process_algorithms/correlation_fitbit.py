import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

"""AHORA DEBO VER SI HAY CORRELACIÓN ENTRE LAS VAIABLES Y LOS OBJETIVOS """

# Configuration
csv_folder = "/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit"  # Change this to your folder path
output_folder = "/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit/correlationResults"  # Folder to save results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob


# Create output folder if it doesn't exist
Path(output_folder).mkdir(exist_ok=True)

# Read and combine all CSV files
def load_all_csvs(folder_path):
    csv_files = glob.glob(f"{folder_path}/*_consolidated_daily_summary.csv")
    print(f"Found {len(csv_files)} CSV files")
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add user identifier from filename
            user_id = Path(file).stem
            df['user_id'] = user_id
            dfs.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    print(f"Total users: {combined_df['user_id'].nunique()}")
    
    return combined_df

# Load data
df = load_all_csvs(csv_folder)

# Display basic information
print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)
print(df.info())
print("\n")
print(df.describe())

# Columns to exclude from analysis
excluded_cols = [
    'sleep_global_levels',
    'sleep_global_logType', 
    'sleep_global_logId',
    'sleep_global_dateOfSleep',
    'sleep_global_endTime',
    'sleep_global_startTime',
    'user_id'  # Keep as identifier but don't analyze
]

# Identify categorical columns (excluding the ones we want to ignore)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col not in excluded_cols]

print(f"\nCategorical columns to encode: {categorical_cols}")
print(f"Excluded columns: {[col for col in excluded_cols if col in df.columns]}")

# Create a copy of dataframe for encoding
df_encoded = df.copy()

# Encode categorical variables
encoding_info = {}
for col in categorical_cols:
    unique_values = df_encoded[col].dropna().unique()
    print(f"\n'{col}' has {len(unique_values)} unique values: {unique_values[:10]}")  # Show first 10
    
    # If few unique values, use label encoding
    if len(unique_values) <= 20:
        # Create mapping
        value_mapping = {val: idx for idx, val in enumerate(unique_values)}
        df_encoded[f'{col}_encoded'] = df_encoded[col].map(value_mapping)
        encoding_info[col] = value_mapping
        print(f"  -> Encoded as '{col}_encoded'")
    else:
        print(f"  -> Too many unique values, skipping encoding")

# Save encoding information
if encoding_info:
    encoding_df = pd.DataFrame([
        {'Column': col, 'Original_Value': orig, 'Encoded_Value': enc}
        for col, mapping in encoding_info.items()
        for orig, enc in mapping.items()
    ])
    encoding_df.to_csv(f"{output_folder}/categorical_encoding.csv", index=False)
    print(f"\nEncoding information saved to {output_folder}/categorical_encoding.csv")

# Select numeric columns (including encoded ones)
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
# Remove user_id if it was somehow converted to numeric
numeric_cols = [col for col in numeric_cols if col != 'user_id']
print(f"\nTotal numeric columns for correlation (including encoded): {len(numeric_cols)}")

# Calculate correlation matrix
correlation_matrix = df_encoded[numeric_cols].corr()

# Save correlation matrix to CSV
correlation_matrix.to_csv(f"{output_folder}/correlation_matrix.csv")
print(f"\nCorrelation matrix saved to {output_folder}/correlation_matrix.csv")

# Create correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, 
            annot=False,  # No numbers displayed
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - All Variables', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
print(f"Correlation heatmap saved to {output_folder}/correlation_heatmap.png")
plt.close()

# Find strong correlations (absolute value > 0.7, excluding diagonal)
strong_correlations = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.7:
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
    
    print("\n" + "="*50)
    print("STRONG CORRELATIONS (|r| > 0.7)")
    print("="*50)
    print(strong_corr_df.to_string(index=False))
else:
    print("\nNo strong correlations (|r| > 0.7) found")

# Create pairplot for top correlated variables (if any strong correlations exist)
if not strong_corr_df.empty and len(strong_corr_df) > 0:
    top_vars = set()
    for _, row in strong_corr_df.head(6).iterrows():
        top_vars.add(row['Variable 1'])
        top_vars.add(row['Variable 2'])
    
    top_vars = list(top_vars)[:6]  # Limit to 6 variables for readability
    
    if len(top_vars) > 1:
        print(f"\nCreating pairplot for top correlated variables: {top_vars}")
        pairplot_fig = sns.pairplot(df_encoded[top_vars].dropna(), 
                                     diag_kind='kde',
                                     plot_kws={'alpha': 0.6})
        pairplot_fig.fig.suptitle('Pairplot of Top Correlated Variables', 
                                   y=1.01, 
                                   fontsize=16)
        plt.savefig(f"{output_folder}/pairplot_top_correlations.png", 
                    dpi=300, 
                    bbox_inches='tight')
        print(f"Pairplot saved to {output_folder}/pairplot_top_correlations.png")
        plt.close()


print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"All results saved to '{output_folder}' folder")
print("\nGenerated files:")
print("1. correlation_matrix.csv - Full correlation matrix")
print("2. correlation_heatmap.png - Visual heatmap")
print("3. strong_correlations.csv - Pairs with |r| > 0.7")
print("4. pairplot_top_correlations.png - Scatter plots of top variables")
print("5. user_summary_stats.csv - Average values per user")