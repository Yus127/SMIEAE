import pandas as pd
import numpy as np
import os

# Define paths
processed_dir = "/Users/YusMolina/Downloads/smieae/data/processed"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("ML-READY DATASET GENERATOR")
print("="*70)

# Load the aggregated files
print("\nStep 1: Loading aggregated data files...")

agg_30min_path = os.path.join(processed_dir, "fitbit_30min_window_aggregated.csv")
agg_1hour_path = os.path.join(processed_dir, "fitbit_1hour_window_aggregated.csv")

if not os.path.exists(agg_30min_path):
    print(f"ERROR: File not found: {agg_30min_path}")
    exit(1)

if not os.path.exists(agg_1hour_path):
    print(f"ERROR: File not found: {agg_1hour_path}")
    exit(1)

df_30min = pd.read_csv(agg_30min_path)
df_1hour = pd.read_csv(agg_1hour_path)

print(f"✓ Loaded 30-minute window data: {len(df_30min)} rows")
print(f"✓ Loaded 1-hour window data: {len(df_1hour)} rows")

# ==============================================================================
# PART 1: CREATE ML-READY DATASETS
# ==============================================================================

print("\n" + "="*70)
print("PART 1: CREATING ML-READY DATASETS")
print("="*70)

def create_ml_dataset(df, window_name):
    """
    Create a clean ML-ready dataset from aggregated data
    """
    print(f"\nProcessing {window_name} window...")
    
    # Identify feature columns (statistics) and target columns (questionnaire responses)
    feature_cols = []
    target_cols = []
    metadata_cols = ['userid', 'response_timestamp', 'window_start', 'window_end', 'record_count']
    
    for col in df.columns:
        if col in metadata_cols:
            continue
        elif col.startswith('q_'):
            target_cols.append(col)
        elif any(col.endswith(suffix) for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']):
            feature_cols.append(col)
    
    print(f"  Found {len(feature_cols)} feature columns")
    print(f"  Found {len(target_cols)} target columns")
    
    # Create ML dataset with metadata, features, and targets
    ml_cols = metadata_cols + feature_cols + target_cols
    ml_df = df[ml_cols].copy()
    
    # Calculate completeness for each feature
    feature_completeness = {}
    for col in feature_cols:
        non_null = ml_df[col].notna().sum()
        total = len(ml_df)
        completeness_pct = (non_null / total) * 100
        feature_completeness[col] = {
            'non_null_count': non_null,
            'null_count': total - non_null,
            'total_count': total,
            'completeness_pct': completeness_pct
        }
    
    return ml_df, feature_cols, target_cols, feature_completeness

# Create ML datasets for both windows
ml_30min, features_30min, targets_30min, completeness_30min = create_ml_dataset(df_30min, "30-minute")
ml_1hour, features_1hour, targets_1hour, completeness_1hour = create_ml_dataset(df_1hour, "1-hour")

# Save ML-ready datasets
output_30min = os.path.join(output_dir, "ml_ready_30min_window.csv")
output_1hour = os.path.join(output_dir, "ml_ready_1hour_window.csv")

ml_30min.to_csv(output_30min, index=False)
ml_1hour.to_csv(output_1hour, index=False)

print(f"\n✓ Saved 30-minute ML dataset: {output_30min}")
print(f"  Shape: {ml_30min.shape} (rows x columns)")
print(f"  Features: {len(features_30min)}")
print(f"  Targets: {len(targets_30min)}")

print(f"\n✓ Saved 1-hour ML dataset: {output_1hour}")
print(f"  Shape: {ml_1hour.shape} (rows x columns)")
print(f"  Features: {len(features_1hour)}")
print(f"  Targets: {len(targets_1hour)}")

# ==============================================================================
# PART 2: CREATE COMBINED DATASET (30min + 1hour features)
# ==============================================================================

print("\n" + "="*70)
print("PART 2: CREATING COMBINED DATASET")
print("="*70)

# Merge on userid and response_timestamp
print("\nMerging 30-minute and 1-hour windows...")

# Prepare dataframes for merging
merge_cols = ['userid', 'response_timestamp']

# Rename feature columns to indicate time window
df_30min_renamed = df_30min.copy()
df_1hour_renamed = df_1hour.copy()

# Keep only merge keys, features, and targets for merging
for col in df_30min_renamed.columns:
    if col not in merge_cols and not col.startswith('q_'):
        if any(col.endswith(suffix) for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']):
            # This is a feature - add window prefix
            new_name = f"w30_{col}"
            df_30min_renamed.rename(columns={col: new_name}, inplace=True)

for col in df_1hour_renamed.columns:
    if col not in merge_cols and not col.startswith('q_'):
        if any(col.endswith(suffix) for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']):
            # This is a feature - add window prefix
            new_name = f"w60_{col}"
            df_1hour_renamed.rename(columns={col: new_name}, inplace=True)

# Merge the datasets
combined_df = df_30min_renamed.merge(
    df_1hour_renamed,
    on=merge_cols,
    how='inner',
    suffixes=('_30min', '_1hour')
)

print(f"✓ Combined dataset shape: {combined_df.shape}")
print(f"  Responses in both windows: {len(combined_df)}")

# Identify combined feature columns
combined_features = [col for col in combined_df.columns 
                    if (col.startswith('w30_') or col.startswith('w60_'))]
combined_targets = [col for col in combined_df.columns if col.startswith('q_')]

# Clean up duplicate target columns
# Keep only one version of each target (they should be the same)
seen_targets = set()
cols_to_keep = merge_cols.copy()

for col in combined_df.columns:
    if col.startswith('q_'):
        base_col = col.replace('_30min', '').replace('_1hour', '')
        if base_col not in seen_targets:
            seen_targets.add(base_col)
            # Rename to remove suffix
            if col != base_col:
                combined_df.rename(columns={col: base_col}, inplace=True)
                cols_to_keep.append(base_col)
            else:
                cols_to_keep.append(col)
    else:
        cols_to_keep.append(col)

# Select final columns
final_cols = list(dict.fromkeys(cols_to_keep))  # Remove duplicates while preserving order
combined_df = combined_df[final_cols]

# Save combined dataset
output_combined = os.path.join(output_dir, "ml_ready_combined_windows.csv")
combined_df.to_csv(output_combined, index=False)

print(f"\n✓ Saved combined ML dataset: {output_combined}")
print(f"  Shape: {combined_df.shape}")
print(f"  Features from 30-min window: {len([c for c in combined_df.columns if c.startswith('w30_')])}")
print(f"  Features from 1-hour window: {len([c for c in combined_df.columns if c.startswith('w60_')])}")
print(f"  Total features: {len([c for c in combined_df.columns if c.startswith('w30_') or c.startswith('w60_')])}")
print(f"  Targets: {len([c for c in combined_df.columns if c.startswith('q_')])}")

# ==============================================================================
# PART 3: DATA COMPLETENESS ANALYSIS
# ==============================================================================

print("\n" + "="*70)
print("PART 3: DATA COMPLETENESS ANALYSIS")
print("="*70)

def create_completeness_report(df, feature_cols, window_name):
    """
    Create a detailed completeness report for features
    """
    print(f"\nAnalyzing {window_name} window features...")
    
    completeness_data = []
    
    for col in feature_cols:
        non_null = df[col].notna().sum()
        total = len(df)
        completeness_pct = (non_null / total) * 100
        
        # Extract base variable name (remove _mean, _std, etc. suffix)
        base_var = col
        stat_type = 'unknown'
        for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']:
            if col.endswith(suffix):
                base_var = col.replace(suffix, '')
                stat_type = suffix[1:]  # Remove leading underscore
                break
        
        completeness_data.append({
            'window': window_name,
            'feature_name': col,
            'base_variable': base_var,
            'statistic_type': stat_type,
            'total_responses': total,
            'non_null_count': non_null,
            'null_count': total - non_null,
            'completeness_pct': completeness_pct,
            'missing_pct': 100 - completeness_pct
        })
    
    return pd.DataFrame(completeness_data)

# Create completeness reports
completeness_30min_df = create_completeness_report(ml_30min, features_30min, "30-minute")
completeness_1hour_df = create_completeness_report(ml_1hour, features_1hour, "1-hour")

# Combine completeness reports
all_completeness = pd.concat([completeness_30min_df, completeness_1hour_df], ignore_index=True)

# Sort by completeness (descending)
all_completeness = all_completeness.sort_values('completeness_pct', ascending=False)

# Save detailed completeness report
output_completeness = os.path.join(output_dir, "feature_completeness_detailed.csv")
all_completeness.to_csv(output_completeness, index=False)

print(f"\n✓ Saved detailed completeness report: {output_completeness}")
print(f"  Total feature-window combinations analyzed: {len(all_completeness)}")

# Create summary by base variable (aggregating across statistics)
print("\nCreating variable completeness summary...")

variable_summary = []

# Get unique base variables
all_base_vars = all_completeness['base_variable'].unique()

for base_var in all_base_vars:
    var_data = all_completeness[all_completeness['base_variable'] == base_var]
    
    # Calculate average completeness across all statistics for this variable
    avg_completeness_30min = var_data[var_data['window'] == '30-minute']['completeness_pct'].mean()
    avg_completeness_1hour = var_data[var_data['window'] == '1-hour']['completeness_pct'].mean()
    
    # Count how many statistics are available
    stats_count_30min = len(var_data[var_data['window'] == '30-minute'])
    stats_count_1hour = len(var_data[var_data['window'] == '1-hour'])
    
    # Overall average
    overall_avg = var_data['completeness_pct'].mean()
    
    variable_summary.append({
        'variable_name': base_var,
        'avg_completeness_30min': avg_completeness_30min,
        'avg_completeness_1hour': avg_completeness_1hour,
        'overall_avg_completeness': overall_avg,
        'statistics_available_30min': stats_count_30min,
        'statistics_available_1hour': stats_count_1hour,
        'total_features': stats_count_30min + stats_count_1hour
    })

variable_summary_df = pd.DataFrame(variable_summary)
variable_summary_df = variable_summary_df.sort_values('overall_avg_completeness', ascending=False)

# Save variable summary
output_var_summary = os.path.join(output_dir, "variable_completeness_summary.csv")
variable_summary_df.to_csv(output_var_summary, index=False)

print(f"✓ Saved variable completeness summary: {output_var_summary}")
print(f"  Unique variables analyzed: {len(variable_summary_df)}")

# Display top 20 most complete variables
print("\n" + "="*70)
print("TOP 20 MOST COMPLETE VARIABLES")
print("="*70)

top_20 = variable_summary_df.head(20)
print("\n{:<50} {:<15} {:<15} {:<15}".format(
    "Variable", "30min (%)", "1hour (%)", "Overall (%)"))
print("-" * 95)

for _, row in top_20.iterrows():
    var_name = row['variable_name']
    if len(var_name) > 47:
        var_name = var_name[:44] + "..."
    print("{:<50} {:<15.2f} {:<15.2f} {:<15.2f}".format(
        var_name,
        row['avg_completeness_30min'] if pd.notna(row['avg_completeness_30min']) else 0,
        row['avg_completeness_1hour'] if pd.notna(row['avg_completeness_1hour']) else 0,
        row['overall_avg_completeness']
    ))

# Display bottom 10 least complete variables
print("\n" + "="*70)
print("BOTTOM 10 LEAST COMPLETE VARIABLES")
print("="*70)

bottom_10 = variable_summary_df.tail(10)
print("\n{:<50} {:<15} {:<15} {:<15}".format(
    "Variable", "30min (%)", "1hour (%)", "Overall (%)"))
print("-" * 95)

for _, row in bottom_10.iterrows():
    var_name = row['variable_name']
    if len(var_name) > 47:
        var_name = var_name[:44] + "..."
    print("{:<50} {:<15.2f} {:<15.2f} {:<15.2f}".format(
        var_name,
        row['avg_completeness_30min'] if pd.notna(row['avg_completeness_30min']) else 0,
        row['avg_completeness_1hour'] if pd.notna(row['avg_completeness_1hour']) else 0,
        row['overall_avg_completeness']
    ))

# Create a summary statistics file
print("\n" + "="*70)
print("CREATING SUMMARY STATISTICS")
print("="*70)

summary_stats = {
    'dataset': ['30-minute window', '1-hour window', 'Combined windows'],
    'total_responses': [len(ml_30min), len(ml_1hour), len(combined_df)],
    'total_features': [len(features_30min), len(features_1hour), len(combined_features)],
    'total_targets': [len(targets_30min), len(targets_1hour), len([c for c in combined_df.columns if c.startswith('q_')])],
    'unique_users': [ml_30min['userid'].nunique(), ml_1hour['userid'].nunique(), combined_df['userid'].nunique()],
    'avg_feature_completeness': [
        completeness_30min_df['completeness_pct'].mean(),
        completeness_1hour_df['completeness_pct'].mean(),
        all_completeness['completeness_pct'].mean()
    ]
}

summary_stats_df = pd.DataFrame(summary_stats)
output_summary = os.path.join(output_dir, "dataset_summary_statistics.csv")
summary_stats_df.to_csv(output_summary, index=False)

print(f"\n✓ Saved summary statistics: {output_summary}")

# Print summary to console
print("\nDATASET SUMMARY:")
print(summary_stats_df.to_string(index=False))

print("\n" + "="*70)
print("✓ ALL PROCESSING COMPLETE!")
print("="*70)

print("\nGenerated files:")
print(f"  1. {output_30min}")
print(f"  2. {output_1hour}")
print(f"  3. {output_combined}")
print(f"  4. {output_completeness}")
print(f"  5. {output_var_summary}")
print(f"  6. {output_summary}")

print("\nRecommendations for ML:")
print("  - Start with 'ml_ready_combined_windows.csv' for maximum feature richness")
print("  - Use 'variable_completeness_summary.csv' to select high-completeness features")
print("  - Consider removing features with <50% completeness for initial models")
print("  - Variables with 'w30_' prefix are from 30-minute window")
print("  - Variables with 'w60_' prefix are from 1-hour window")
print("  - Target variables start with 'q_' prefix")
