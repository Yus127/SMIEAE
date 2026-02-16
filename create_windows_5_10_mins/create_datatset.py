import pandas as pd
import numpy as np
import os

# Define paths
processed_dir = "/Users/YusMolina/Downloads/smieae/data/processed"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("ML-READY DATASET GENERATOR (SEPARATE WINDOWS)")

# Load the aggregated files
print("\nStep 1: Loading aggregated data files...")

agg_5min_path = os.path.join(processed_dir, "fitbit_5min_window_aggregated.csv")
agg_10min_path = os.path.join(processed_dir, "fitbit_10min_window_aggregated.csv")

if not os.path.exists(agg_5min_path):
    print(f"ERROR: File not found: {agg_5min_path}")
    exit(1)

if not os.path.exists(agg_10min_path):
    print(f"ERROR: File not found: {agg_10min_path}")
    exit(1)

df_5min = pd.read_csv(agg_5min_path)
df_10min = pd.read_csv(agg_10min_path)

print(f" Loaded 5-minute window data: {len(df_5min)} rows")
print(f" Loaded 10-minute window data: {len(df_10min)} rows")

# PART 1: CREATE ML-READY DATASETS (SEPARATE)

print("\n" + "="*70)
print("PART 1: CREATING ML-READY DATASETS")

def create_ml_dataset(df, window_name, min_completeness=10.0):
    """
    Create a clean ML-ready dataset from aggregated data
    Removes features with less than min_completeness% data
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
    
    print(f"  Found {len(feature_cols)} feature columns (before filtering)")
    print(f"  Found {len(target_cols)} target columns")
    
    # Calculate completeness for each feature
    feature_completeness = {}
    features_to_keep = []
    features_removed = []
    
    for col in feature_cols:
        non_null = df[col].notna().sum()
        total = len(df)
        completeness_pct = (non_null / total) * 100
        
        feature_completeness[col] = {
            'non_null_count': non_null,
            'null_count': total - non_null,
            'total_count': total,
            'completeness_pct': completeness_pct
        }
        
        # Filter based on completeness threshold
        if completeness_pct >= min_completeness:
            features_to_keep.append(col)
        else:
            features_removed.append((col, completeness_pct))
    
    print(f"  Removed {len(features_removed)} features with <{min_completeness}% completeness")
    print(f"  Kept {len(features_to_keep)} features with >={min_completeness}% completeness")
    
    # Create ML dataset with metadata, filtered features, and targets
    ml_cols = metadata_cols + features_to_keep + target_cols
    ml_df = df[ml_cols].copy()
    
    return ml_df, features_to_keep, target_cols, feature_completeness, features_removed

# Create ML datasets for both windows (filtering features with <10% completeness)
ml_5min, features_5min, targets_5min, completeness_5min, removed_5min = create_ml_dataset(df_5min, "5-minute", min_completeness=10.0)
ml_10min, features_10min, targets_10min, completeness_10min, removed_10min = create_ml_dataset(df_10min, "10-minute", min_completeness=10.0)

# Save ML-ready datasets
output_5min = os.path.join(output_dir, "ml_ready_5min_window.csv")
output_10min = os.path.join(output_dir, "ml_ready_10min_window.csv")

output_dir_removed_var =  os.path.join(output_dir, "removed_variables_summaries")

ml_5min.to_csv(output_5min, index=False)
ml_10min.to_csv(output_10min, index=False)

# Save list of removed features for reference
if removed_5min:
    removed_5min_df = pd.DataFrame(removed_5min, columns=['feature_name', 'completeness_pct'])
    removed_5min_df = removed_5min_df.sort_values('completeness_pct', ascending=True)
    output_removed_5min = os.path.join(output_dir_removed_var, "removed_features_5min.csv")
    removed_5min_df.to_csv(output_removed_5min, index=False)
    print(f"\n Saved list of removed features (5-min): {output_removed_5min}")

if removed_10min:
    removed_10min_df = pd.DataFrame(removed_10min, columns=['feature_name', 'completeness_pct'])
    removed_10min_df = removed_10min_df.sort_values('completeness_pct', ascending=True)
    output_removed_10min = os.path.join(output_dir_removed_var, "removed_features_10min.csv")
    removed_10min_df.to_csv(output_removed_10min, index=False)
    print(f"\n Saved list of removed features (10-min): {output_removed_10min}")

print(f"\n Saved 5-minute ML dataset: {output_5min}")
print(f"  Shape: {ml_5min.shape} (rows x columns)")
print(f"  Features: {len(features_5min)}")
print(f"  Targets: {len(targets_5min)}")
print(f"  Unique users: {ml_5min['userid'].nunique()}")

print(f"\n Saved 10-minute ML dataset: {output_10min}")
print(f"  Shape: {ml_10min.shape} (rows x columns)")
print(f"  Features: {len(features_10min)}")
print(f"  Targets: {len(targets_10min)}")
print(f"  Unique users: {ml_10min['userid'].nunique()}")

# PART 2: DATA COMPLETENESS ANALYSIS

print("\n" + "="*70)
print("PART 2: DATA COMPLETENESS ANALYSIS")

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
completeness_5min_df = create_completeness_report(ml_5min, features_5min, "5-minute")
completeness_10min_df = create_completeness_report(ml_10min, features_10min, "10-minute")

# Save individual completeness reports
output_completeness_5min = os.path.join(output_dir_removed_var, "feature_completeness_5min.csv")
output_completeness_10min = os.path.join(output_dir_removed_var, "feature_completeness_10min.csv")

completeness_5min_df.to_csv(output_completeness_5min, index=False)
completeness_10min_df.to_csv(output_completeness_10min, index=False)

print(f"\n Saved 5-minute completeness report: {output_completeness_5min}")
print(f"  Features analyzed: {len(completeness_5min_df)}")

print(f"\n Saved 10-minute completeness report: {output_completeness_10min}")
print(f"  Features analyzed: {len(completeness_10min_df)}")

# PART 3: VARIABLE COMPLETENESS SUMMARIES

print("\n" + "="*70)
print("PART 3: VARIABLE COMPLETENESS SUMMARIES")

def create_variable_summary(completeness_df, window_name):
    """
    Create summary by base variable (aggregating across statistics)
    """
    print(f"\nCreating variable summary for {window_name} window...")
    
    variable_summary = []
    
    # Get unique base variables
    all_base_vars = completeness_df['base_variable'].unique()
    
    for base_var in all_base_vars:
        var_data = completeness_df[completeness_df['base_variable'] == base_var]
        
        # Calculate average completeness across all statistics for this variable
        avg_completeness = var_data['completeness_pct'].mean()
        max_completeness = var_data['completeness_pct'].max()
        min_completeness = var_data['completeness_pct'].min()
        
        # Count how many statistics are available
        stats_count = len(var_data)
        
        # Get list of statistic types
        stat_types = sorted(var_data['statistic_type'].unique())
        
        variable_summary.append({
            'variable_name': base_var,
            'avg_completeness': avg_completeness,
            'max_completeness': max_completeness,
            'min_completeness': min_completeness,
            'statistics_count': stats_count,
            'statistics_types': ', '.join(stat_types)
        })
    
    return pd.DataFrame(variable_summary).sort_values('avg_completeness', ascending=False)

# Create variable summaries
var_summary_5min = create_variable_summary(completeness_5min_df, "5-minute")
var_summary_10min = create_variable_summary(completeness_10min_df, "10-minute")

# Save variable summaries
output_var_summary_5min = os.path.join(output_dir_removed_var, "variable_summary_5min.csv")
output_var_summary_10min = os.path.join(output_dir_removed_var, "variable_summary_10min.csv")

var_summary_5min.to_csv(output_var_summary_5min, index=False)
var_summary_10min.to_csv(output_var_summary_10min, index=False)

print(f"\n Saved 5-minute variable summary: {output_var_summary_5min}")
print(f"  Unique variables: {len(var_summary_5min)}")

print(f"\n Saved 10-minute variable summary: {output_var_summary_10min}")
print(f"  Unique variables: {len(var_summary_10min)}")

# PART 4: DISPLAY TOP/BOTTOM VARIABLES

print("\n" + "="*70)
print("TOP 20 MOST COMPLETE VARIABLES (5-MINUTE WINDOW)")

top_20_5min = var_summary_5min.head(20)
print("\n{:<50} {:<15} {:<10}".format("Variable", "Avg Complete (%)", "Stats Count"))
print("-" * 75)

for _, row in top_20_5min.iterrows():
    var_name = row['variable_name']
    if len(var_name) > 47:
        var_name = var_name[:44] + "..."
    print("{:<50} {:<15.2f} {:<10}".format(
        var_name,
        row['avg_completeness'],
        row['statistics_count']
    ))

print("\n" + "="*70)
print("TOP 20 MOST COMPLETE VARIABLES (10-MINUTE WINDOW)")

top_20_10min = var_summary_10min.head(20)
print("\n{:<50} {:<15} {:<10}".format("Variable", "Avg Complete (%)", "Stats Count"))
print("-" * 75)

for _, row in top_20_10min.iterrows():
    var_name = row['variable_name']
    if len(var_name) > 47:
        var_name = var_name[:44] + "..."
    print("{:<50} {:<15.2f} {:<10}".format(
        var_name,
        row['avg_completeness'],
        row['statistics_count']
    ))

# PART 5: SUMMARY STATISTICS

print("\n" + "="*70)
print("DATASET SUMMARY STATISTICS")

summary_stats_5min = {
    'metric': [
        'Total responses',
        'Total features',
        'Total targets',
        'Unique users',
        'Avg feature completeness (%)',
        'Median feature completeness (%)',
        'Features with >80% completeness',
        'Features with >50% completeness'
    ],
    'value': [
        len(ml_5min),
        len(features_5min),
        len(targets_5min),
        ml_5min['userid'].nunique(),
        completeness_5min_df['completeness_pct'].mean(),
        completeness_5min_df['completeness_pct'].median(),
        (completeness_5min_df['completeness_pct'] > 80).sum(),
        (completeness_5min_df['completeness_pct'] > 50).sum()
    ]
}

summary_stats_10min = {
    'metric': [
        'Total responses',
        'Total features',
        'Total targets',
        'Unique users',
        'Avg feature completeness (%)',
        'Median feature completeness (%)',
        'Features with >80% completeness',
        'Features with >50% completeness'
    ],
    'value': [
        len(ml_10min),
        len(features_10min),
        len(targets_10min),
        ml_10min['userid'].nunique(),
        completeness_10min_df['completeness_pct'].mean(),
        completeness_10min_df['completeness_pct'].median(),
        (completeness_10min_df['completeness_pct'] > 80).sum(),
        (completeness_10min_df['completeness_pct'] > 50).sum()
    ]
}

summary_5min_df = pd.DataFrame(summary_stats_5min)
summary_10min_df = pd.DataFrame(summary_stats_10min)

# Save summaries
output_summary_5min = os.path.join(output_dir_removed_var, "dataset_summary_5min.csv")
output_summary_10min = os.path.join(output_dir_removed_var, "dataset_summary_10min.csv")

summary_5min_df.to_csv(output_summary_5min, index=False)
summary_10min_df.to_csv(output_summary_10min, index=False)

print(f"\n Saved 5-minute summary: {output_summary_5min}")
print(f"\n5-MINUTE WINDOW SUMMARY:")
print(summary_5min_df.to_string(index=False))

print(f"\n Saved 10-minute summary: {output_summary_10min}")
print(f"\n10-MINUTE WINDOW SUMMARY:")
print(summary_10min_df.to_string(index=False))

# FINAL SUMMARY

print("\n" + "="*70)

print("\nGenerated files for 5-MINUTE WINDOW:")
print(f"  1. {output_5min}")
print(f"  2. {output_completeness_5min}")
print(f"  3. {output_var_summary_5min}")
print(f"  4. {output_summary_5min}")
if removed_5min:
    print(f"  5. {output_removed_5min} ({len(removed_5min)} features removed)")

print("\nGenerated files for 10-MINUTE WINDOW:")
print(f"  1. {output_10min}")
print(f"  2. {output_completeness_10min}")
print(f"  3. {output_var_summary_10min}")
print(f"  4. {output_summary_10min}")
if removed_10min:
    print(f"  5. {output_removed_10min} ({len(removed_10min)} features removed)")

print("\nRecommendations for ML:")
print("  5-MINUTE WINDOW:")
print(f"    - Main dataset: ml_ready_5min_window.csv")
print(f"    - {len(ml_5min)} total responses from {ml_5min['userid'].nunique()} users")
print(f"    - {len(features_5min)} features (after filtering), {len(targets_5min)} targets")
print(f"    - {len(removed_5min)} features removed due to <10% completeness")
print(f"    - Avg feature completeness: {completeness_5min_df['completeness_pct'].mean():.1f}%")
print("\n  10-MINUTE WINDOW:")
print(f"    - Main dataset: ml_ready_10min_window.csv")
print(f"    - {len(ml_10min)} total responses from {ml_10min['userid'].nunique()} users")
print(f"    - {len(features_10min)} features (after filtering), {len(targets_10min)} targets")
print(f"    - {len(removed_10min)} features removed due to <10% completeness")
print(f"    - Avg feature completeness: {completeness_10min_df['completeness_pct'].mean():.1f}%")
print("\n  GENERAL TIPS:")
print("    - Features with <10% completeness have been automatically removed")
print("    - Consider further filtering features with <50% completeness for robust models")
print("    - Target variables start with 'q_' prefix")
print("    - Each window provides different temporal context")
print("    - Compare model performance across both windows to determine optimal timeframe")