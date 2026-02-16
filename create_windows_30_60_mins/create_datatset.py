import pandas as pd
import numpy as np
import os

processed_dir = "/Users/YusMolina/Downloads/smieae/data/processed"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"

os.makedirs(output_dir, exist_ok=True)

print("ML-READY DATASET GENERATOR (SEPARATE WINDOWS)")

# Load the aggregated files

agg_30min_path = os.path.join(processed_dir, "fitbit_30min_window_aggregated.csv")
agg_60min_path = os.path.join(processed_dir, "fitbit_1hour_window_aggregated.csv")

if not os.path.exists(agg_30min_path):
    print(f"ERROR: File not found: {agg_30min_path}")
    exit(1)

if not os.path.exists(agg_60min_path):
    print(f"ERROR: File not found: {agg_60min_path}")
    exit(1)

df_30min = pd.read_csv(agg_30min_path)
df_60min = pd.read_csv(agg_60min_path)

print(f" Loaded 30-minute window data: {len(df_30min)} rows")
print(f" Loaded 60-minute window data: {len(df_60min)} rows")

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
ml_30min, features_30min, targets_30min, completeness_30min, removed_30min = create_ml_dataset(df_30min, "30-minute", min_completeness=10.0)
ml_60min, features_60min, targets_60min, completeness_60min, removed_60min = create_ml_dataset(df_60min, "60-minute", min_completeness=10.0)

# Save ML-ready datasets
output_30min = os.path.join(output_dir, "ml_ready_30min_window.csv")
output_60min = os.path.join(output_dir, "ml_ready_60min_window.csv")

output_dir_removed_var =  os.path.join(output_dir, "removed_variables_summaries")

ml_30min.to_csv(output_30min, index=False)
ml_60min.to_csv(output_60min, index=False)

# Save list of removed features for reference
if removed_30min:
    removed_30min_df = pd.DataFrame(removed_30min, columns=['feature_name', 'completeness_pct'])
    removed_30min_df = removed_30min_df.sort_values('completeness_pct', ascending=True)
    output_removed_30min = os.path.join(output_dir_removed_var, "removed_features_30min.csv")
    removed_30min_df.to_csv(output_removed_30min, index=False)
    print(f"\n Saved list of removed features (30-min): {output_removed_30min}")

if removed_60min:
    removed_60min_df = pd.DataFrame(removed_60min, columns=['feature_name', 'completeness_pct'])
    removed_60min_df = removed_60min_df.sort_values('completeness_pct', ascending=True)
    output_removed_60min = os.path.join(output_dir_removed_var, "removed_features_60min.csv")
    removed_60min_df.to_csv(output_removed_60min, index=False)
    print(f"\n Saved list of removed features (60-min): {output_removed_60min}")

print(f"\n Saved 30-minute ML dataset: {output_30min}")
print(f"  Shape: {ml_30min.shape} (rows x columns)")
print(f"  Features: {len(features_30min)}")
print(f"  Targets: {len(targets_30min)}")
print(f"  Unique users: {ml_30min['userid'].nunique()}")

print(f"\n Saved 60-minute ML dataset: {output_60min}")
print(f"  Shape: {ml_60min.shape} (rows x columns)")
print(f"  Features: {len(features_60min)}")
print(f"  Targets: {len(targets_60min)}")
print(f"  Unique users: {ml_60min['userid'].nunique()}")

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
completeness_30min_df = create_completeness_report(ml_30min, features_30min, "30-minute")
completeness_60min_df = create_completeness_report(ml_60min, features_60min, "60-minute")

# Save individual completeness reports
output_completeness_30min = os.path.join(output_dir_removed_var, "feature_completeness_30min.csv")
output_completeness_60min = os.path.join(output_dir_removed_var, "feature_completeness_60min.csv")

completeness_30min_df.to_csv(output_completeness_30min, index=False)
completeness_60min_df.to_csv(output_completeness_60min, index=False)

print(f"\n Saved 30-minute completeness report: {output_completeness_30min}")
print(f"  Features analyzed: {len(completeness_30min_df)}")

print(f"\n Saved 60-minute completeness report: {output_completeness_60min}")
print(f"  Features analyzed: {len(completeness_60min_df)}")

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
var_summary_30min = create_variable_summary(completeness_30min_df, "30-minute")
var_summary_60min = create_variable_summary(completeness_60min_df, "60-minute")

# Save variable summaries
output_var_summary_30min = os.path.join(output_dir_removed_var, "variable_summary_30min.csv")
output_var_summary_60min = os.path.join(output_dir_removed_var, "variable_summary_60min.csv")

var_summary_30min.to_csv(output_var_summary_30min, index=False)
var_summary_60min.to_csv(output_var_summary_60min, index=False)

print(f"\n Saved 30-minute variable summary: {output_var_summary_30min}")
print(f"  Unique variables: {len(var_summary_30min)}")

print(f"\n Saved 60-minute variable summary: {output_var_summary_60min}")
print(f"  Unique variables: {len(var_summary_60min)}")

# PART 4: DISPLAY TOP/BOTTOM VARIABLES

print("\n" + "="*70)
print("TOP 20 MOST COMPLETE VARIABLES (30-MINUTE WINDOW)")

top_20_30min = var_summary_30min.head(20)
print("\n{:<50} {:<15} {:<10}".format("Variable", "Avg Complete (%)", "Stats Count"))
print("-" * 75)

for _, row in top_20_30min.iterrows():
    var_name = row['variable_name']
    if len(var_name) > 47:
        var_name = var_name[:44] + "..."
    print("{:<50} {:<15.2f} {:<10}".format(
        var_name,
        row['avg_completeness'],
        row['statistics_count']
    ))

print("\n" + "="*70)
print("TOP 20 MOST COMPLETE VARIABLES (60-MINUTE WINDOW)")

top_20_60min = var_summary_60min.head(20)
print("\n{:<50} {:<15} {:<10}".format("Variable", "Avg Complete (%)", "Stats Count"))
print("-" * 75)

for _, row in top_20_60min.iterrows():
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

summary_stats_30min = {
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
        len(ml_30min),
        len(features_30min),
        len(targets_30min),
        ml_30min['userid'].nunique(),
        completeness_30min_df['completeness_pct'].mean(),
        completeness_30min_df['completeness_pct'].median(),
        (completeness_30min_df['completeness_pct'] > 80).sum(),
        (completeness_30min_df['completeness_pct'] > 50).sum()
    ]
}

summary_stats_60min = {
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
        len(ml_60min),
        len(features_60min),
        len(targets_60min),
        ml_60min['userid'].nunique(),
        completeness_60min_df['completeness_pct'].mean(),
        completeness_60min_df['completeness_pct'].median(),
        (completeness_60min_df['completeness_pct'] > 80).sum(),
        (completeness_60min_df['completeness_pct'] > 50).sum()
    ]
}

summary_30min_df = pd.DataFrame(summary_stats_30min)
summary_60min_df = pd.DataFrame(summary_stats_60min)

# Save summaries
output_summary_30min = os.path.join(output_dir_removed_var, "dataset_summary_30min.csv")
output_summary_60min = os.path.join(output_dir_removed_var, "dataset_summary_60min.csv")

summary_30min_df.to_csv(output_summary_30min, index=False)
summary_60min_df.to_csv(output_summary_60min, index=False)

print(f"\n Saved 30-minute summary: {output_summary_30min}")
print(f"\n30-MINUTE WINDOW SUMMARY:")
print(summary_30min_df.to_string(index=False))

print(f"\n Saved 60-minute summary: {output_summary_60min}")
print(f"\n60-MINUTE WINDOW SUMMARY:")
print(summary_60min_df.to_string(index=False))

# FINAL SUMMARY

print("\n" + "="*70)

print("\nGenerated files for 30-MINUTE WINDOW:")
print(f"  1. {output_30min}")
print(f"  2. {output_completeness_30min}")
print(f"  3. {output_var_summary_30min}")
print(f"  4. {output_summary_30min}")
if removed_30min:
    print(f"  5. {output_removed_30min} ({len(removed_30min)} features removed)")

print("\nGenerated files for 60-MINUTE WINDOW:")
print(f"  1. {output_60min}")
print(f"  2. {output_completeness_60min}")
print(f"  3. {output_var_summary_60min}")
print(f"  4. {output_summary_60min}")
if removed_60min:
    print(f"  5. {output_removed_60min} ({len(removed_60min)} features removed)")

print("\nRecommendations for ML:")
print("  30-MINUTE WINDOW:")
print(f"    - Main dataset: ml_ready_30min_window.csv")
print(f"    - {len(ml_30min)} total responses from {ml_30min['userid'].nunique()} users")
print(f"    - {len(features_30min)} features (after filtering), {len(targets_30min)} targets")
print(f"    - {len(removed_30min)} features removed due to <10% completeness")
print(f"    - Avg feature completeness: {completeness_30min_df['completeness_pct'].mean():.1f}%")
print("\n  60-MINUTE WINDOW:")
print(f"    - Main dataset: ml_ready_60min_window.csv")
print(f"    - {len(ml_60min)} total responses from {ml_60min['userid'].nunique()} users")
print(f"    - {len(features_60min)} features (after filtering), {len(targets_60min)} targets")
print(f"    - {len(removed_60min)} features removed due to <10% completeness")
print(f"    - Avg feature completeness: {completeness_60min_df['completeness_pct'].mean():.1f}%")
print("\n  GENERAL TIPS:")
print("    - Features with <10% completeness have been automatically removed")
print("    - Consider further filtering features with <50% completeness for robust models")
print("    - Target variables start with 'q_' prefix")
print("    - Each window provides different temporal context")
print("    - Compare model performance across both windows to determine optimal timeframe")