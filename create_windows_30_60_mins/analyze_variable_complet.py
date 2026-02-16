import pandas as pd
import numpy as np
import os
import glob

# Define paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"

print("MOST COMPLETE VARIABLES ANALYSIS")

# Try to find any available ML dataset
possible_files = [
    "enriched/ml_ready_30min_window_enriched.csv",
    "enriched/ml_ready_60min_window_enriched.csv"#,
    #"enriched/ml_ready_combined_windows_enriched.csv"
]

dataset_file = None
for file in possible_files:
    full_path = os.path.join(ml_dir, file)
    if os.path.exists(full_path):
        dataset_file = full_path
        break

if dataset_file is None:
    # Try to find any CSV file
    csv_files = glob.glob(os.path.join(ml_dir, "**/*.csv"), recursive=True)
    if csv_files:
        dataset_file = csv_files[0]

if dataset_file is None:
    print("\nERROR: No ML datasets found!")
    print(f"Please check that files exist in: {ml_dir}")
    exit(1)

print(f"\nAnalyzing dataset: {os.path.basename(dataset_file)}")
print("-"*80)

# Load the dataset
df = pd.read_csv(dataset_file)
print(f" Loaded dataset with {len(df)} rows and {len(df.columns)} columns\n")

# Identify feature columns (exclude metadata and target columns)
metadata_cols = ['userid', 'response_timestamp', 'window_start', 'window_end', 
                'record_count', 'university', 'course', 'date_only']
target_cols = [col for col in df.columns if col.startswith('q_')]

# Get all feature columns
feature_cols = [col for col in df.columns 
               if col not in metadata_cols and col not in target_cols]

print(f"Dataset composition:")
print(f"  • Metadata columns: {len([c for c in metadata_cols if c in df.columns])}")
print(f"  • Feature columns: {len(feature_cols)}")
print(f"  • Target columns: {len(target_cols)}")

# Calculate completeness for each feature
completeness_data = []

for col in feature_cols:
    non_null = df[col].notna().sum()
    total = len(df)
    completeness_pct = (non_null / total) * 100
    
    # Try to extract base variable name
    base_var = col
    window = "N/A"
    stat_type = "N/A"
    
    # Check for window prefix
    if col.startswith('w30_'):
        window = "30min"
        base_var = col.replace('w30_', '')
    elif col.startswith('w60_'):
        window = "60min"
        base_var = col.replace('w60_', '')
    
    # Check for statistic suffix
    for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']:
        if base_var.endswith(suffix):
            base_var = base_var.replace(suffix, '')
            stat_type = suffix[1:]
            break
    
    completeness_data.append({
        'feature_name': col,
        'base_variable': base_var,
        'window': window,
        'statistic': stat_type,
        'non_null_count': non_null,
        'total_count': total,
        'completeness_pct': completeness_pct
    })

completeness_df = pd.DataFrame(completeness_data)
completeness_df = completeness_df.sort_values('completeness_pct', ascending=False)

# Create summary by base variable
print("\n" + "="*80)
print("VARIABLE COMPLETENESS RANKING")

variable_summary = completeness_df.groupby('base_variable').agg({
    'completeness_pct': ['mean', 'min', 'max', 'count']
}).reset_index()

variable_summary.columns = ['variable_name', 'avg_completeness', 'min_completeness', 
                           'max_completeness', 'num_features']
variable_summary = variable_summary.sort_values('avg_completeness', ascending=False)

print(f"\n{'Rank':<6} {'Variable Name':<45} {'Avg %':<10} {'Min %':<10} {'Max %':<10} {'Features':<10}")
print("-"*80)

for idx, row in variable_summary.iterrows():
    rank = idx + 1
    var_name = row['variable_name']
    if len(var_name) > 42:
        var_name = var_name[:39] + "..."
    
    print(f"{rank:<6} {var_name:<45} {row['avg_completeness']:>8.2f}% {row['min_completeness']:>8.2f}% {row['max_completeness']:>8.2f}% {int(row['num_features']):<10}")

# Detailed breakdown for top variables
print("\n" + "="*80)
print("DETAILED BREAKDOWN OF TOP 5 VARIABLES")

for idx, row in variable_summary.head(5).iterrows():
    var_name = row['variable_name']
    print(f"\n{idx+1}. {var_name}")
    print(f"   Average completeness: {row['avg_completeness']:.2f}%")
    
    # Get all features for this variable
    var_features = completeness_df[completeness_df['base_variable'] == var_name]
    
    if len(var_features) > 0:
        print(f"   Available statistics:")
        for _, feat in var_features.iterrows():
            window_str = f"[{feat['window']}]" if feat['window'] != "N/A" else ""
            stat_str = f"{feat['statistic']}" if feat['statistic'] != "N/A" else ""
            print(f"     • {window_str} {stat_str}: {feat['completeness_pct']:.2f}% ({feat['non_null_count']}/{feat['total_count']})")

# Categorization
print("\n" + "="*80)
print("COMPLETENESS CATEGORIES")

excellent = variable_summary[variable_summary['avg_completeness'] >= 80]
good = variable_summary[(variable_summary['avg_completeness'] >= 50) & 
                       (variable_summary['avg_completeness'] < 80)]
fair = variable_summary[(variable_summary['avg_completeness'] >= 20) & 
                       (variable_summary['avg_completeness'] < 50)]
poor = variable_summary[(variable_summary['avg_completeness'] >= 10) & 
                       (variable_summary['avg_completeness'] < 20)]
very_poor = variable_summary[variable_summary['avg_completeness'] < 10]

print(f"\n EXCELLENT (≥80% completeness): {len(excellent)} variables")
if len(excellent) > 0:
    for _, row in excellent.iterrows():
        print(f"   • {row['variable_name']}: {row['avg_completeness']:.2f}%")

print(f"\n○ GOOD (50-80% completeness): {len(good)} variables")
if len(good) > 0:
    for _, row in good.iterrows():
        print(f"   • {row['variable_name']}: {row['avg_completeness']:.2f}%")

print(f"\n△ FAIR (20-50% completeness): {len(fair)} variables")
if len(fair) > 0:
    for _, row in fair.iterrows():
        print(f"   • {row['variable_name']}: {row['avg_completeness']:.2f}%")

print(f"\n POOR (10-20% completeness): {len(poor)} variables")
if len(poor) > 0:
    for _, row in poor.iterrows():
        print(f"   • {row['variable_name']}: {row['avg_completeness']:.2f}%")

print(f"\n VERY POOR (<10% completeness): {len(very_poor)} variables")
if len(very_poor) > 0:
    for _, row in very_poor.iterrows():
        print(f"   • {row['variable_name']}: {row['avg_completeness']:.2f}%")

# ML Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS FOR MACHINE LEARNING")

if len(excellent) > 0:
    print(f"\n PRIMARY FEATURES (Start here):")
    print(f"   Use these {len(excellent)} variable(s) for your initial models:")
    for _, row in excellent.iterrows():
        print(f"    {row['variable_name']} ({row['avg_completeness']:.2f}% complete)")
        print(f"     → Excellent data availability, reliable predictions")

if len(good) > 0:
    print(f"\n SECONDARY FEATURES (Add if needed):")
    print(f"   Consider these {len(good)} variable(s) to enhance models:")
    for _, row in good.iterrows():
        print(f"   ○ {row['variable_name']} ({row['avg_completeness']:.2f}% complete)")

if len(fair) > 0 or len(poor) > 0 or len(very_poor) > 0:
    sparse_count = len(fair) + len(poor) + len(very_poor)
    print(f"\n  SPARSE FEATURES (Use with caution):")
    print(f"   {sparse_count} variable(s) have <50% completeness")
    print(f"   → Require missing data strategies:")
    print(f"     • Tree-based models (XGBoost, LightGBM) handle NaN natively")
    print(f"     • Imputation (forward-fill, median, etc.)")
    print(f"     • Remove from analysis if not critical")

# Context-aware recommendations
if 'daily_total_steps' in df.columns:
    steps_completeness = (df['daily_total_steps'].notna().sum() / len(df)) * 100
    print(f"\n PHYSICAL ACTIVITY:")
    print(f"   daily_total_steps: {steps_completeness:.2f}% complete")
    if steps_completeness > 50:
        print(f"   → Good supplement to physiological features")

if 'is_exam_period' in df.columns:
    exam_count = df['is_exam_period'].sum()
    print(f"\n CONTEXTUAL FEATURES:")
    print(f"   Exam period responses: {exam_count}/{len(df)} ({exam_count/len(df)*100:.1f}%)")
    print(f"   → Use to analyze stress differences during exams")

print("\n" + "="*80)
print("SUMMARY")
print(f"\nDataset: {len(df)} observations")
print(f"Total variables: {len(variable_summary)}")
print(f"High-quality variables (≥50%): {len(excellent) + len(good)}")
print(f"Overall average completeness: {variable_summary['avg_completeness'].mean():.2f}%")

if len(excellent) > 0:
    best_var = variable_summary.iloc[0]
    print(f"\n BEST VARIABLE: {best_var['variable_name']}")
    print(f"   Completeness: {best_var['avg_completeness']:.2f}%")
    print(f"   Number of features: {int(best_var['num_features'])}")
    print(f"\n    RECOMMENDATION: Start your ML modeling with this variable!")

# Save detailed report
output_file = os.path.join(ml_dir, "variable_completeness_analysis.csv")
completeness_df.to_csv(output_file, index=False)
print(f"\n Detailed analysis saved to: {output_file}")

summary_file = os.path.join(ml_dir, "variable_summary.csv")
variable_summary.to_csv(summary_file, index=False)
print(f" Summary saved to: {summary_file}")

print("\n" + "="*80)