import pandas as pd
import numpy as np
import os

# Define paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/optimized"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("ML DATASET OPTIMIZER")
print("="*70)

# Load the combined dataset and completeness info
print("\nLoading data...")
combined_df = pd.read_csv(os.path.join(ml_dir, "ml_ready_combined_windows.csv"))
completeness_df = pd.read_csv(os.path.join(ml_dir, "feature_completeness_detailed.csv"))

print(f"✓ Loaded combined dataset: {combined_df.shape}")
print(f"✓ Loaded completeness data: {len(completeness_df)} features analyzed")

# Identify feature and target columns
feature_cols = [col for col in combined_df.columns if col.startswith('w30_') or col.startswith('w60_')]
target_cols = [col for col in combined_df.columns if col.startswith('q_')]
metadata_cols = ['userid', 'response_timestamp']

print(f"\nDataset composition:")
print(f"  Features: {len(feature_cols)}")
print(f"  Targets: {len(target_cols)}")
print(f"  Metadata: {len(metadata_cols)}")

# Analyze feature completeness
feature_completeness = {}
for col in feature_cols:
    non_null = combined_df[col].notna().sum()
    completeness_pct = (non_null / len(combined_df)) * 100
    feature_completeness[col] = completeness_pct

completeness_series = pd.Series(feature_completeness).sort_values(ascending=False)

print("\n" + "="*70)
print("FEATURE COMPLETENESS DISTRIBUTION")
print("="*70)

# Show distribution by completeness ranges
ranges = [
    (80, 100, "Excellent (80-100%)"),
    (50, 80, "Good (50-80%)"),
    (20, 50, "Fair (20-50%)"),
    (10, 20, "Poor (10-20%)"),
    (0, 10, "Very Poor (0-10%)")
]

for min_val, max_val, label in ranges:
    count = ((completeness_series >= min_val) & (completeness_series < max_val)).sum()
    if min_val == 0:
        count = (completeness_series < max_val).sum()
    if max_val == 100:
        count = (completeness_series >= min_val).sum()
    print(f"  {label}: {count} features")

# ==============================================================================
# CREATE OPTIMIZED DATASETS WITH DIFFERENT COMPLETENESS THRESHOLDS
# ==============================================================================

print("\n" + "="*70)
print("CREATING OPTIMIZED DATASETS")
print("="*70)

thresholds = [50, 20, 10, 5]

for threshold in thresholds:
    print(f"\n--- Threshold: {threshold}% completeness ---")
    
    # Select features above threshold
    selected_features = [col for col in feature_cols if feature_completeness[col] >= threshold]
    
    print(f"  Selected features: {len(selected_features)}/{len(feature_cols)}")
    
    if len(selected_features) == 0:
        print(f"  ⚠ No features meet {threshold}% threshold")
        continue
    
    # Create optimized dataset
    optimized_cols = metadata_cols + selected_features + target_cols
    optimized_df = combined_df[optimized_cols].copy()
    
    # Save dataset
    output_file = os.path.join(output_dir, f"ml_optimized_min{threshold}pct.csv")
    optimized_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved: {output_file}")
    print(f"    Shape: {optimized_df.shape}")
    
    # Calculate average completeness
    avg_completeness = np.mean([feature_completeness[col] for col in selected_features])
    print(f"    Avg feature completeness: {avg_completeness:.2f}%")

# ==============================================================================
# CREATE HEART RATE FOCUSED DATASET (MOST COMPLETE VARIABLE)
# ==============================================================================

print("\n" + "="*70)
print("CREATING HEART RATE FOCUSED DATASET")
print("="*70)

hr_features = [col for col in feature_cols if 'heart_rate' in col.lower()]
print(f"\nHeart rate features found: {len(hr_features)}")

for feat in hr_features:
    print(f"  {feat}: {feature_completeness[feat]:.2f}%")

hr_cols = metadata_cols + hr_features + target_cols
hr_df = combined_df[hr_cols].copy()

output_hr = os.path.join(output_dir, "ml_heart_rate_focused.csv")
hr_df.to_csv(output_hr, index=False)

print(f"\n✓ Saved heart rate focused dataset: {output_hr}")
print(f"  Shape: {hr_df.shape}")
print(f"  This dataset has the most complete data (~86% completeness)")

# ==============================================================================
# CREATE IMPUTED DATASETS
# ==============================================================================

print("\n" + "="*70)
print("CREATING IMPUTED DATASETS")
print("="*70)

print("\nCreating datasets with different imputation strategies...")

# Strategy 1: Drop rows with any missing features (complete cases only)
print("\n1. Complete cases only (no missing values):")
complete_df = combined_df.dropna(subset=feature_cols)
print(f"   Rows remaining: {len(complete_df)}/{len(combined_df)} ({len(complete_df)/len(combined_df)*100:.1f}%)")

if len(complete_df) > 0:
    output_complete = os.path.join(output_dir, "ml_complete_cases_only.csv")
    complete_df.to_csv(output_complete, index=False)
    print(f"   ✓ Saved: {output_complete}")
else:
    print("   ⚠ No complete cases found")

# Strategy 2: Mean imputation for features with >50% completeness
print("\n2. Mean imputation (features with >50% completeness):")
high_complete_features = [col for col in feature_cols if feature_completeness[col] >= 50]

if len(high_complete_features) > 0:
    imputed_df = combined_df.copy()
    for col in high_complete_features:
        mean_val = imputed_df[col].mean()
        imputed_df[col].fillna(mean_val, inplace=True)
    
    output_imputed = os.path.join(output_dir, "ml_mean_imputed_high_complete.csv")
    imputed_df.to_csv(output_imputed, index=False)
    print(f"   Features imputed: {len(high_complete_features)}")
    print(f"   ✓ Saved: {output_imputed}")
else:
    print("   ⚠ No features with >50% completeness for imputation")

# Strategy 3: Forward fill + median imputation for heart rate
print("\n3. Heart rate with forward fill + median:")
hr_imputed_df = combined_df[hr_cols].copy()

# Sort by userid and timestamp for forward fill
hr_imputed_df = hr_imputed_df.sort_values(['userid', 'response_timestamp'])

for col in hr_features:
    # Forward fill within each user
    hr_imputed_df[col] = hr_imputed_df.groupby('userid')[col].ffill()
    # Backward fill any remaining
    hr_imputed_df[col] = hr_imputed_df.groupby('userid')[col].bfill()
    # Fill any still remaining with median
    median_val = hr_imputed_df[col].median()
    hr_imputed_df[col].fillna(median_val, inplace=True)

output_hr_imputed = os.path.join(output_dir, "ml_heart_rate_imputed.csv")
hr_imputed_df.to_csv(output_hr_imputed, index=False)
print(f"   ✓ Saved: {output_hr_imputed}")
print(f"   All heart rate features now have complete data")

# ==============================================================================
# FEATURE IMPORTANCE ANALYSIS PREPARATION
# ==============================================================================

print("\n" + "="*70)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*70)

# Group features by base variable and window
feature_groups = {}

for col in feature_cols:
    # Extract base variable name
    if col.startswith('w30_'):
        window = '30min'
        base = col.replace('w30_', '')
    else:
        window = '60min'
        base = col.replace('w60_', '')
    
    # Remove statistic suffix
    for suffix in ['_mean', '_std', '_min', '_max', '_median', '_count']:
        if base.endswith(suffix):
            base = base.replace(suffix, '')
            stat = suffix[1:]
            break
    
    if base not in feature_groups:
        feature_groups[base] = []
    feature_groups[base].append({
        'feature': col,
        'window': window,
        'statistic': stat,
        'completeness': feature_completeness[col]
    })

# Create feature selection guide
selection_guide = []

for base_var, features in feature_groups.items():
    avg_completeness = np.mean([f['completeness'] for f in features])
    max_completeness = np.max([f['completeness'] for f in features])
    feature_count = len(features)
    
    # Recommendation
    if max_completeness >= 80:
        recommendation = "HIGHLY RECOMMENDED - Excellent data quality"
    elif max_completeness >= 50:
        recommendation = "RECOMMENDED - Good data quality"
    elif max_completeness >= 20:
        recommendation = "USE WITH CAUTION - Fair data quality"
    elif max_completeness >= 10:
        recommendation = "CONSIDER EXCLUDING - Poor data quality"
    else:
        recommendation = "EXCLUDE - Very poor data quality"
    
    selection_guide.append({
        'base_variable': base_var,
        'feature_count': feature_count,
        'avg_completeness': avg_completeness,
        'max_completeness': max_completeness,
        'recommendation': recommendation
    })

selection_guide_df = pd.DataFrame(selection_guide)
selection_guide_df = selection_guide_df.sort_values('max_completeness', ascending=False)

output_guide = os.path.join(output_dir, "feature_selection_guide.csv")
selection_guide_df.to_csv(output_guide, index=False)

print(f"\n✓ Saved feature selection guide: {output_guide}")
print("\nFeature recommendations:")
print("-" * 70)

for _, row in selection_guide_df.iterrows():
    print(f"\n{row['base_variable']}:")
    print(f"  Max completeness: {row['max_completeness']:.2f}%")
    print(f"  {row['recommendation']}")

# ==============================================================================
# TARGET VARIABLE ANALYSIS
# ==============================================================================

print("\n" + "="*70)
print("TARGET VARIABLE ANALYSIS")
print("="*70)

target_analysis = []

for target in target_cols:
    non_null = combined_df[target].notna().sum()
    completeness = (non_null / len(combined_df)) * 100
    
    if combined_df[target].dtype in ['int64', 'float64']:
        mean_val = combined_df[target].mean()
        std_val = combined_df[target].std()
        min_val = combined_df[target].min()
        max_val = combined_df[target].max()
    else:
        mean_val = std_val = min_val = max_val = None
    
    target_analysis.append({
        'target': target,
        'completeness_pct': completeness,
        'non_null_count': non_null,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    })

target_analysis_df = pd.DataFrame(target_analysis)
output_targets = os.path.join(output_dir, "target_variables_analysis.csv")
target_analysis_df.to_csv(output_targets, index=False)

print(f"\n✓ Saved target analysis: {output_targets}")
print("\nTarget variables:")
for _, row in target_analysis_df.iterrows():
    print(f"  {row['target']}: {row['completeness_pct']:.1f}% complete")

# ==============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ==============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY AND ML RECOMMENDATIONS")
print("="*70)

summary = f"""
DATA QUALITY SUMMARY:
- Total responses: {len(combined_df)}
- Total features: {len(feature_cols)}
- Features with >80% completeness: {len([f for f in feature_cols if feature_completeness[f] >= 80])}
- Features with >50% completeness: {len([f for f in feature_cols if feature_completeness[f] >= 50])}
- Features with >20% completeness: {len([f for f in feature_cols if feature_completeness[f] >= 20])}

RECOMMENDED ML APPROACH:

1. START WITH HEART RATE DATA:
   - Use: ml_heart_rate_focused.csv or ml_heart_rate_imputed.csv
   - Heart rate has 86% completeness - your most reliable predictor
   - This gives you the best chance of building a working model

2. IF YOU NEED MORE FEATURES:
   - Use: ml_optimized_min20pct.csv
   - Includes features with ≥20% completeness
   - Will require handling missing data (imputation or ML algorithms that handle NaN)

3. HANDLING MISSING DATA:
   - Option A: Use tree-based models (XGBoost, LightGBM, CatBoost) - they handle missing values
   - Option B: Use the imputed datasets (ml_heart_rate_imputed.csv)
   - Option C: Use complete cases only (but this reduces sample size significantly)

4. MODEL TYPES TO CONSIDER:
   - Regression: If predicting continuous stress/anxiety scores
   - Classification: If categorizing into stress/anxiety levels
   - Time-series models: If temporal patterns are important

5. FEATURE ENGINEERING IDEAS:
   - Heart rate variability between windows (w60_hr - w30_hr)
   - Rate of change metrics
   - Interaction terms between time windows
   - User-specific baseline deviations

6. VALIDATION STRATEGY:
   - Use user-stratified cross-validation (don't mix users between train/test)
   - This prevents data leakage from same user appearing in both sets
"""

print(summary)

# Save summary to file
summary_file = os.path.join(output_dir, "ML_RECOMMENDATIONS.txt")
with open(summary_file, 'w') as f:
    f.write(summary)

print(f"\n✓ Saved recommendations: {summary_file}")

print("\n" + "="*70)
print("✓ OPTIMIZATION COMPLETE!")
print("="*70)

print(f"\nAll files saved to: {output_dir}")
print("\nGenerated files:")
print("  1. ml_optimized_min*pct.csv - Datasets filtered by completeness threshold")
print("  2. ml_heart_rate_focused.csv - Heart rate features only (best quality)")
print("  3. ml_heart_rate_imputed.csv - Heart rate with imputation (complete data)")
print("  4. feature_selection_guide.csv - Feature recommendations")
print("  5. target_variables_analysis.csv - Target variable statistics")
print("  6. ML_RECOMMENDATIONS.txt - Detailed ML strategy")
