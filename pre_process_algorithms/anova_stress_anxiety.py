import pandas as pd
import numpy as np
from scipy import stats

# Load the dataset
df = pd.read_csv('/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data.csv')

print("="*80)
print("CREATING LOG-TRANSFORMED DATASET FOR ML MODELING")
print("="*80)

# ============================================================================
# STEP 1: IDENTIFY SEVERELY SKEWED VARIABLES TO TRANSFORM
# ============================================================================

# Original list of severely skewed variables (from diagnostics)
severely_skewed_original = [
    'sleep_global_minutesAwake',
    'sleep_global_efficiency',
    'sleep_stage_transitions',
    'hrv_details_rmssd_std',
    'hrv_details_rmssd_max'
]

# Your new predictor variables
original_predictors = [
    # Activity variables
    'daily_total_steps',
    'activity_level_sedentary_count',
    'activity_level_lightly_active_count',
    'activity_level_moderately_active_count',
    'activity_level_very_active_count',
    
    # Heart rate during activity
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_min',
    'heart_rate_activity_beats per minute_max',
    
    # Sleep summary
    'sleep_global_duration',
    'sleep_global_minutesAwake',  # TRANSFORMED
    'sleep_global_efficiency',  # TRANSFORMED
    'sleep_stage_transitions',  # TRANSFORMED
    'deep_sleep_minutes',
    'last_sleep_stage',
    'respiratory_rate_summary_rem_sleep_signal_to_noise',
    'rem_avg_segment_duration_minutes',
    
    # Additional physiological variables
    'daily_hrv_summary_rmssd',
    'daily_respiratory_rate_daily_respiratory_rate',
    'minute_spo2_value_mean',
    'minute_spo2_value_min',
    
    # HRV
    'hrv_details_rmssd_mean',
    'hrv_details_rmssd_std',  # TRANSFORMED
    'hrv_details_rmssd_min',
    'hrv_details_rmssd_max',  # TRANSFORMED
    
    # Target variables
    'stress_level',
    'anxiety_level'
]

# Variables that need transformation (intersection of severely skewed and your list)
variables_to_transform = [
    'sleep_global_minutesAwake',
    'sleep_global_efficiency',
    'sleep_stage_transitions',
    'hrv_details_rmssd_std',
    'hrv_details_rmssd_max'
]

print("\nVariables that will be log-transformed:")
for var in variables_to_transform:
    if var in df.columns:
        print(f"  ✓ {var}")
    else:
        print(f"  ✗ {var} (NOT FOUND in dataset)")

# ============================================================================
# STEP 2: CHECK SKEWNESS FOR NEW VARIABLES
# ============================================================================

print("\n" + "="*80)
print("CHECKING SKEWNESS FOR ALL PREDICTORS")
print("="*80)

skewness_report = []
for var in original_predictors:
    if var not in df.columns:
        print(f"Warning: {var} not found in dataset")
        continue
    
    data = df[var].dropna()
    if len(data) < 3:
        continue
    
    skewness = stats.skew(data)
    severely_skewed = abs(skewness) > 2
    
    skewness_report.append({
        'Variable': var,
        'Skewness': skewness,
        'Abs_Skewness': abs(skewness),
        'Severely_Skewed': severely_skewed,
        'Needs_Transform': var in variables_to_transform or severely_skewed
    })

skew_df = pd.DataFrame(skewness_report).sort_values('Abs_Skewness', ascending=False)
print("\nSkewness Report (sorted by severity):")
print(skew_df.to_string(index=False))

# Identify any NEW severely skewed variables not in our transform list
new_severely_skewed = skew_df[
    (skew_df['Severely_Skewed'] == True) & 
    (~skew_df['Variable'].isin(variables_to_transform))
]['Variable'].tolist()

if new_severely_skewed:
    print("\n⚠️  WARNING: Found additional severely skewed variables:")
    for var in new_severely_skewed:
        skew_val = skew_df[skew_df['Variable'] == var]['Skewness'].values[0]
        print(f"  • {var} (skewness = {skew_val:.2f})")
        variables_to_transform.append(var)
    print("\n✓ Adding these to transformation list")

# ============================================================================
# STEP 3: APPLY LOG TRANSFORMATIONS
# ============================================================================

print("\n" + "="*80)
print("APPLYING LOG TRANSFORMATIONS")
print("="*80)

df_transformed = df.copy()
transformation_log = []

for var in variables_to_transform:
    if var not in df.columns:
        print(f"  ✗ Skipping {var} (not found)")
        continue
    
    # Create new column name
    new_var_name = f"{var}_log"
    
    # Check if variable has negative values
    min_val = df[var].min()
    
    if var == 'sleep_global_efficiency':
        # Efficiency is left-skewed (negative skew), use reversed transformation
        max_val = df[var].max()
        df_transformed[new_var_name] = np.log1p(max_val - df[var] + 1)
        transform_type = "Reversed log (for negative skew)"
    elif min_val < 0:
        # Has negative values, shift before log
        shift = abs(min_val) + 1
        df_transformed[new_var_name] = np.log1p(df[var] + shift)
        transform_type = f"Log with shift (+{shift:.2f})"
    else:
        # Standard log transformation
        df_transformed[new_var_name] = np.log1p(df[var])
        transform_type = "Standard log(x+1)"
    
    # Calculate before/after skewness
    original_skew = stats.skew(df[var].dropna())
    transformed_skew = stats.skew(df_transformed[new_var_name].dropna())
    
    transformation_log.append({
        'Original_Variable': var,
        'Transformed_Variable': new_var_name,
        'Transform_Type': transform_type,
        'Original_Skewness': original_skew,
        'Transformed_Skewness': transformed_skew,
        'Improvement': abs(original_skew) - abs(transformed_skew)
    })
    
    print(f"  ✓ {var} → {new_var_name}")
    print(f"     Type: {transform_type}")
    print(f"     Skewness: {original_skew:.2f} → {transformed_skew:.2f} (Δ = {abs(original_skew) - abs(transformed_skew):.2f})")

# ============================================================================
# STEP 4: CREATE FINAL ML-READY DATASET
# ============================================================================

print("\n" + "="*80)
print("CREATING ML-READY DATASET")
print("="*80)

# Build list of columns for ML (replace transformed variables)
ml_columns = []
for var in original_predictors:
    if var in variables_to_transform:
        # Use transformed version
        ml_columns.append(f"{var}_log")
    else:
        # Use original version
        ml_columns.append(var)

# Filter to only include columns that exist
ml_columns_existing = [col for col in ml_columns if col in df_transformed.columns]

print(f"\nTotal predictors: {len(original_predictors)}")
print(f"Variables transformed: {len(variables_to_transform)}")
print(f"Final ML columns: {len(ml_columns_existing)}")

# Create ML dataset
df_ml = df_transformed[ml_columns_existing].copy()

print("\nML-Ready Dataset Columns:")
for i, col in enumerate(ml_columns_existing, 1):
    marker = " [LOG-TRANSFORMED]" if "_log" in col else ""
    print(f"{i:2d}. {col}{marker}")

# ============================================================================
# STEP 5: DATA QUALITY REPORT
# ============================================================================

print("\n" + "="*80)
print("DATA QUALITY REPORT")
print("="*80)

quality_report = []
for col in ml_columns_existing:
    n_total = len(df_ml)
    n_missing = df_ml[col].isna().sum()
    n_valid = n_total - n_missing
    pct_missing = (n_missing / n_total) * 100
    
    quality_report.append({
        'Variable': col,
        'Total_Rows': n_total,
        'Valid': n_valid,
        'Missing': n_missing,
        'Pct_Missing': pct_missing
    })

quality_df = pd.DataFrame(quality_report).sort_values('Pct_Missing', ascending=False)
print("\nMissing Data Summary (sorted by % missing):")
print(quality_df.head(10).to_string(index=False))

high_missing = quality_df[quality_df['Pct_Missing'] > 10]
if len(high_missing) > 0:
    print(f"\n⚠️  WARNING: {len(high_missing)} variables have >10% missing data:")
    print(high_missing[['Variable', 'Pct_Missing']].to_string(index=False))

# ============================================================================
# STEP 6: SAVE DATASETS
# ============================================================================

print("\n" + "="*80)
print("SAVING DATASETS")
print("="*80)

base_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/'

# Save full transformed dataset (all original columns + log columns)
output_full = f'{base_path}combined_daily_data_with_log_transforms.csv'
df_transformed.to_csv(output_full, index=False)
print(f"✓ Full dataset with log transforms: {output_full}")
print(f"   Shape: {df_transformed.shape}")

# Save ML-ready dataset (only selected predictors with transformations applied)
output_ml = f'{base_path}ml_ready_dataset_transformed.csv'
df_ml.to_csv(output_ml, index=False)
print(f"✓ ML-ready dataset: {output_ml}")
print(f"   Shape: {df_ml.shape}")

# Save transformation log
output_log = f'{base_path}anova/transformation_log.csv'
pd.DataFrame(transformation_log).to_csv(output_log, index=False)
print(f"✓ Transformation log: {output_log}")

# Save skewness report
output_skew = f'{base_path}anova/skewness_report.csv'
skew_df.to_csv(output_skew, index=False)
print(f"✓ Skewness report: {output_skew}")

# Save data quality report
output_quality = f'{base_path}anova/data_quality_report.csv'
quality_df.to_csv(output_quality, index=False)
print(f"✓ Data quality report: {output_quality}")

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("TRANSFORMATION SUMMARY")
print("="*80)

trans_df = pd.DataFrame(transformation_log)
if len(trans_df) > 0:
    print(f"\nTotal transformations applied: {len(trans_df)}")
    print(f"Average skewness improvement: {trans_df['Improvement'].mean():.2f}")
    print(f"Best improvement: {trans_df['Improvement'].max():.2f} ({trans_df.loc[trans_df['Improvement'].idxmax(), 'Original_Variable']})")
    
    print("\nTransformation Details:")
    print(trans_df[['Original_Variable', 'Original_Skewness', 'Transformed_Skewness', 'Improvement']].to_string(index=False))

# ============================================================================
# STEP 8: CREATE STRESS AND ANXIETY GROUPS FOR ANOVA
# ============================================================================

print("\n" + "="*80)
print("RUNNING ANOVA ANALYSIS WITH TRANSFORMED VARIABLES")
print("="*80)

def create_groups(data, column, new_column_name):
    """Create Low/Medium/High groups based on tertiles (33%, 66%)"""
    tertiles = data[column].quantile([0.33, 0.67])
    
    conditions = [
        data[column] <= tertiles.iloc[0],
        (data[column] > tertiles.iloc[0]) & (data[column] <= tertiles.iloc[1]),
        data[column] > tertiles.iloc[1]
    ]
    choices = ['Low', 'Medium', 'High']
    
    data[new_column_name] = np.select(conditions, choices, default='Unknown')
    return data

df_transformed = create_groups(df_transformed, 'stress_level', 'stress_group')
df_transformed = create_groups(df_transformed, 'anxiety_level', 'anxiety_group')

print("✓ Created stress and anxiety groups")

# ============================================================================
# STEP 9: ANOVA ANALYSIS FUNCTIONS
# ============================================================================

from scipy.stats import levene, f_oneway
from statsmodels.stats.multitest import multipletests

def welch_anova(groups):
    """Welch's ANOVA for unequal variances"""
    groups_clean = [g.dropna() for g in groups if len(g.dropna()) > 0]
    
    if len(groups_clean) < 2:
        return np.nan, np.nan
    
    k = len(groups_clean)
    n_i = np.array([len(g) for g in groups_clean])
    mean_i = np.array([g.mean() for g in groups_clean])
    var_i = np.array([g.var(ddof=1) for g in groups_clean])
    w_i = n_i / var_i
    
    grand_mean = np.sum(w_i * mean_i) / np.sum(w_i)
    numerator = np.sum(w_i * (mean_i - grand_mean)**2) / (k - 1)
    tmp = (1 - w_i/np.sum(w_i))**2 / (n_i - 1)
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum(tmp)
    
    f_stat = numerator / denominator
    df1 = k - 1
    df2 = 1 / (3 * np.sum(tmp) / (k**2 - 1))
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return f_stat, p_value

def calculate_eta_squared(groups):
    """Calculate eta-squared effect size"""
    groups_clean = [g.dropna() for g in groups if len(g.dropna()) > 0]
    
    if len(groups_clean) < 2:
        return np.nan
    
    grand_mean = np.concatenate(groups_clean).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_clean)
    all_data = np.concatenate(groups_clean)
    ss_total = np.sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    
    return eta_squared

def run_anova_analysis(data, group_column, predictor_vars, analysis_name):
    """Run comprehensive ANOVA analysis"""
    results = []
    
    print(f"\nRunning {analysis_name} ANOVA...")
    
    for var in predictor_vars:
        if var not in data.columns:
            print(f"  Skipping {var} (not found)")
            continue
        
        groups = [data[data[group_column] == level][var].dropna() 
                  for level in ['Low', 'Medium', 'High']]
        
        valid_groups = [g for g in groups if len(g) >= 2]
        if len(valid_groups) < 2:
            print(f"  Skipping {var} (insufficient data)")
            continue
        
        group_means = [g.mean() for g in groups]
        group_stds = [g.std() for g in groups]
        group_ns = [len(g) for g in groups]
        
        levene_stat, levene_p = levene(*groups)
        homogeneity = levene_p > 0.05
        
        if homogeneity:
            f_stat, p_value = f_oneway(*groups)
            anova_type = "Standard ANOVA"
        else:
            f_stat, p_value = welch_anova(groups)
            anova_type = "Welch's ANOVA"
        
        eta_sq = calculate_eta_squared(groups)
        
        results.append({
            'Variable': var,
            'ANOVA_Type': anova_type,
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Levene_P': levene_p,
            'Homogeneity': homogeneity,
            'Eta_Squared': eta_sq,
            'Low_Mean': group_means[0],
            'Low_SD': group_stds[0],
            'Low_N': group_ns[0],
            'Medium_Mean': group_means[1],
            'Medium_SD': group_stds[1],
            'Medium_N': group_ns[1],
            'High_Mean': group_means[2],
            'High_SD': group_stds[2],
            'High_N': group_ns[2]
        })
    
    print(f"  ✓ Completed {len(results)} tests")
    return pd.DataFrame(results)

# ============================================================================
# STEP 10: RUN ANOVA WITH TRANSFORMED VARIABLES
# ============================================================================

# Get predictor list (excluding target variables for ANOVA)
anova_predictors = [col for col in ml_columns_existing if col not in ['stress_level', 'anxiety_level']]

print(f"\nRunning ANOVA on {len(anova_predictors)} transformed predictors")

# Run stress group ANOVA
stress_results_new = run_anova_analysis(df_transformed, 'stress_group', anova_predictors, 'Stress-Group')

if len(stress_results_new) > 0:
    rejected, corrected_p, _, _ = multipletests(
        stress_results_new['P_Value'], 
        alpha=0.05, 
        method='fdr_bh'
    )
    stress_results_new['P_Value_Corrected'] = corrected_p
    stress_results_new['Significant_FDR'] = rejected

# Run anxiety group ANOVA
anxiety_results_new = run_anova_analysis(df_transformed, 'anxiety_group', anova_predictors, 'Anxiety-Group')

if len(anxiety_results_new) > 0:
    rejected, corrected_p, _, _ = multipletests(
        anxiety_results_new['P_Value'], 
        alpha=0.05, 
        method='fdr_bh'
    )
    anxiety_results_new['P_Value_Corrected'] = corrected_p
    anxiety_results_new['Significant_FDR'] = rejected

# ============================================================================
# STEP 11: SAVE ANOVA RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING ANOVA RESULTS")
print("="*80)

anova_path = f'{base_path}anova/'

# Save new ANOVA results
stress_anova_file = f'{anova_path}stress_anova_results_new_variables.csv'
anxiety_anova_file = f'{anova_path}anxiety_anova_results_new_variables.csv'

stress_results_new.to_csv(stress_anova_file, index=False)
anxiety_results_new.to_csv(anxiety_anova_file, index=False)

print(f"✓ Stress ANOVA results: {stress_anova_file}")
print(f"✓ Anxiety ANOVA results: {anxiety_anova_file}")

# ============================================================================
# STEP 12: DISPLAY KEY FINDINGS
# ============================================================================

print("\n" + "="*80)
print("ANOVA RESULTS SUMMARY - STRESS GROUPS")
print("="*80)

sig_stress = stress_results_new[stress_results_new['Significant_FDR'] == True].sort_values('Eta_Squared', ascending=False)
print(f"\nSignificant predictors (FDR corrected): {len(sig_stress)}")
if len(sig_stress) > 0:
    print("\nTop 10 by effect size:")
    display_cols = ['Variable', 'ANOVA_Type', 'P_Value_Corrected', 'Eta_Squared', 'Low_Mean', 'Medium_Mean', 'High_Mean']
    print(sig_stress[display_cols].head(10).to_string(index=False))

print("\n" + "="*80)
print("ANOVA RESULTS SUMMARY - ANXIETY GROUPS")
print("="*80)

sig_anxiety = anxiety_results_new[anxiety_results_new['Significant_FDR'] == True].sort_values('Eta_Squared', ascending=False)
print(f"\nSignificant predictors (FDR corrected): {len(sig_anxiety)}")
if len(sig_anxiety) > 0:
    print("\nTop 10 by effect size:")
    print(sig_anxiety[display_cols].head(10).to_string(index=False))

# ============================================================================
# STEP 13: HIGHLIGHT NEW FINDINGS
# ============================================================================

print("\n" + "="*80)
print("NEW VARIABLES - INITIAL FINDINGS")
print("="*80)

new_vars_in_analysis = [
    'last_sleep_stage',
    'respiratory_rate_summary_rem_sleep_signal_to_noise',
    'rem_avg_segment_duration_minutes',
    'minute_spo2_value_min'
]

print("\nChecking significance of newly added variables:")
for var in new_vars_in_analysis:
    # Check stress
    stress_row = stress_results_new[stress_results_new['Variable'] == var]
    if len(stress_row) > 0:
        sig = "✓ SIGNIFICANT" if stress_row['Significant_FDR'].values[0] else "✗ Not significant"
        p_val = stress_row['P_Value_Corrected'].values[0]
        eta = stress_row['Eta_Squared'].values[0]
        print(f"\n{var} (STRESS):")
        print(f"  {sig} | p={p_val:.4f}, η²={eta:.4f}")
    
    # Check anxiety
    anxiety_row = anxiety_results_new[anxiety_results_new['Variable'] == var]
    if len(anxiety_row) > 0:
        sig = "✓ SIGNIFICANT" if anxiety_row['Significant_FDR'].values[0] else "✗ Not significant"
        p_val = anxiety_row['P_Value_Corrected'].values[0]
        eta = anxiety_row['Eta_Squared'].values[0]
        print(f"{var} (ANXIETY):")
        print(f"  {sig} | p={p_val:.4f}, η²={eta:.4f}")

print("\n" + "="*80)
print("READY FOR ML MODELING!")
print("="*80)
print("\nNext steps:")
print("1. Load the ML-ready dataset:")
print(f"   df = pd.read_csv('{output_ml}')")
print("\n2. Split features and targets:")
print("   X = df.drop(['stress_level', 'anxiety_level'], axis=1)")
print("   y_stress = df['stress_level']")
print("   y_anxiety = df['anxiety_level']")
print("\n3. Handle missing data (if any):")
print("   from sklearn.impute import SimpleImputer")
print("   imputer = SimpleImputer(strategy='median')")
print("   X_imputed = imputer.fit_transform(X)")
print("\n4. Split train/test and start modeling!")
print("   from sklearn.model_selection import train_test_split")
print("   X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_stress, test_size=0.2)")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETE!")
print("="*80)