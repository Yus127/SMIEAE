import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Try different models
try:
    import lightgbm as lgb
    HAS_LGBM = True
except:
    HAS_LGBM = False

print("="*80)
print("ENHANCED ML MODEL: STRESS & ANXIETY PREDICTION")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/7] Loading data...")
df = pd.read_csv('/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data_with_log_transforms.csv')
print(f"✓ Dataset loaded: {df.shape}")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n[2/7] Engineering new features...")

# Separate features and targets first
X_original = df.drop(['stress_level', 'anxiety_level'], axis=1)
y_stress = df['stress_level']
y_anxiety = df['anxiety_level']

# Remove invalid target values
valid_mask = (~y_stress.isna()) & (~y_anxiety.isna()) & (~np.isinf(y_stress)) & (~np.isinf(y_anxiety))
X_original = X_original[valid_mask]
y_stress = y_stress[valid_mask]
y_anxiety = y_anxiety[valid_mask]

print(f"✓ Valid samples: {len(X_original)}")

# Create engineered features
X_engineered = X_original.copy()

# 1. Activity ratios
if 'daily_total_steps' in X_engineered.columns and 'activity_level_sedentary_count' in X_engineered.columns:
    X_engineered['activity_sedentary_ratio'] = X_engineered['activity_level_sedentary_count'] / (X_engineered['daily_total_steps'] + 1)
    print("  ✓ Created activity_sedentary_ratio")

# 2. Heart rate range and variability
hr_cols = [col for col in X_engineered.columns if 'heart_rate_activity' in col and not '_log' in col]
if len(hr_cols) >= 2:
    if 'heart_rate_activity_beats per minute_max' in X_engineered.columns and 'heart_rate_activity_beats per minute_min' in X_engineered.columns:
        X_engineered['heart_rate_range'] = X_engineered['heart_rate_activity_beats per minute_max'] - X_engineered['heart_rate_activity_beats per minute_min']
        print("  ✓ Created heart_rate_range")

# 3. Sleep efficiency indicators (if not using log version)
if 'sleep_global_duration' in X_engineered.columns and 'sleep_global_minutesAwake_log' in X_engineered.columns:
    # Convert log back for this calculation
    minutesAwake = np.expm1(X_engineered['sleep_global_minutesAwake_log'])
    X_engineered['sleep_quality_score'] = X_engineered['sleep_global_duration'] / (minutesAwake + 1)
    print("  ✓ Created sleep_quality_score")

# 4. HRV complexity
if 'hrv_details_rmssd_std_log' in X_engineered.columns and 'hrv_details_rmssd_mean' in X_engineered.columns:
    # Coefficient of variation for HRV
    rmssd_std = np.expm1(X_engineered['hrv_details_rmssd_std_log'])
    X_engineered['hrv_coefficient_variation'] = rmssd_std / (X_engineered['hrv_details_rmssd_mean'] + 1)
    print("  ✓ Created hrv_coefficient_variation")

# 5. Physiological stress index (combine HR and respiratory rate)
if 'heart_rate_activity_beats per minute_mean' in X_engineered.columns and 'daily_respiratory_rate_daily_respiratory_rate' in X_engineered.columns:
    # Normalize and combine (higher values = more physiological arousal)
    hr_norm = (X_engineered['heart_rate_activity_beats per minute_mean'] - X_engineered['heart_rate_activity_beats per minute_mean'].mean()) / X_engineered['heart_rate_activity_beats per minute_mean'].std()
    rr_norm = (X_engineered['daily_respiratory_rate_daily_respiratory_rate'] - X_engineered['daily_respiratory_rate_daily_respiratory_rate'].mean()) / X_engineered['daily_respiratory_rate_daily_respiratory_rate'].std()
    X_engineered['physiological_arousal_index'] = (hr_norm + rr_norm) / 2
    print("  ✓ Created physiological_arousal_index")

# 6. Activity level diversity (entropy-like measure)
activity_cols = [col for col in X_engineered.columns if 'activity_level_' in col and 'count' in col]
if len(activity_cols) >= 3:
    activity_total = X_engineered[activity_cols].sum(axis=1)
    activity_diversity = 0
    for col in activity_cols:
        prop = X_engineered[col] / (activity_total + 1)
        activity_diversity -= prop * np.log(prop + 1e-10)
    X_engineered['activity_diversity'] = activity_diversity
    print("  ✓ Created activity_diversity")

print(f"\n✓ Total features after engineering: {X_engineered.shape[1]} (added {X_engineered.shape[1] - X_original.shape[1]})")

# ============================================================================
# STEP 3: HANDLE MISSING DATA
# ============================================================================

print("\n[3/7] Handling missing data and infinities...")

# Replace infinities
X_engineered = X_engineered.replace([np.inf, -np.inf], np.nan)

# Impute
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_engineered),
    columns=X_engineered.columns,
    index=X_engineered.index
)

print(f"✓ Imputation complete")
print(f"  Remaining NaN: {X_imputed.isna().sum().sum()}")

# ============================================================================
# STEP 4: FEATURE SCALING
# ============================================================================

print("\n[4/7] Scaling features...")

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)

print("✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 5: TRY MULTIPLE MODELS
# ============================================================================

print("\n[5/7] Training multiple models...")

# Split data
X_train, X_test, y_train_stress, y_test_stress = train_test_split(
    X_scaled, y_stress, test_size=0.2, random_state=42
)
_, _, y_train_anxiety, y_test_anxiety = train_test_split(
    X_scaled, y_anxiety, test_size=0.2, random_state=42
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Define models to try
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
}

if HAS_LGBM:
    models['LightGBM'] = lgb.LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1)

results = []

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    
    # Train on stress
    model_stress = model
    model_stress.fit(X_train, y_train_stress)
    y_pred_stress = model_stress.predict(X_test)
    
    stress_r2 = r2_score(y_test_stress, y_pred_stress)
    stress_rmse = np.sqrt(mean_squared_error(y_test_stress, y_pred_stress))
    stress_mae = mean_absolute_error(y_test_stress, y_pred_stress)
    
    # Train on anxiety (need new instance)
    if model_name == 'Random Forest':
        model_anxiety = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=20, random_state=42, n_jobs=-1)
    elif model_name == 'Gradient Boosting':
        model_anxiety = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    else:
        model_anxiety = lgb.LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, verbose=-1)
    
    model_anxiety.fit(X_train, y_train_anxiety)
    y_pred_anxiety = model_anxiety.predict(X_test)
    
    anxiety_r2 = r2_score(y_test_anxiety, y_pred_anxiety)
    anxiety_rmse = np.sqrt(mean_squared_error(y_test_anxiety, y_pred_anxiety))
    anxiety_mae = mean_absolute_error(y_test_anxiety, y_pred_anxiety)
    
    results.append({
        'Model': model_name,
        'Stress_R2': stress_r2,
        'Stress_RMSE': stress_rmse,
        'Stress_MAE': stress_mae,
        'Anxiety_R2': anxiety_r2,
        'Anxiety_RMSE': anxiety_rmse,
        'Anxiety_MAE': anxiety_mae
    })
    
    print(f"    Stress R²: {stress_r2:.3f}, RMSE: {stress_rmse:.2f}")
    print(f"    Anxiety R²: {anxiety_r2:.3f}, RMSE: {anxiety_rmse:.2f}")

# ============================================================================
# STEP 6: RESULTS COMPARISON
# ============================================================================

print("\n[6/7] Comparing model performance...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("MODEL COMPARISON - STRESS PREDICTION")
print("="*80)
print(results_df[['Model', 'Stress_R2', 'Stress_RMSE', 'Stress_MAE']].to_string(index=False))

print("\n" + "="*80)
print("MODEL COMPARISON - ANXIETY PREDICTION")
print("="*80)
print(results_df[['Model', 'Anxiety_R2', 'Anxiety_RMSE', 'Anxiety_MAE']].to_string(index=False))

# Find best models
best_stress_model = results_df.loc[results_df['Stress_R2'].idxmax(), 'Model']
best_anxiety_model = results_df.loc[results_df['Anxiety_R2'].idxmax(), 'Model']

print(f"\n🏆 Best for Stress: {best_stress_model} (R² = {results_df['Stress_R2'].max():.3f})")
print(f"🏆 Best for Anxiety: {best_anxiety_model} (R² = {results_df['Anxiety_R2'].max():.3f})")

# ============================================================================
# STEP 7: CLASSIFICATION APPROACH (ALTERNATIVE)
# ============================================================================

print("\n[7/7] Trying classification approach (Low/Medium/High)...")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create categorical targets
def create_categories(y):
    tertiles = y.quantile([0.33, 0.67])
    categories = pd.cut(y, bins=[-np.inf, tertiles.iloc[0], tertiles.iloc[1], np.inf], 
                       labels=['Low', 'Medium', 'High'])
    return categories

y_stress_cat = create_categories(y_stress)
y_anxiety_cat = create_categories(y_anxiety)

# Split for classification
X_train_c, X_test_c, y_train_stress_c, y_test_stress_c = train_test_split(
    X_scaled, y_stress_cat, test_size=0.2, random_state=42
)
_, _, y_train_anxiety_c, y_test_anxiety_c = train_test_split(
    X_scaled, y_anxiety_cat, test_size=0.2, random_state=42
)

# Train classifiers
clf_stress = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
clf_stress.fit(X_train_c, y_train_stress_c)
y_pred_stress_c = clf_stress.predict(X_test_c)

clf_anxiety = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
clf_anxiety.fit(X_train_c, y_train_anxiety_c)
y_pred_anxiety_c = clf_anxiety.predict(X_test_c)

stress_acc = accuracy_score(y_test_stress_c, y_pred_stress_c)
anxiety_acc = accuracy_score(y_test_anxiety_c, y_pred_anxiety_c)

print(f"\n✓ Classification Results:")
print(f"  Stress classification accuracy: {stress_acc:.1%}")
print(f"  Anxiety classification accuracy: {anxiety_acc:.1%}")
print(f"\n  (Baseline = 33.3% if guessing randomly)")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================

base_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/anova/'
results_df.to_csv(f'{base_path}enhanced_model_comparison.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\n📊 KEY INSIGHTS:")
if results_df['Stress_R2'].max() < 0.10:
    print("\n⚠️  STRESS PREDICTION:")
    print("  • Very weak predictive power (R² < 10%)")
    print("  • Physiological variables alone are insufficient")
    print("  • Consider: temporal patterns, context, psychological factors")
else:
    print(f"\n✓ STRESS PREDICTION: R² = {results_df['Stress_R2'].max():.1%}")

if results_df['Anxiety_R2'].max() < 0.15:
    print("\n⚠️  ANXIETY PREDICTION:")
    print("  • Weak to moderate predictive power")
    print("  • Some physiological signal present but incomplete")
    print("  • Respiratory rate likely most important")
else:
    print(f"\n✓ ANXIETY PREDICTION: R² = {results_df['Anxiety_R2'].max():.1%}")

if stress_acc > 0.40 or anxiety_acc > 0.40:
    print(f"\n💡 CLASSIFICATION APPROACH:")
    print(f"  • Predicting categories works better than exact values")
    print(f"  • Stress: {stress_acc:.1%} accuracy")
    print(f"  • Anxiety: {anxiety_acc:.1%} accuracy")

print("\n💡 RECOMMENDATIONS:")
print("  1. Collect temporal/contextual features (time of day, day of week)")
print("  2. Use person-specific models (train separately per individual)")
print("  3. Consider time-series features (rolling averages, trends)")
print("  4. Combine physiological + behavioral + contextual data")
print("  5. Use classification (Low/Med/High) instead of regression")

print("\n" + "="*80)