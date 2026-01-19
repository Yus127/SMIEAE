import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STRESS PREDICTION WITH PROPER DATA NORMALIZATION")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"
output_dir = "/Users/YusMolina/Downloads/smieae/results/ml_models_normalized"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "shap"), exist_ok=True)

# Load data
print("\n1. LOADING DATA")
print("-"*80)

data_file = os.path.join(ml_dir, "ml_ready_combined_windows_enriched.csv")
if not os.path.exists(data_file):
    data_file = os.path.join("/Users/YusMolina/Downloads/smieae/data/ml_ready", 
                            "ml_ready_combined_windows_enriched.csv")

print(f"Loading: {data_file}")
df = pd.read_csv(data_file)
print(f"✓ Loaded {len(df)} observations")

# Define feature columns
feature_columns = [
    'is_exam_period',
    'days_until_exam',
    'is_pre_exam_week',
    'is_easter_break',
    'daily_total_steps',
    # Heart rate features - 30 min window
    'w30_heart_rate_activity_beats per minute_mean',
    'w30_heart_rate_activity_beats per minute_std',
    'w30_heart_rate_activity_beats per minute_min',
    'w30_heart_rate_activity_beats per minute_max',
    'w30_heart_rate_activity_beats per minute_median',
    # Heart rate features - 1 hour window
    'w60_heart_rate_activity_beats per minute_mean',
    'w60_heart_rate_activity_beats per minute_std',
    'w60_heart_rate_activity_beats per minute_min',
    'w60_heart_rate_activity_beats per minute_max',
    'w60_heart_rate_activity_beats per minute_median',
]

available_features = [col for col in feature_columns if col in df.columns]
print(f"\n✓ Using {len(available_features)} features")

# Target
target_columns = [col for col in df.columns if col.startswith('q_i_stress') or col.startswith('q_i_anxiety')]
if 'q_i_stress_sliderNeutralPos' in df.columns:
    primary_target = 'q_i_stress_sliderNeutralPos'
else:
    stress_cols = [col for col in target_columns if 'stress' in col.lower()]
    primary_target = stress_cols[0] if stress_cols else target_columns[0]

print(f"Target variable: {primary_target}")

# 2. DATA PREPROCESSING
print("\n2. DATA PREPROCESSING")
print("-"*80)

df_clean = df[available_features + [primary_target]].dropna(subset=[primary_target])

# Impute missing features with median
for col in available_features:
    if df_clean[col].isna().any():
        median_val = df_clean[col].median()
        missing_count = df_clean[col].isna().sum()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"  Imputed {col}: {missing_count} missing values with median {median_val:.2f}")

print(f"\n✓ Clean dataset: {len(df_clean)} observations")

# 3. EXPLORATORY DATA ANALYSIS - BEFORE NORMALIZATION
print("\n3. EXPLORATORY DATA ANALYSIS - FEATURE DISTRIBUTIONS")
print("-"*80)

X_raw = df_clean[available_features].copy()
y_continuous = df_clean[primary_target].copy()

# Show statistics before normalization
print("\nFeature Statistics (BEFORE Normalization):")
print("-"*80)
stats_before = X_raw.describe().T[['mean', 'std', 'min', 'max']]
print(stats_before.to_string())

# Visualize distributions before normalization
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.ravel()

for idx, col in enumerate(available_features[:15]):  # Plot first 15 features
    axes[idx].hist(X_raw[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{col}\n(mean={X_raw[col].mean():.2f}, std={X_raw[col].std():.2f})', 
                       fontsize=8)
    axes[idx].set_xlabel('')
    axes[idx].tick_params(labelsize=7)

# Hide unused subplots
for idx in range(len(available_features), 16):
    axes[idx].axis('off')

plt.suptitle('Feature Distributions BEFORE Normalization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'distributions_before_normalization.png'), 
           dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: distributions_before_normalization.png")
plt.close()

# 4. CREATE TARGET VARIABLE (3-CLASS)
print("\n4. TARGET VARIABLE CREATION")
print("-"*80)

# Calculate tertiles
percentile_33 = np.percentile(y_continuous, 33.33)
percentile_67 = np.percentile(y_continuous, 66.67)

# Create 3 classes
y = pd.cut(y_continuous, 
           bins=[-np.inf, percentile_33, percentile_67, np.inf],
           labels=[0, 1, 2],
           include_lowest=True).astype(int)

print(f"Continuous range: [{y_continuous.min():.2f}, {y_continuous.max():.2f}]")
print(f"33rd percentile: {percentile_33:.2f}")
print(f"67th percentile: {percentile_67:.2f}")
print(f"\nClass distribution:")
print(f"  Class 0 (Low):    {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  Class 1 (Medium): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  Class 2 (High):   {(y==2).sum()} ({(y==2).sum()/len(y)*100:.1f}%)")

# 5. TRAIN-VALIDATION-TEST SPLIT (70-15-15)
print("\n5. DATA SPLIT: 70% TRAIN - 15% VALIDATION - 15% TEST")
print("-"*80)

# First split: 70% train, 30% temp
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.30, random_state=42, stratify=y
)

# Second split: 15% validation, 15% test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Total: {len(X_raw)} observations")
print(f"\nTraining:   {len(X_train_raw)} ({len(X_train_raw)/len(X_raw)*100:.1f}%)")
print(f"  Low: {(y_train==0).sum()}, Medium: {(y_train==1).sum()}, High: {(y_train==2).sum()}")
print(f"Validation: {len(X_val_raw)} ({len(X_val_raw)/len(X_raw)*100:.1f}%)")
print(f"  Low: {(y_val==0).sum()}, Medium: {(y_val==1).sum()}, High: {(y_val==2).sum()}")
print(f"Test:       {len(X_test_raw)} ({len(X_test_raw)/len(X_raw)*100:.1f}%)")
print(f"  Low: {(y_test==0).sum()}, Medium: {(y_test==1).sum()}, High: {(y_test==2).sum()}")

# 6. NORMALIZATION
print("\n6. DATA NORMALIZATION")
print("-"*80)
print("\nWe will test THREE normalization methods:")
print("  1. StandardScaler (Z-score): (X - mean) / std")
print("  2. MinMaxScaler (0-1 range): (X - min) / (max - min)")
print("  3. RobustScaler (median-based): (X - median) / IQR")

# 6.1 StandardScaler (Z-score normalization)
print("\n6.1 Applying StandardScaler (Z-score normalization)")
print("  ✓ Removes mean, scales to unit variance")
print("  ✓ Good for: Gaussian-distributed features")
print("  ✓ Sensitive to outliers")

scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train_raw)
X_val_standard = scaler_standard.transform(X_val_raw)
X_test_standard = scaler_standard.transform(X_test_raw)

# Convert to DataFrame for easier handling
X_train_standard_df = pd.DataFrame(X_train_standard, columns=available_features, index=X_train_raw.index)
X_val_standard_df = pd.DataFrame(X_val_standard, columns=available_features, index=X_val_raw.index)
X_test_standard_df = pd.DataFrame(X_test_standard, columns=available_features, index=X_test_raw.index)

# 6.2 MinMaxScaler
print("\n6.2 Applying MinMaxScaler (0-1 normalization)")
print("  ✓ Scales features to [0, 1] range")
print("  ✓ Good for: Bounded features, neural networks")
print("  ✓ Very sensitive to outliers")

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train_raw)
X_val_minmax = scaler_minmax.transform(X_val_raw)
X_test_minmax = scaler_minmax.transform(X_test_raw)

X_train_minmax_df = pd.DataFrame(X_train_minmax, columns=available_features, index=X_train_raw.index)
X_val_minmax_df = pd.DataFrame(X_val_minmax, columns=available_features, index=X_val_raw.index)
X_test_minmax_df = pd.DataFrame(X_test_minmax, columns=available_features, index=X_test_raw.index)

# 6.3 RobustScaler
print("\n6.3 Applying RobustScaler (IQR normalization)")
print("  ✓ Uses median and IQR (robust to outliers)")
print("  ✓ Good for: Data with outliers")
print("  ✓ Most robust method")

scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train_raw)
X_val_robust = scaler_robust.transform(X_val_raw)
X_test_robust = scaler_robust.transform(X_test_raw)

X_train_robust_df = pd.DataFrame(X_train_robust, columns=available_features, index=X_train_raw.index)
X_val_robust_df = pd.DataFrame(X_val_robust, columns=available_features, index=X_val_raw.index)
X_test_robust_df = pd.DataFrame(X_test_robust, columns=available_features, index=X_test_raw.index)

# Show statistics after normalization
print("\n" + "="*80)
print("NORMALIZATION COMPARISON")
print("="*80)

comparison_feature = 'w30_heart_rate_activity_beats per minute_mean'
if comparison_feature in available_features:
    print(f"\nExample feature: {comparison_feature}")
    print("-"*80)
    print(f"Original:     mean={X_train_raw[comparison_feature].mean():.2f}, std={X_train_raw[comparison_feature].std():.2f}")
    print(f"StandardScale: mean={X_train_standard_df[comparison_feature].mean():.2f}, std={X_train_standard_df[comparison_feature].std():.2f}")
    print(f"MinMaxScale:   mean={X_train_minmax_df[comparison_feature].mean():.2f}, std={X_train_minmax_df[comparison_feature].std():.2f}")
    print(f"RobustScale:   mean={X_train_robust_df[comparison_feature].mean():.2f}, std={X_train_robust_df[comparison_feature].std():.2f}")

# Visualize normalization effects
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].hist(X_train_raw[comparison_feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 0].set_title('Original Distribution', fontweight='bold')
axes[0, 0].set_xlabel(comparison_feature)
axes[0, 0].axvline(X_train_raw[comparison_feature].mean(), color='r', linestyle='--', label='mean')
axes[0, 0].legend()

# StandardScaler
axes[0, 1].hist(X_train_standard_df[comparison_feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
axes[0, 1].set_title('StandardScaler (Z-score)', fontweight='bold')
axes[0, 1].set_xlabel(f'{comparison_feature} (normalized)')
axes[0, 1].axvline(X_train_standard_df[comparison_feature].mean(), color='r', linestyle='--', label='mean≈0')
axes[0, 1].legend()

# MinMaxScaler
axes[1, 0].hist(X_train_minmax_df[comparison_feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 0].set_title('MinMaxScaler (0-1)', fontweight='bold')
axes[1, 0].set_xlabel(f'{comparison_feature} (normalized)')
axes[1, 0].axvline(X_train_minmax_df[comparison_feature].mean(), color='r', linestyle='--', label='mean')
axes[1, 0].legend()

# RobustScaler
axes[1, 1].hist(X_train_robust_df[comparison_feature].dropna(), bins=30, edgecolor='black', alpha=0.7, color='plum')
axes[1, 1].set_title('RobustScaler (IQR)', fontweight='bold')
axes[1, 1].set_xlabel(f'{comparison_feature} (normalized)')
axes[1, 1].axvline(X_train_robust_df[comparison_feature].mean(), color='r', linestyle='--', label='median≈0')
axes[1, 1].legend()

plt.suptitle(f'Normalization Methods Comparison\nFeature: {comparison_feature}', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'normalization_comparison.png'), 
           dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: normalization_comparison.png")
plt.close()

# 7. TRAIN MODELS WITH DIFFERENT NORMALIZATIONS
print("\n" + "="*80)
print("7. TRAINING MODELS WITH NORMALIZED DATA")
print("="*80)

# We'll primarily use StandardScaler (most common) but also test others
normalization_methods = {
    'StandardScaler': (X_train_standard_df, X_val_standard_df, X_test_standard_df),
    'MinMaxScaler': (X_train_minmax_df, X_val_minmax_df, X_test_minmax_df),
    'RobustScaler': (X_train_robust_df, X_val_robust_df, X_test_robust_df),
    'No Normalization': (X_train_raw, X_val_raw, X_test_raw)
}

# For the main analysis, we'll use StandardScaler (most common in ML)
X_train = X_train_standard_df
X_val = X_val_standard_df
X_test = X_test_standard_df

print("\nUsing StandardScaler for main analysis")
print("(This is the most common normalization method in ML)")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=6, 
                                 learning_rate=0.1, objective='multi:softmax', num_class=3),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, decision_function_shape='ovr'),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Note: Tree-based models don't strictly need normalization, but we apply it for consistency
    # Linear models (LR, SVM, NB) definitely benefit from normalization
    
    model.fit(X_train, y_train)
    
    # Validation predictions
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Validation metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1_per_class = f1_score(y_val, y_val_pred, average=None, zero_division=0)
    
    # Test metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0)
    
    # ROC AUC
    try:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        test_auc = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
    except:
        test_auc = np.nan
    
    print(f"  VALIDATION: Acc={val_accuracy:.4f}, F1={val_f1:.4f}")
    print(f"  TEST:       Acc={test_accuracy:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
    print(f"  Per-class F1 (test): Low={test_f1_per_class[0]:.4f}, Med={test_f1_per_class[1]:.4f}, High={test_f1_per_class[2]:.4f}")
    
    # Store results
    results[name] = {
        'model': model,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'val_f1_per_class': val_f1_per_class,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_f1_per_class': test_f1_per_class,
        'test_auc': test_auc,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }

# 8. COMPARE RESULTS
print("\n" + "="*80)
print("8. MODEL COMPARISON (WITH NORMALIZATION)")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Val Acc': [results[m]['val_accuracy'] for m in results],
    'Val F1': [results[m]['val_f1'] for m in results],
    'Test Acc': [results[m]['test_accuracy'] for m in results],
    'Test F1': [results[m]['test_f1'] for m in results],
    'Test AUC': [results[m]['test_auc'] for m in results],
    'F1 Low': [results[m]['test_f1_per_class'][0] for m in results],
    'F1 Med': [results[m]['test_f1_per_class'][1] for m in results],
    'F1 High': [results[m]['test_f1_per_class'][2] for m in results]
})

comparison_df = comparison_df.sort_values('Val F1', ascending=False)
print("\n" + comparison_df.to_string(index=False))

comparison_df.to_csv(os.path.join(output_dir, "model_comparison_normalized.csv"), index=False)

best_model_name = comparison_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Validation F1: {comparison_df.iloc[0]['Val F1']:.4f}")
print(f"   Test F1:       {comparison_df.iloc[0]['Test F1']:.4f}")

# 9. SAVE NORMALIZED DATA
print("\n" + "="*80)
print("9. SAVING NORMALIZED DATASETS")
print("="*80)

# Save normalized data for future use
normalized_train = X_train_standard_df.copy()
normalized_train['target'] = y_train.values
normalized_train.to_csv(os.path.join(output_dir, 'train_normalized.csv'), index=False)

normalized_val = X_val_standard_df.copy()
normalized_val['target'] = y_val.values
normalized_val.to_csv(os.path.join(output_dir, 'validation_normalized.csv'), index=False)

normalized_test = X_test_standard_df.copy()
normalized_test['target'] = y_test.values
normalized_test.to_csv(os.path.join(output_dir, 'test_normalized.csv'), index=False)

print("✓ Saved: train_normalized.csv")
print("✓ Saved: validation_normalized.csv")
print("✓ Saved: test_normalized.csv")

# Save scaler for future use
import joblib
joblib.dump(scaler_standard, os.path.join(output_dir, 'scaler_standard.pkl'))
print("✓ Saved: scaler_standard.pkl (for future predictions)")

# 10. FINAL SUMMARY
print("\n" + "="*80)
print("10. SUMMARY")
print("="*80)

print(f"\n✅ DATA PROPERLY NORMALIZED using StandardScaler")
print(f"   • All features scaled to mean=0, std=1")
print(f"   • Training set statistics used (no data leakage)")
print(f"   • Same transformation applied to validation and test")

print(f"\n📊 Dataset: {len(df_clean)} total observations")
print(f"   • Train:      {len(X_train)} (70%)")
print(f"   • Validation: {len(X_val)} (15%)")
print(f"   • Test:       {len(X_test)} (15%)")

print(f"\n🎯 Best Model: {best_model_name}")
print(f"   • Validation F1: {results[best_model_name]['val_f1']:.4f}")
print(f"   • Test F1:       {results[best_model_name]['test_f1']:.4f}")
print(f"   • Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")

print(f"\n📁 Results saved to: {output_dir}")

print("\n" + "="*80)
print("✅ COMPLETE! Data normalized and models trained.")
print("="*80)