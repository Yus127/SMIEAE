import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# PATHS
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/'

print("="*80)
print("COMPREHENSIVE MODEL ANALYSIS & VISUALIZATION")
print("="*80)

# Load data
df = pd.read_csv(INPUT_PATH)
print(f"\nDataset: {len(df)} rows, {len(df.columns)} columns")

# Feature columns
feature_columns = [
    'daily_total_steps', 'activity_level_sedentary_count',
    'daily_respiratory_rate_daily_respiratory_rate', 'minute_spo2_value_mean',
    'daily_hrv_summary_rmssd', 'hrv_details_rmssd_min',
    'sleep_global_duration', 'sleep_global_efficiency',
    'deep_sleep_minutes', 'rem_sleep_minutes', 'sleep_stage_transitions',
    'daily_hrv_summary_entropy', 'heart_rate_activity_beats per minute_mean',
    'is_exam_period', 'is_semana_santa', 'days_to_next_exam',
    'days_since_last_exam', 'weeks_to_next_exam', 'weeks_since_last_exam'
]

def create_target_classes(series, p33, p67):
    classes = pd.cut(series, bins=[-np.inf, p33, p67, np.inf], labels=[0, 1, 2])
    return classes.astype(int)

def prepare_data(df, target_col, feature_cols):
    df_clean = df[df[target_col].notna()].copy()
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    y = create_target_classes(df_clean[target_col], p33, p67)
    X = df_clean[feature_cols].copy()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, p33, p67

# ============================================================================
# PREPARE DATA FOR BOTH TARGETS
# ============================================================================

print("\nPreparing data...")
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, p33_s, p67_s = prepare_data(df, 'stress_level', feature_columns)
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a, p33_a, p67_a = prepare_data(df, 'anxiety_level', feature_columns)

# ============================================================================
# TRAIN BEST MODELS (XGBoost)
# ============================================================================

print("\nTraining XGBoost models...")

# Stress model
stress_model = XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    eval_metric='mlogloss', use_label_encoder=False
)
stress_model.fit(X_train_s, y_train_s)

# Anxiety model
anxiety_model = XGBClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    eval_metric='mlogloss', use_label_encoder=False
)
anxiety_model.fit(X_train_a, y_train_a)

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRICES
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices: XGBoost Models', fontsize=16, fontweight='bold')

# Stress - Train
y_pred_train_s = stress_model.predict(X_train_s)
cm_train_s = confusion_matrix(y_train_s, y_pred_train_s)
sns.heatmap(cm_train_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,0], 
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,0].set_title('Stress - Train Set')
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

# Stress - Val
y_pred_val_s = stress_model.predict(X_val_s)
cm_val_s = confusion_matrix(y_val_s, y_pred_val_s)
sns.heatmap(cm_val_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,1].set_title('Stress - Validation Set')
axes[0,1].set_ylabel('True Label')
axes[0,1].set_xlabel('Predicted Label')

# Stress - Test
y_pred_test_s = stress_model.predict(X_test_s)
cm_test_s = confusion_matrix(y_test_s, y_pred_test_s)
sns.heatmap(cm_test_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,2],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,2].set_title('Stress - Test Set')
axes[0,2].set_ylabel('True Label')
axes[0,2].set_xlabel('Predicted Label')

# Anxiety - Train
y_pred_train_a = anxiety_model.predict(X_train_a)
cm_train_a = confusion_matrix(y_train_a, y_pred_train_a)
sns.heatmap(cm_train_a, annot=True, fmt='d', cmap='Greens', ax=axes[1,0],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1,0].set_title('Anxiety - Train Set')
axes[1,0].set_ylabel('True Label')
axes[1,0].set_xlabel('Predicted Label')

# Anxiety - Val
y_pred_val_a = anxiety_model.predict(X_val_a)
cm_val_a = confusion_matrix(y_val_a, y_pred_val_a)
sns.heatmap(cm_val_a, annot=True, fmt='d', cmap='Greens', ax=axes[1,1],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1,1].set_title('Anxiety - Validation Set')
axes[1,1].set_ylabel('True Label')
axes[1,1].set_xlabel('Predicted Label')

# Anxiety - Test
y_pred_test_a = anxiety_model.predict(X_test_a)
cm_test_a = confusion_matrix(y_test_a, y_pred_test_a)
sns.heatmap(cm_test_a, annot=True, fmt='d', cmap='Greens', ax=axes[1,2],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1,2].set_title('Anxiety - Test Set')
axes[1,2].set_ylabel('True Label')
axes[1,2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices.png")

# ============================================================================
# VISUALIZATION 2: FEATURE IMPORTANCE
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Stress feature importance
stress_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': stress_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

axes[0].barh(stress_importance['feature'], stress_importance['importance'], color='steelblue')
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Top 15 Features - Stress Prediction', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Anxiety feature importance
anxiety_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': anxiety_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

axes[1].barh(anxiety_importance['feature'], anxiety_importance['importance'], color='seagreen')
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Top 15 Features - Anxiety Prediction', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance.png")

# ============================================================================
# VISUALIZATION 3: PERFORMANCE COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Stress - Accuracy across sets
sets = ['Train', 'Val', 'Test']
stress_accs = [
    (y_pred_train_s == y_train_s).mean(),
    (y_pred_val_s == y_val_s).mean(),
    (y_pred_test_s == y_test_s).mean()
]
axes[0,0].bar(sets, stress_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[0,0].set_ylabel('Accuracy', fontsize=12)
axes[0,0].set_title('Stress - Accuracy Across Datasets', fontsize=14, fontweight='bold')
axes[0,0].set_ylim([0, 1])
axes[0,0].grid(axis='y', alpha=0.3)
for i, v in enumerate(stress_accs):
    axes[0,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

# Anxiety - Accuracy across sets
anxiety_accs = [
    (y_pred_train_a == y_train_a).mean(),
    (y_pred_val_a == y_val_a).mean(),
    (y_pred_test_a == y_test_a).mean()
]
axes[0,1].bar(sets, anxiety_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
axes[0,1].set_ylabel('Accuracy', fontsize=12)
axes[0,1].set_title('Anxiety - Accuracy Across Datasets', fontsize=14, fontweight='bold')
axes[0,1].set_ylim([0, 1])
axes[0,1].grid(axis='y', alpha=0.3)
for i, v in enumerate(anxiety_accs):
    axes[0,1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

# Stress - Per-class accuracy (test set)
stress_per_class = cm_test_s.diagonal() / cm_test_s.sum(axis=1)
classes = ['Low\n(0-33%)', 'Medium\n(33-67%)', 'High\n(67-100%)']
axes[1,0].bar(classes, stress_per_class, color=['#90EE90', '#FFD700', '#FF6B6B'], alpha=0.7)
axes[1,0].set_ylabel('Accuracy', fontsize=12)
axes[1,0].set_title('Stress - Per-Class Accuracy (Test Set)', fontsize=14, fontweight='bold')
axes[1,0].set_ylim([0, 1])
axes[1,0].grid(axis='y', alpha=0.3)
for i, v in enumerate(stress_per_class):
    axes[1,0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

# Anxiety - Per-class accuracy (test set)
anxiety_per_class = cm_test_a.diagonal() / cm_test_a.sum(axis=1)
axes[1,1].bar(classes, anxiety_per_class, color=['#90EE90', '#FFD700', '#FF6B6B'], alpha=0.7)
axes[1,1].set_ylabel('Accuracy', fontsize=12)
axes[1,1].set_title('Anxiety - Per-Class Accuracy (Test Set)', fontsize=14, fontweight='bold')
axes[1,1].set_ylim([0, 1])
axes[1,1].grid(axis='y', alpha=0.3)
for i, v in enumerate(anxiety_per_class):
    axes[1,1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'performance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: performance_comparison.png")

# ============================================================================
# DETAILED ANALYSIS REPORT
# ============================================================================

print("\n" + "="*80)
print("DETAILED ANALYSIS REPORT")
print("="*80)

print("\n" + "-"*80)
print("STRESS PREDICTION - XGBoost (Regularized)")
print("-"*80)
print(f"\nTest Set Performance:")
print(f"  Overall Accuracy: {stress_accs[2]:.4f} ({stress_accs[2]*100:.2f}%)")
print(f"  Low Class Accuracy: {stress_per_class[0]:.4f} ({stress_per_class[0]*100:.2f}%)")
print(f"  Medium Class Accuracy: {stress_per_class[1]:.4f} ({stress_per_class[1]*100:.2f}%)")
print(f"  High Class Accuracy: {stress_per_class[2]:.4f} ({stress_per_class[2]*100:.2f}%)")
print(f"\nGeneralization:")
print(f"  Train-Test Gap: {stress_accs[0] - stress_accs[2]:.4f}")
print(f"  Overfitting Status: {'Acceptable' if stress_accs[0] - stress_accs[2] < 0.2 else 'Significant'}")

print("\n" + "-"*80)
print("ANXIETY PREDICTION - XGBoost (Regularized)")
print("-"*80)
print(f"\nTest Set Performance:")
print(f"  Overall Accuracy: {anxiety_accs[2]:.4f} ({anxiety_accs[2]*100:.2f}%)")
print(f"  Low Class Accuracy: {anxiety_per_class[0]:.4f} ({anxiety_per_class[0]*100:.2f}%)")
print(f"  Medium Class Accuracy: {anxiety_per_class[1]:.4f} ({anxiety_per_class[1]*100:.2f}%)")
print(f"  High Class Accuracy: {anxiety_per_class[2]:.4f} ({anxiety_per_class[2]*100:.2f}%)")
print(f"\nGeneralization:")
print(f"  Train-Test Gap: {anxiety_accs[0] - anxiety_accs[2]:.4f}")
print(f"  Overfitting Status: {'Acceptable' if anxiety_accs[0] - anxiety_accs[2] < 0.2 else 'Significant'}")

print("\n" + "-"*80)
print("TOP 5 MOST IMPORTANT FEATURES")
print("-"*80)
print("\nFor Stress Prediction:")
for idx, row in stress_importance.tail(5).iloc[::-1].iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\nFor Anxiety Prediction:")
for idx, row in anxiety_importance.tail(5).iloc[::-1].iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
print("Files created:")
print("  - confusion_matrices.png")
print("  - feature_importance.png")
print("  - performance_comparison.png")