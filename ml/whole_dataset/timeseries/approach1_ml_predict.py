import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
import os

# PATHS
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model1/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("="*80)
print("COMPREHENSIVE MODEL ANALYSIS & VISUALIZATION - TIME SERIES SPLIT")
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

def prepare_data_timeseries(df, target_col, feature_cols):
    """
    Prepare data with time series split (70-15-15)
    No shuffling to preserve temporal order
    """
    df_clean = df[df[target_col].notna()].copy()
    
    # Calculate percentiles for class creation
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    
    # Create target classes
    y = create_target_classes(df_clean[target_col], p33, p67)
    X = df_clean[feature_cols].copy()
    
    # Time series split: 70-15-15
    n_samples = len(X)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split sequentially (no shuffling for time series)
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size + val_size)
    test_idx = slice(train_size + val_size, None)
    
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]
    
    # Imputation
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"\n{target_col} - Split sizes:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
    
    print(f"\n{target_col} - Class distribution:")
    print(f"  Train: {np.bincount(y_train)}")
    print(f"  Val:   {np.bincount(y_val)}")
    print(f"  Test:  {np.bincount(y_test)}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, p33, p67

# ============================================================================
# PREPARE DATA FOR BOTH TARGETS
# ============================================================================

print("\nPreparing data with time series split...")
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, p33_s, p67_s = prepare_data_timeseries(df, 'stress_level', feature_columns)
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a, p33_a, p67_a = prepare_data_timeseries(df, 'anxiety_level', feature_columns)

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
# GENERATE PREDICTIONS AND PROBABILITIES
# ============================================================================

# Stress predictions
y_pred_train_s = stress_model.predict(X_train_s)
y_pred_val_s = stress_model.predict(X_val_s)
y_pred_test_s = stress_model.predict(X_test_s)

y_pred_proba_train_s = stress_model.predict_proba(X_train_s)
y_pred_proba_val_s = stress_model.predict_proba(X_val_s)
y_pred_proba_test_s = stress_model.predict_proba(X_test_s)

# Anxiety predictions
y_pred_train_a = anxiety_model.predict(X_train_a)
y_pred_val_a = anxiety_model.predict(X_val_a)
y_pred_test_a = anxiety_model.predict(X_test_a)

y_pred_proba_train_a = anxiety_model.predict_proba(X_train_a)
y_pred_proba_val_a = anxiety_model.predict_proba(X_val_a)
y_pred_proba_test_a = anxiety_model.predict_proba(X_test_a)

# ============================================================================
# CALCULATE METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics for multiclass classification"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # ROC AUC (one-vs-rest for multiclass)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        if y_true_bin.shape[1] == 3:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
            metrics['roc_auc_ovr_weighted'] = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
        else:
            metrics['roc_auc_ovr'] = np.nan
            metrics['roc_auc_ovr_weighted'] = np.nan
    except:
        metrics['roc_auc_ovr'] = np.nan
        metrics['roc_auc_ovr_weighted'] = np.nan
    
    return metrics

print("\nCalculating metrics...")

# Stress metrics
stress_metrics_train = calculate_metrics(y_train_s, y_pred_train_s, y_pred_proba_train_s)
stress_metrics_val = calculate_metrics(y_val_s, y_pred_val_s, y_pred_proba_val_s)
stress_metrics_test = calculate_metrics(y_test_s, y_pred_test_s, y_pred_proba_test_s)

# Anxiety metrics
anxiety_metrics_train = calculate_metrics(y_train_a, y_pred_train_a, y_pred_proba_train_a)
anxiety_metrics_val = calculate_metrics(y_val_a, y_pred_val_a, y_pred_proba_val_a)
anxiety_metrics_test = calculate_metrics(y_test_a, y_pred_test_a, y_pred_proba_test_a)

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRICES
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices: XGBoost Models (Time Series Split)', fontsize=16, fontweight='bold')

# Stress - Train
cm_train_s = confusion_matrix(y_train_s, y_pred_train_s)
sns.heatmap(cm_train_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,0], 
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,0].set_title('Stress - Train Set')
axes[0,0].set_ylabel('True Label')
axes[0,0].set_xlabel('Predicted Label')

# Stress - Val
cm_val_s = confusion_matrix(y_val_s, y_pred_val_s)
sns.heatmap(cm_val_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,1].set_title('Stress - Validation Set')
axes[0,1].set_ylabel('True Label')
axes[0,1].set_xlabel('Predicted Label')

# Stress - Test
cm_test_s = confusion_matrix(y_test_s, y_pred_test_s)
sns.heatmap(cm_test_s, annot=True, fmt='d', cmap='Blues', ax=axes[0,2],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[0,2].set_title('Stress - Test Set')
axes[0,2].set_ylabel('True Label')
axes[0,2].set_xlabel('Predicted Label')

# Anxiety - Train
cm_train_a = confusion_matrix(y_train_a, y_pred_train_a)
sns.heatmap(cm_train_a, annot=True, fmt='d', cmap='Greens', ax=axes[1,0],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1,0].set_title('Anxiety - Train Set')
axes[1,0].set_ylabel('True Label')
axes[1,0].set_xlabel('Predicted Label')

# Anxiety - Val
cm_val_a = confusion_matrix(y_val_a, y_pred_val_a)
sns.heatmap(cm_val_a, annot=True, fmt='d', cmap='Greens', ax=axes[1,1],
            xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
axes[1,1].set_title('Anxiety - Validation Set')
axes[1,1].set_ylabel('True Label')
axes[1,1].set_xlabel('Predicted Label')

# Anxiety - Test
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
# VISUALIZATION 2: ROC CURVES
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('ROC Curves: One-vs-Rest Multiclass Classification (Time Series Split)', fontsize=16, fontweight='bold')

def plot_roc_curves(y_true, y_pred_proba, ax, title, n_classes=3):
    """Plot ROC curves for multiclass classification (one-vs-rest)"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Low', 'Medium', 'High']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

# Stress ROC curves
plot_roc_curves(y_train_s, y_pred_proba_train_s, axes[0,0], 'Stress - Train Set')
plot_roc_curves(y_val_s, y_pred_proba_val_s, axes[0,1], 'Stress - Validation Set')
plot_roc_curves(y_test_s, y_pred_proba_test_s, axes[0,2], 'Stress - Test Set')

# Anxiety ROC curves
plot_roc_curves(y_train_a, y_pred_proba_train_a, axes[1,0], 'Anxiety - Train Set')
plot_roc_curves(y_val_a, y_pred_proba_val_a, axes[1,1], 'Anxiety - Validation Set')
plot_roc_curves(y_test_a, y_pred_proba_test_a, axes[1,2], 'Anxiety - Test Set')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'roc_curves.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves.png")

# ============================================================================
# VISUALIZATION 3: FEATURE IMPORTANCE
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
# VISUALIZATION 4: COMPREHENSIVE METRICS COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Comprehensive Performance Metrics (Time Series Split)', fontsize=16, fontweight='bold')

# Stress - Multiple metrics across datasets
metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)', 'ROC AUC\n(OvR)']
stress_train_vals = [stress_metrics_train['accuracy'], stress_metrics_train['precision_macro'], 
                     stress_metrics_train['recall_macro'], stress_metrics_train['f1_macro'],
                     stress_metrics_train['roc_auc_ovr']]
stress_val_vals = [stress_metrics_val['accuracy'], stress_metrics_val['precision_macro'], 
                   stress_metrics_val['recall_macro'], stress_metrics_val['f1_macro'],
                   stress_metrics_val['roc_auc_ovr']]
stress_test_vals = [stress_metrics_test['accuracy'], stress_metrics_test['precision_macro'], 
                    stress_metrics_test['recall_macro'], stress_metrics_test['f1_macro'],
                    stress_metrics_test['roc_auc_ovr']]

x = np.arange(len(metric_names))
width = 0.25

axes[0,0].bar(x - width, stress_train_vals, width, label='Train', color='#1f77b4', alpha=0.8)
axes[0,0].bar(x, stress_val_vals, width, label='Validation', color='#ff7f0e', alpha=0.8)
axes[0,0].bar(x + width, stress_test_vals, width, label='Test', color='#2ca02c', alpha=0.8)
axes[0,0].set_ylabel('Score', fontsize=12)
axes[0,0].set_title('Stress - Metrics Across Datasets', fontsize=14, fontweight='bold')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(metric_names, fontsize=10)
axes[0,0].set_ylim([0, 1.1])
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# Anxiety - Multiple metrics across datasets
anxiety_train_vals = [anxiety_metrics_train['accuracy'], anxiety_metrics_train['precision_macro'], 
                      anxiety_metrics_train['recall_macro'], anxiety_metrics_train['f1_macro'],
                      anxiety_metrics_train['roc_auc_ovr']]
anxiety_val_vals = [anxiety_metrics_val['accuracy'], anxiety_metrics_val['precision_macro'], 
                    anxiety_metrics_val['recall_macro'], anxiety_metrics_val['f1_macro'],
                    anxiety_metrics_val['roc_auc_ovr']]
anxiety_test_vals = [anxiety_metrics_test['accuracy'], anxiety_metrics_test['precision_macro'], 
                     anxiety_metrics_test['recall_macro'], anxiety_metrics_test['f1_macro'],
                     anxiety_metrics_test['roc_auc_ovr']]

axes[0,1].bar(x - width, anxiety_train_vals, width, label='Train', color='#1f77b4', alpha=0.8)
axes[0,1].bar(x, anxiety_val_vals, width, label='Validation', color='#ff7f0e', alpha=0.8)
axes[0,1].bar(x + width, anxiety_test_vals, width, label='Test', color='#2ca02c', alpha=0.8)
axes[0,1].set_ylabel('Score', fontsize=12)
axes[0,1].set_title('Anxiety - Metrics Across Datasets', fontsize=14, fontweight='bold')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(metric_names, fontsize=10)
axes[0,1].set_ylim([0, 1.1])
axes[0,1].legend()
axes[0,1].grid(axis='y', alpha=0.3)

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
plt.savefig(OUTPUT_DIR + 'comprehensive_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: comprehensive_metrics.png")

# ============================================================================
# DETAILED ANALYSIS REPORT
# ============================================================================

print("\n" + "="*80)
print("DETAILED ANALYSIS REPORT - TIME SERIES SPLIT")
print("="*80)

def print_metrics_table(metrics_dict, dataset_name):
    """Print metrics in a formatted table"""
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  Accuracy:              {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.2f}%)")
    print(f"  Precision (Macro):     {metrics_dict['precision_macro']:.4f}")
    print(f"  Precision (Weighted):  {metrics_dict['precision_weighted']:.4f}")
    print(f"  Recall (Macro):        {metrics_dict['recall_macro']:.4f}")
    print(f"  Recall (Weighted):     {metrics_dict['recall_weighted']:.4f}")
    print(f"  F1 Score (Macro):      {metrics_dict['f1_macro']:.4f}")
    print(f"  F1 Score (Weighted):   {metrics_dict['f1_weighted']:.4f}")
    if not np.isnan(metrics_dict['roc_auc_ovr']):
        print(f"  ROC AUC (OvR Macro):   {metrics_dict['roc_auc_ovr']:.4f}")
        print(f"  ROC AUC (OvR Weighted):{metrics_dict['roc_auc_ovr_weighted']:.4f}")

print("\n" + "-"*80)
print("STRESS PREDICTION - XGBoost (Regularized)")
print("-"*80)

print_metrics_table(stress_metrics_train, "Train")
print_metrics_table(stress_metrics_val, "Validation")
print_metrics_table(stress_metrics_test, "Test")

print(f"\nPer-Class Performance (Test Set):")
print(f"  Low Class Accuracy:    {stress_per_class[0]:.4f} ({stress_per_class[0]*100:.2f}%)")
print(f"  Medium Class Accuracy: {stress_per_class[1]:.4f} ({stress_per_class[1]*100:.2f}%)")
print(f"  High Class Accuracy:   {stress_per_class[2]:.4f} ({stress_per_class[2]*100:.2f}%)")

print(f"\nGeneralization Analysis:")
train_test_gap_s = stress_metrics_train['accuracy'] - stress_metrics_test['accuracy']
print(f"  Train-Test Gap:        {train_test_gap_s:.4f}")
print(f"  Overfitting Status:    {'Acceptable' if train_test_gap_s < 0.2 else 'Significant'}")

print("\n" + "-"*80)
print("ANXIETY PREDICTION - XGBoost (Regularized)")
print("-"*80)

print_metrics_table(anxiety_metrics_train, "Train")
print_metrics_table(anxiety_metrics_val, "Validation")
print_metrics_table(anxiety_metrics_test, "Test")

print(f"\nPer-Class Performance (Test Set):")
print(f"  Low Class Accuracy:    {anxiety_per_class[0]:.4f} ({anxiety_per_class[0]*100:.2f}%)")
print(f"  Medium Class Accuracy: {anxiety_per_class[1]:.4f} ({anxiety_per_class[1]*100:.2f}%)")
print(f"  High Class Accuracy:   {anxiety_per_class[2]:.4f} ({anxiety_per_class[2]*100:.2f}%)")

print(f"\nGeneralization Analysis:")
train_test_gap_a = anxiety_metrics_train['accuracy'] - anxiety_metrics_test['accuracy']
print(f"  Train-Test Gap:        {train_test_gap_a:.4f}")
print(f"  Overfitting Status:    {'Acceptable' if train_test_gap_a < 0.2 else 'Significant'}")

print("\n" + "-"*80)
print("TOP 5 MOST IMPORTANT FEATURES")
print("-"*80)
print("\nFor Stress Prediction:")
for idx, row in stress_importance.tail(5).iloc[::-1].iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print("\nFor Anxiety Prediction:")
for idx, row in anxiety_importance.tail(5).iloc[::-1].iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# ============================================================================
# SAVE METRICS TO CSV
# ============================================================================

print("\n" + "-"*80)
print("SAVING METRICS TO CSV")
print("-"*80)

# Create comprehensive metrics dataframe
metrics_data = []

for model_name, metrics_list in [('Stress', [stress_metrics_train, stress_metrics_val, stress_metrics_test]),
                                  ('Anxiety', [anxiety_metrics_train, anxiety_metrics_val, anxiety_metrics_test])]:
    for dataset, metrics in zip(['Train', 'Validation', 'Test'], metrics_list):
        row = {
            'Model': model_name,
            'Dataset': dataset,
            'Accuracy': metrics['accuracy'],
            'Precision_Macro': metrics['precision_macro'],
            'Precision_Weighted': metrics['precision_weighted'],
            'Recall_Macro': metrics['recall_macro'],
            'Recall_Weighted': metrics['recall_weighted'],
            'F1_Macro': metrics['f1_macro'],
            'F1_Weighted': metrics['f1_weighted'],
            'ROC_AUC_OvR_Macro': metrics['roc_auc_ovr'],
            'ROC_AUC_OvR_Weighted': metrics['roc_auc_ovr_weighted']
        }
        metrics_data.append(row)

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(OUTPUT_DIR + 'model_metrics.csv', index=False)
print(f"Saved: model_metrics.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll visualizations and metrics saved to: {OUTPUT_DIR}")
print("Files created:")
print("  - confusion_matrices.png")
print("  - roc_curves.png")
print("  - feature_importance.png")
print("  - comprehensive_metrics.png")
print("  - model_metrics.csv")
print("\nNote: Time series split (70-15-15) preserves temporal order - no shuffling applied")