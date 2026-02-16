import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
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
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model_comparison/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("MULTI-MODEL COMPARISON: STRESS & ANXIETY PREDICTION")

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

# PREPARE DATA FOR BOTH TARGETS

print("\nPreparing data...")
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, p33_s, p67_s = prepare_data(df, 'stress_level', feature_columns)
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a, p33_a, p67_a = prepare_data(df, 'anxiety_level', feature_columns)

# DEFINE ALL MODELS

print("\nInitializing models...")

models = {
    'Logistic Regression': LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        random_state=42
    ),
    'Naive Bayes': GaussianNB()
}

# TRAIN ALL MODELS AND COLLECT PREDICTIONS

print("\nTraining all models...")

stress_results = {}
anxiety_results = {}

for model_name, model_template in models.items():
    print(f"\nTraining {model_name}...")
    
    # Stress model
    from sklearn.base import clone
    stress_model = clone(model_template)
    stress_model.fit(X_train_s, y_train_s)
    
    stress_results[model_name] = {
        'model': stress_model,
        'train_pred': stress_model.predict(X_train_s),
        'val_pred': stress_model.predict(X_val_s),
        'test_pred': stress_model.predict(X_test_s),
        'train_proba': stress_model.predict_proba(X_train_s),
        'val_proba': stress_model.predict_proba(X_val_s),
        'test_proba': stress_model.predict_proba(X_test_s)
    }
    
    # Anxiety model
    anxiety_model = clone(model_template)
    anxiety_model.fit(X_train_a, y_train_a)
    
    anxiety_results[model_name] = {
        'model': anxiety_model,
        'train_pred': anxiety_model.predict(X_train_a),
        'val_pred': anxiety_model.predict(X_val_a),
        'test_pred': anxiety_model.predict(X_test_a),
        'train_proba': anxiety_model.predict_proba(X_train_a),
        'val_proba': anxiety_model.predict_proba(X_val_a),
        'test_proba': anxiety_model.predict_proba(X_test_a)
    }

# CALCULATE METRICS FOR ALL MODELS

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

print("\nCalculating metrics for all models...")

stress_metrics = {}
anxiety_metrics = {}

for model_name in models.keys():
    stress_metrics[model_name] = {
        'train': calculate_metrics(y_train_s, stress_results[model_name]['train_pred'], stress_results[model_name]['train_proba']),
        'val': calculate_metrics(y_val_s, stress_results[model_name]['val_pred'], stress_results[model_name]['val_proba']),
        'test': calculate_metrics(y_test_s, stress_results[model_name]['test_pred'], stress_results[model_name]['test_proba'])
    }
    
    anxiety_metrics[model_name] = {
        'train': calculate_metrics(y_train_a, anxiety_results[model_name]['train_pred'], anxiety_results[model_name]['train_proba']),
        'val': calculate_metrics(y_val_a, anxiety_results[model_name]['val_pred'], anxiety_results[model_name]['val_proba']),
        'test': calculate_metrics(y_test_a, anxiety_results[model_name]['test_pred'], anxiety_results[model_name]['test_proba'])
    }

# VISUALIZATION 1: MODEL COMPARISON - TEST SET PERFORMANCE

print("\nGenerating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Model Comparison: Test Set Performance', fontsize=16, fontweight='bold')

# Stress comparison
metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)', 'ROC AUC\n(OvR)']
x = np.arange(len(metric_names))
width = 0.15

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, model_name in enumerate(models.keys()):
    metrics = stress_metrics[model_name]['test']
    values = [
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro'],
        metrics['roc_auc_ovr']
    ]
    axes[0].bar(x + i*width - 2*width, values, width, label=model_name, color=colors[i], alpha=0.8)

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Stress Prediction - Test Set', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metric_names, fontsize=10)
axes[0].set_ylim([0, 1.1])
axes[0].legend(loc='lower right', fontsize=9)
axes[0].grid(axis='y', alpha=0.3)

# Anxiety comparison
for i, model_name in enumerate(models.keys()):
    metrics = anxiety_metrics[model_name]['test']
    values = [
        metrics['accuracy'],
        metrics['precision_macro'],
        metrics['recall_macro'],
        metrics['f1_macro'],
        metrics['roc_auc_ovr']
    ]
    axes[1].bar(x + i*width - 2*width, values, width, label=model_name, color=colors[i], alpha=0.8)

axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Anxiety Prediction - Test Set', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metric_names, fontsize=10)
axes[1].set_ylim([0, 1.1])
axes[1].legend(loc='lower right', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'model_comparison_test.png', dpi=300, bbox_inches='tight')
print("Saved: model_comparison_test.png")

# VISUALIZATION 2: CONFUSION MATRICES FOR ALL MODELS

fig, axes = plt.subplots(5, 2, figsize=(14, 28))
fig.suptitle('Confusion Matrices: All Models (Test Set)', fontsize=16, fontweight='bold')

for i, model_name in enumerate(models.keys()):
    # Stress
    cm_stress = confusion_matrix(y_test_s, stress_results[model_name]['test_pred'])
    sns.heatmap(cm_stress, annot=True, fmt='d', cmap='Blues', ax=axes[i, 0],
                xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
    axes[i, 0].set_title(f'{model_name} - Stress', fontsize=12, fontweight='bold')
    axes[i, 0].set_ylabel('True Label')
    axes[i, 0].set_xlabel('Predicted Label')
    
    # Anxiety
    cm_anxiety = confusion_matrix(y_test_a, anxiety_results[model_name]['test_pred'])
    sns.heatmap(cm_anxiety, annot=True, fmt='d', cmap='Greens', ax=axes[i, 1],
                xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
    axes[i, 1].set_title(f'{model_name} - Anxiety', fontsize=12, fontweight='bold')
    axes[i, 1].set_ylabel('True Label')
    axes[i, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'confusion_matrices_all_models.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrices_all_models.png")

# VISUALIZATION 3: ROC CURVES FOR ALL MODELS

def plot_roc_curves_multimodel(y_true, results_dict, ax, title, n_classes=3):
    """Plot ROC curves for multiple models"""
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (model_name, color) in enumerate(zip(results_dict.keys(), colors)):
        y_pred_proba = results_dict[model_name]['test_proba']
        
        # Calculate micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(fpr_micro, tpr_micro, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc_micro:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('ROC Curves: All Models (Test Set, Micro-Average)', fontsize=16, fontweight='bold')

plot_roc_curves_multimodel(y_test_s, stress_results, axes[0], 'Stress Prediction')
plot_roc_curves_multimodel(y_test_a, anxiety_results, axes[1], 'Anxiety Prediction')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'roc_curves_all_models.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_all_models.png")

# VISUALIZATION 4: GENERALIZATION ANALYSIS (TRAIN VS TEST)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Generalization Analysis: Train vs Test Accuracy', fontsize=16, fontweight='bold')

model_names_list = list(models.keys())

# Stress
stress_train_acc = [stress_metrics[m]['train']['accuracy'] for m in model_names_list]
stress_test_acc = [stress_metrics[m]['test']['accuracy'] for m in model_names_list]

x_pos = np.arange(len(model_names_list))
width = 0.35

axes[0].bar(x_pos - width/2, stress_train_acc, width, label='Train', color='#1f77b4', alpha=0.7)
axes[0].bar(x_pos + width/2, stress_test_acc, width, label='Test', color='#2ca02c', alpha=0.7)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Stress Prediction', fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_names_list, rotation=15, ha='right', fontsize=10)
axes[0].set_ylim([0, 1.1])
axes[0].legend(fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Add gap annotations
for i, (train_acc, test_acc) in enumerate(zip(stress_train_acc, stress_test_acc)):
    gap = train_acc - test_acc
    axes[0].text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.3f}', 
                ha='center', fontsize=9, fontweight='bold')

# Anxiety
anxiety_train_acc = [anxiety_metrics[m]['train']['accuracy'] for m in model_names_list]
anxiety_test_acc = [anxiety_metrics[m]['test']['accuracy'] for m in model_names_list]

axes[1].bar(x_pos - width/2, anxiety_train_acc, width, label='Train', color='#1f77b4', alpha=0.7)
axes[1].bar(x_pos + width/2, anxiety_test_acc, width, label='Test', color='#2ca02c', alpha=0.7)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Anxiety Prediction', fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_names_list, rotation=15, ha='right', fontsize=10)
axes[1].set_ylim([0, 1.1])
axes[1].legend(fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

# Add gap annotations
for i, (train_acc, test_acc) in enumerate(zip(anxiety_train_acc, anxiety_test_acc)):
    gap = train_acc - test_acc
    axes[1].text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.3f}', 
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'generalization_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: generalization_analysis.png")

# VISUALIZATION 5: FEATURE IMPORTANCE COMPARISON (for tree-based models)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('Feature Importance: Tree-Based Models', fontsize=16, fontweight='bold')

tree_models = ['Random Forest', 'XGBoost']

# Stress feature importance
for model_name in tree_models:
    if hasattr(stress_results[model_name]['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': stress_results[model_name]['model'].feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[0].barh(np.arange(len(importance)) + (0.4 if model_name == 'XGBoost' else 0), 
                    importance['importance'], 
                    height=0.35,
                    label=model_name,
                    alpha=0.8)

axes[0].set_yticks(np.arange(len(importance)) + 0.2)
axes[0].set_yticklabels(importance['feature'])
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Stress Prediction - Top 10 Features', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='x', alpha=0.3)

# Anxiety feature importance
for model_name in tree_models:
    if hasattr(anxiety_results[model_name]['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': anxiety_results[model_name]['model'].feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        axes[1].barh(np.arange(len(importance)) + (0.4 if model_name == 'XGBoost' else 0), 
                    importance['importance'], 
                    height=0.35,
                    label=model_name,
                    alpha=0.8)

axes[1].set_yticks(np.arange(len(importance)) + 0.2)
axes[1].set_yticklabels(importance['feature'])
axes[1].set_xlabel('Importance', fontsize=12)
axes[1].set_title('Anxiety Prediction - Top 10 Features', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance_comparison.png")

# SAVE ALL METRICS TO CSV

print("\nSaving metrics to CSV...")

all_metrics_data = []

for model_name in models.keys():
    for target_type, metrics_dict in [('Stress', stress_metrics[model_name]), 
                                       ('Anxiety', anxiety_metrics[model_name])]:
        for dataset in ['train', 'val', 'test']:
            metrics = metrics_dict[dataset]
            row = {
                'Model': model_name,
                'Target': target_type,
                'Dataset': dataset.capitalize(),
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
            all_metrics_data.append(row)

metrics_df = pd.DataFrame(all_metrics_data)
metrics_df.to_csv(OUTPUT_DIR + 'all_models_metrics.csv', index=False)
print("Saved: all_models_metrics.csv")

# DETAILED ANALYSIS REPORT

print("\n" + "="*80)
print("DETAILED ANALYSIS REPORT")

for target_name, metrics_collection in [('STRESS', stress_metrics), ('ANXIETY', anxiety_metrics)]:
    print(f"\n{'='*80}")
    print(f"{target_name} PREDICTION - ALL MODELS COMPARISON")
    
    for model_name in models.keys():
        print(f"\n{'-'*80}")
        print(f"{model_name}")
        print('-'*80)
        
        for dataset in ['train', 'val', 'test']:
            metrics = metrics_collection[model_name][dataset]
            print(f"\n{dataset.capitalize()} Set:")
            print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
            print(f"  F1 Score (Macro):  {metrics['f1_macro']:.4f}")
            if not np.isnan(metrics['roc_auc_ovr']):
                print(f"  ROC AUC (OvR):     {metrics['roc_auc_ovr']:.4f}")
        
        # Generalization gap
        train_acc = metrics_collection[model_name]['train']['accuracy']
        test_acc = metrics_collection[model_name]['test']['accuracy']
        gap = train_acc - test_acc
        print(f"\nGeneralization Gap (Train-Test): {gap:.4f}")
        print(f"Overfitting Status: {'Minimal' if gap < 0.05 else 'Acceptable' if gap < 0.15 else 'Significant'}")

print("\n" + "="*80)
print("BEST MODELS SUMMARY")

# Find best models
print("\nBest Test Set Performance:")
print("\nStress Prediction:")
best_stress = max(stress_metrics.items(), key=lambda x: x[1]['test']['accuracy'])
print(f"  Best Model: {best_stress[0]}")
print(f"  Test Accuracy: {best_stress[1]['test']['accuracy']:.4f} ({best_stress[1]['test']['accuracy']*100:.2f}%)")
print(f"  Test F1 (Macro): {best_stress[1]['test']['f1_macro']:.4f}")

print("\nAnxiety Prediction:")
best_anxiety = max(anxiety_metrics.items(), key=lambda x: x[1]['test']['accuracy'])
print(f"  Best Model: {best_anxiety[0]}")
print(f"  Test Accuracy: {best_anxiety[1]['test']['accuracy']:.4f} ({best_anxiety[1]['test']['accuracy']*100:.2f}%)")
print(f"  Test F1 (Macro): {best_anxiety[1]['test']['f1_macro']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print(f"\nAll visualizations and metrics saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - model_comparison_test.png")
print("  - confusion_matrices_all_models.png")
print("  - roc_curves_all_models.png")
print("  - generalization_analysis.png")
print("  - feature_importance_comparison.png")
print("  - all_models_metrics.csv")