import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')
import os

INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model3/model_results_pca_3class_regularized_timeseries.csv'
ROC_OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model3/'
os.makedirs('/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model3', exist_ok=True)

print("3-CLASS CLASSIFICATION - STRONGLY REGULARIZED")
print("Train: 70% | Validation: 15% | Test: 15% (TIME SERIES SPLIT)")

# Load data
print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# FEATURE ENGINEERING

print("\n" + "="*80)
print("FEATURE ENGINEERING")

print("\nCalculating personal baselines...")
user_baselines = df.groupby('userid').agg({
    'heart_rate_activity_beats per minute_mean': 'mean',
    'daily_total_steps': 'mean',
    'daily_hrv_summary_rmssd': 'mean',
    'daily_respiratory_rate_daily_respiratory_rate': 'mean',
    'sleep_global_duration': 'mean'
}).reset_index()

user_baselines.columns = ['userid', 'hr_baseline', 'steps_baseline', 'hrv_baseline', 
                          'resp_baseline', 'sleep_baseline']

df = df.merge(user_baselines, on='userid', how='left')

print("Creating features...")
df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
df['activity_ratio'] = (df['daily_total_steps']) / (df['steps_baseline'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)
df['exam_proximity_inverse'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['post_exam_proximity_inverse'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)
df['hr_dev_x_exam'] = df['hr_deviation'] * df['is_exam_period']
df['activity_x_exam'] = df['activity_ratio'] * df['is_exam_period']
df['hrv_dev_x_exam'] = df['hrv_deviation'] * df['is_exam_period']
df['sleep_quality'] = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['activity_intensity'] = df['daily_total_steps'] / (df['activity_level_sedentary_count'] + 1)
df['autonomic_balance'] = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)
df['recovery_score'] = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['cardio_stress'] = df['heart_rate_activity_beats per minute_mean'] / (df['daily_hrv_summary_rmssd'] + 1)
df['sleep_fragmentation'] = df['wake_count'] / (df['sleep_global_duration'] / 60 + 1)
df['resp_efficiency'] = df['minute_spo2_value_mean'] / (df['daily_respiratory_rate_daily_respiratory_rate'] + 1)

advanced_features = [
    'daily_total_steps', 'activity_level_sedentary_count', 'daily_respiratory_rate_daily_respiratory_rate',
    'minute_spo2_value_mean', 'daily_hrv_summary_rmssd', 'hrv_details_rmssd_min',
    'sleep_global_duration', 'sleep_global_efficiency', 'deep_sleep_minutes', 'rem_sleep_minutes',
    'heart_rate_activity_beats per minute_mean', 'hr_deviation', 'activity_ratio', 'hrv_deviation',
    'sleep_deviation', 'is_exam_period', 'exam_proximity_inverse', 'post_exam_proximity_inverse',
    'hr_dev_x_exam', 'activity_x_exam', 'hrv_dev_x_exam', 'sleep_quality', 'activity_intensity',
    'autonomic_balance', 'recovery_score', 'cardio_stress', 'sleep_fragmentation', 'resp_efficiency'
]

print(f"Using {len(advanced_features)} features")

# PREPARE DATA WITH PCA AND TIME SERIES SPLIT

def create_target_classes(series, p33, p67):
    classes = pd.cut(series, bins=[-np.inf, p33, p67, np.inf], labels=[0, 1, 2])
    return classes.astype(int)

def prepare_data_with_pca_timeseries(df, target_col, feature_cols, n_components=0.80):
    """Prepare data with PCA and 70-15-15 time series split"""
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR: {target_col} (TIME SERIES SPLIT)")
    print(f"{'='*80}")
    
    df_clean = df[df[target_col].notna()].copy()
    print(f"\nRows: {len(df_clean)}")
    
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    
    y = create_target_classes(df_clean[target_col], p33, p67)
    
    print(f"Classes: Low {(y==0).sum()} | Med {(y==1).sum()} | High {(y==2).sum()}")
    
    X = df_clean[feature_cols].copy()
    
    # Time series split: 70-15-15
    n_samples = len(X)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split sequentially (no shuffling for time series)
    X_train = X.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    
    y_train = y.iloc[:train_size]
    y_val = y.iloc[train_size:train_size + val_size]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"Split (time series): Train {len(X_train)} ({len(X_train)/n_samples*100:.1f}%) | "
          f"Val {len(X_val)} ({len(X_val)/n_samples*100:.1f}%) | "
          f"Test {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
    
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
    
    # PCA
    print(f"\nApplying PCA (80% variance)...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Components: {pca.n_components_} | Variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, p33, p67, pca

# TRAIN STRONGLY REGULARIZED MODELS WITH COMPREHENSIVE METRICS

def calculate_metrics_multiclass(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics for multiclass classification"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ROC AUC (one-vs-rest for multiclass)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
    except:
        metrics['roc_auc'] = np.nan
    
    return metrics

def train_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, target_name):
    print(f"\n{'='*80}")
    print(f"TRAINING MODELS FOR: {target_name}")
    print(f"{'='*80}")
    
    models = {
        'Logistic Regression': LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42,
            C=0.5,  # STRONGER regularization
            penalty='l2'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,  # REDUCED from 100
            max_depth=3,  # REDUCED from 5
            min_samples_split=20,  # INCREASED
            min_samples_leaf=10,  # INCREASED
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=50,  # REDUCED
            max_depth=2,  # REDUCED
            learning_rate=0.03,  # REDUCED
            subsample=0.7,  # REDUCED
            colsample_bytree=0.7,  # REDUCED
            reg_alpha=0.5,  # INCREASED
            reg_lambda=2.0,  # INCREASED
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=0.3,  # STRONGER regularization
            gamma='scale',
            random_state=42,
            probability=True
        )
    }
    
    results = {}
    roc_data = {}  # Store ROC curve data for plotting
    
    for model_name, model in models.items():
        print(f"\n{'-'*80}")
        print(f"Training: {model_name}")
        print(f"{'-'*80}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Probabilities
        y_train_proba = model.predict_proba(X_train)
        y_val_proba = model.predict_proba(X_val)
        y_test_proba = model.predict_proba(X_test)
        
        # Store ROC data for test set (for multiclass OvR)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        roc_data[model_name] = {
            'y_test_bin': y_test_bin,
            'y_test_proba': y_test_proba
        }
        
        # Calculate metrics
        train_metrics = calculate_metrics_multiclass(y_train, y_train_pred, y_train_proba)
        val_metrics = calculate_metrics_multiclass(y_val, y_val_pred, y_val_proba)
        test_metrics = calculate_metrics_multiclass(y_test, y_test_pred, y_test_proba)
        
        results[model_name] = {
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'train_roc_auc': train_metrics['roc_auc'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'test_roc_auc': test_metrics['roc_auc']
        }
        
        gap = train_metrics['accuracy'] - test_metrics['accuracy']
        status = "" if gap <= 0.05 else "" if gap <= 0.10 else ""
        
        print(f"\nTrain Metrics:")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        print(f"  F1 Score:  {train_metrics['f1']:.4f}")
        if not np.isnan(train_metrics['roc_auc']):
            print(f"  ROC AUC:   {train_metrics['roc_auc']:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1 Score:  {val_metrics['f1']:.4f}")
        if not np.isnan(val_metrics['roc_auc']):
            print(f"  ROC AUC:   {val_metrics['roc_auc']:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1 Score:  {test_metrics['f1']:.4f}")
        if not np.isnan(test_metrics['roc_auc']):
            print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
        
        print(f"\n{status} Generalization: Train-Test Gap = {gap:.4f}")
        
        print("\nTest Set - Confusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        print("  Rows: True labels (Low=0, Med=1, High=2)")
        print("  Cols: Predicted labels")
    
    # ENSEMBLE with strongly regularized models
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MODEL: Soft Voting")
    print(f"{'='*80}")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, 
                                     random_state=42, C=0.5, penalty='l2')),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=20, 
                                         min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.03, subsample=0.7, 
                                 colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, random_state=42, 
                                 eval_metric='mlogloss', use_label_encoder=False))
        ],
        voting='soft'
    )
    
    print("\nTraining ensemble...")
    voting_clf.fit(X_train, y_train)
    
    # Predictions
    y_train_pred_ens = voting_clf.predict(X_train)
    y_val_pred_ens = voting_clf.predict(X_val)
    y_test_pred_ens = voting_clf.predict(X_test)
    
    # Probabilities
    y_train_proba_ens = voting_clf.predict_proba(X_train)
    y_val_proba_ens = voting_clf.predict_proba(X_val)
    y_test_proba_ens = voting_clf.predict_proba(X_test)
    
    # Store ROC data for ensemble
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    roc_data['Ensemble'] = {
        'y_test_bin': y_test_bin,
        'y_test_proba': y_test_proba_ens
    }
    
    # Calculate metrics
    train_metrics_ens = calculate_metrics_multiclass(y_train, y_train_pred_ens, y_train_proba_ens)
    val_metrics_ens = calculate_metrics_multiclass(y_val, y_val_pred_ens, y_val_proba_ens)
    test_metrics_ens = calculate_metrics_multiclass(y_test, y_test_pred_ens, y_test_proba_ens)
    
    results['Ensemble'] = {
        'train_accuracy': train_metrics_ens['accuracy'],
        'train_precision': train_metrics_ens['precision'],
        'train_recall': train_metrics_ens['recall'],
        'train_f1': train_metrics_ens['f1'],
        'train_roc_auc': train_metrics_ens['roc_auc'],
        'val_accuracy': val_metrics_ens['accuracy'],
        'val_precision': val_metrics_ens['precision'],
        'val_recall': val_metrics_ens['recall'],
        'val_f1': val_metrics_ens['f1'],
        'val_roc_auc': val_metrics_ens['roc_auc'],
        'test_accuracy': test_metrics_ens['accuracy'],
        'test_precision': test_metrics_ens['precision'],
        'test_recall': test_metrics_ens['recall'],
        'test_f1': test_metrics_ens['f1'],
        'test_roc_auc': test_metrics_ens['roc_auc']
    }
    
    gap_ens = train_metrics_ens['accuracy'] - test_metrics_ens['accuracy']
    status = "" if gap_ens <= 0.05 else "" if gap_ens <= 0.10 else ""
    
    print(f"\nTrain Metrics:")
    print(f"  Accuracy:  {train_metrics_ens['accuracy']:.4f}")
    print(f"  Precision: {train_metrics_ens['precision']:.4f}")
    print(f"  Recall:    {train_metrics_ens['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics_ens['f1']:.4f}")
    print(f"  ROC AUC:   {train_metrics_ens['roc_auc']:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_metrics_ens['accuracy']:.4f}")
    print(f"  Precision: {val_metrics_ens['precision']:.4f}")
    print(f"  Recall:    {val_metrics_ens['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics_ens['f1']:.4f}")
    print(f"  ROC AUC:   {val_metrics_ens['roc_auc']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics_ens['accuracy']:.4f}")
    print(f"  Precision: {test_metrics_ens['precision']:.4f}")
    print(f"  Recall:    {test_metrics_ens['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics_ens['f1']:.4f}")
    print(f"  ROC AUC:   {test_metrics_ens['roc_auc']:.4f}")
    
    print(f"\n{status} Generalization: Train-Test Gap = {gap_ens:.4f}")
    
    print("\nTest Set - Confusion Matrix:")
    cm_ens = confusion_matrix(y_test, y_test_pred_ens)
    print(cm_ens)
    
    return results, roc_data

# FUNCTION TO PLOT MULTICLASS ROC CURVES (One-vs-Rest)

def plot_multiclass_roc_curves(roc_data, target_name, save_path):
    """Plot ROC curves for multiclass classification using One-vs-Rest approach"""
    
    n_classes = 3
    class_names = ['Low', 'Medium', 'High']
    
    # Create a figure with subplots for each model
    n_models = len(roc_data)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for each class
    class_colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    for idx, (model_name, data) in enumerate(roc_data.items()):
        ax = axes[idx]
        
        y_test_bin = data['y_test_bin']
        y_test_proba = data['y_test_proba']
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
            roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_proba[:, i])
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
        roc_auc["micro"] = roc_auc_score(y_test_bin, y_test_proba, average='micro', multi_class='ovr')
        
        # Plot ROC curves for each class
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=2.5,
               label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Hide the extra subplot
    if n_models < len(axes):
        axes[-1].axis('off')
    
    # Add overall title
    fig.suptitle(f'ROC Curves (One-vs-Rest) - {target_name}', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Add text box with info
    textstr = 'Time Series Split (70-15-15)\n3-Class: Low-Medium-High\nOne-vs-Rest Approach'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    fig.text(0.99, 0.01, textstr, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # Save figure
    filename = f"roc_curve_multiclass_{target_name.lower().replace(' ', '_')}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\n Multiclass ROC curve saved to: {full_path}")
    
    plt.close()
    
    # Also create a combined plot showing only micro-average for all models
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    
    model_colors = {
        'Logistic Regression': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'SVM (RBF)': '#d62728',
        'Ensemble': '#e377c2'
    }
    
    for model_name, data in roc_data.items():
        y_test_bin = data['y_test_bin']
        y_test_proba = data['y_test_proba']
        
        # Compute micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
        roc_auc_micro = roc_auc_score(y_test_bin, y_test_proba, average='micro', multi_class='ovr')
        
        ax_combined.plot(fpr_micro, tpr_micro, 
                        color=model_colors.get(model_name, 'black'),
                        lw=2.5 if model_name == 'Ensemble' else 2,
                        label=f"{model_name} (AUC = {roc_auc_micro:.3f})",
                        alpha=0.9)
    
    # Plot diagonal
    ax_combined.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax_combined.set_xlim([0.0, 1.0])
    ax_combined.set_ylim([0.0, 1.05])
    ax_combined.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_combined.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_combined.set_title(f'ROC Curves (Micro-Average) - {target_name}', 
                         fontsize=14, fontweight='bold', pad=20)
    ax_combined.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax_combined.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    textstr = 'Time Series Split (70-15-15)\n3-Class: Low-Medium-High\nMicro-Average AUC'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax_combined.text(0.98, 0.02, textstr, transform=ax_combined.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    filename_combined = f"roc_curve_combined_{target_name.lower().replace(' ', '_')}.png"
    full_path_combined = os.path.join(save_path, filename_combined)
    plt.savefig(full_path_combined, dpi=300, bbox_inches='tight')
    print(f" Combined ROC curve saved to: {full_path_combined}")
    
    plt.close()

# EXECUTE

X_train_stress, X_val_stress, X_test_stress, y_train_stress, y_val_stress, y_test_stress, p33_stress, p67_stress, pca_stress = prepare_data_with_pca_timeseries(
    df, 'stress_level', advanced_features, n_components=0.80
)
stress_results, stress_roc_data = train_evaluate_models(
    X_train_stress, X_val_stress, X_test_stress, 
    y_train_stress, y_val_stress, y_test_stress, 
    "STRESS"
)

# Plot ROC curves for stress
plot_multiclass_roc_curves(stress_roc_data, "Stress Level", ROC_OUTPUT_PATH)

X_train_anxiety, X_val_anxiety, X_test_anxiety, y_train_anxiety, y_val_anxiety, y_test_anxiety, p33_anxiety, p67_anxiety, pca_anxiety = prepare_data_with_pca_timeseries(
    df, 'anxiety_level', advanced_features, n_components=0.80
)
anxiety_results, anxiety_roc_data = train_evaluate_models(
    X_train_anxiety, X_val_anxiety, X_test_anxiety, 
    y_train_anxiety, y_val_anxiety, y_test_anxiety, 
    "ANXIETY"
)

# Plot ROC curves for anxiety
plot_multiclass_roc_curves(anxiety_roc_data, "Anxiety Level", ROC_OUTPUT_PATH)

# FINAL SUMMARY

print("\n" + "="*80)
print("FINAL SUMMARY - TIME SERIES SPLIT (70-15-15)")

print("\n" + "-"*80)
print("STRESS LEVEL PREDICTION (3-Class: Low-Med-High)")
print("-"*80)
print(f"{'Model':<25} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Test Prec':<10} {'Test Rec':<10} {'Test F1':<10} {'Test AUC':<10} {'Gap':<10}")
print("-"*80)
for name, m in stress_results.items():
    gap = m['train_accuracy'] - m['test_accuracy']
    auc_str = f"{m['test_roc_auc']:.4f}" if not np.isnan(m['test_roc_auc']) else "N/A"
    print(f"{name:<25} {m['train_accuracy']:<10.4f} {m['val_accuracy']:<10.4f} {m['test_accuracy']:<10.4f} "
          f"{m['test_precision']:<10.4f} {m['test_recall']:<10.4f} {m['test_f1']:<10.4f} {auc_str:<10} {gap:<10.4f}")

print("\n" + "-"*80)
print("ANXIETY LEVEL PREDICTION (3-Class: Low-Med-High)")
print("-"*80)
print(f"{'Model':<25} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Test Prec':<10} {'Test Rec':<10} {'Test F1':<10} {'Test AUC':<10} {'Gap':<10}")
print("-"*80)
for name, m in anxiety_results.items():
    gap = m['train_accuracy'] - m['test_accuracy']
    auc_str = f"{m['test_roc_auc']:.4f}" if not np.isnan(m['test_roc_auc']) else "N/A"
    print(f"{name:<25} {m['train_accuracy']:<10.4f} {m['val_accuracy']:<10.4f} {m['test_accuracy']:<10.4f} "
          f"{m['test_precision']:<10.4f} {m['test_recall']:<10.4f} {m['test_f1']:<10.4f} {auc_str:<10} {gap:<10.4f}")

best_stress = max(stress_results.items(), key=lambda x: x[1]['val_accuracy'])
best_anxiety = max(anxiety_results.items(), key=lambda x: x[1]['val_accuracy'])

print("\n" + "="*80)
print("BEST MODELS (by validation accuracy)")
print(f"\nStress:  {best_stress[0]}")
print(f"  Train Accuracy:      {best_stress[1]['train_accuracy']:.4f}")
print(f"  Validation Accuracy: {best_stress[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy:       {best_stress[1]['test_accuracy']:.4f}")
print(f"  Test Precision:      {best_stress[1]['test_precision']:.4f}")
print(f"  Test Recall:         {best_stress[1]['test_recall']:.4f}")
print(f"  Test F1:             {best_stress[1]['test_f1']:.4f}")
print(f"  Test ROC AUC:        {best_stress[1]['test_roc_auc']:.4f}")

print(f"\nAnxiety: {best_anxiety[0]}")
print(f"  Train Accuracy:      {best_anxiety[1]['train_accuracy']:.4f}")
print(f"  Validation Accuracy: {best_anxiety[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy:       {best_anxiety[1]['test_accuracy']:.4f}")
print(f"  Test Precision:      {best_anxiety[1]['test_precision']:.4f}")
print(f"  Test Recall:         {best_anxiety[1]['test_recall']:.4f}")
print(f"  Test F1:             {best_anxiety[1]['test_f1']:.4f}")
print(f"  Test ROC AUC:        {best_anxiety[1]['test_roc_auc']:.4f}")

# Save results
results_df = pd.DataFrame({
    'Model': list(stress_results.keys()),
    'Stress_Train_Acc': [v['train_accuracy'] for v in stress_results.values()],
    'Stress_Val_Acc': [v['val_accuracy'] for v in stress_results.values()],
    'Stress_Test_Acc': [v['test_accuracy'] for v in stress_results.values()],
    'Stress_Test_Precision': [v['test_precision'] for v in stress_results.values()],
    'Stress_Test_Recall': [v['test_recall'] for v in stress_results.values()],
    'Stress_Test_F1': [v['test_f1'] for v in stress_results.values()],
    'Stress_Test_AUC': [v['test_roc_auc'] for v in stress_results.values()],
    'Stress_Gap': [v['train_accuracy'] - v['test_accuracy'] for v in stress_results.values()],
    'Anxiety_Train_Acc': [v['train_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Val_Acc': [v['val_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_Acc': [v['test_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_Precision': [v['test_precision'] for v in anxiety_results.values()],
    'Anxiety_Test_Recall': [v['test_recall'] for v in anxiety_results.values()],
    'Anxiety_Test_F1': [v['test_f1'] for v in anxiety_results.values()],
    'Anxiety_Test_AUC': [v['test_roc_auc'] for v in anxiety_results.values()],
    'Anxiety_Gap': [v['train_accuracy'] - v['test_accuracy'] for v in anxiety_results.values()]
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"ROC curves saved to: {ROC_OUTPUT_PATH}")
print("\n" + "="*80)