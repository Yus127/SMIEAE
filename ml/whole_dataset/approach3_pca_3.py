import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')
import os

INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model3/model_results_pca_3class_regularized.csv'
ROC_OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model3/roc_curves.png'
os.makedirs('/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model3', exist_ok=True)

print("3-CLASS CLASSIFICATION - STRONGLY REGULARIZED")
print("Train: 70% | Validation: 15% | Test: 15%")

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

# PREPARE DATA WITH PCA

def create_target_classes(series, p33, p67):
    classes = pd.cut(series, bins=[-np.inf, p33, p67, np.inf], labels=[0, 1, 2])
    return classes.astype(int)

def prepare_data_with_pca(df, target_col, feature_cols, n_components=0.80):
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR: {target_col}")
    print(f"{'='*80}")
    
    df_clean = df[df[target_col].notna()].copy()
    print(f"\nRows: {len(df_clean)}")
    
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    
    y = create_target_classes(df_clean[target_col], p33, p67)
    
    print(f"Classes: Low {(y==0).sum()} | Med {(y==1).sum()} | High {(y==2).sum()}")
    
    X = df_clean[feature_cols].copy()
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"Split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"\nPCA (80% variance)...")
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Components: {pca.n_components_} | Variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test, p33, p67, pca

# PLOT ROC CURVES

def plot_roc_curves(models_dict, X_test, y_test, target_name, n_classes=3):
    """
    Plot ROC curves for multi-class classification using one-vs-rest approach
    """
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'ROC Curves - {target_name}', fontsize=16, fontweight='bold')
    
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    class_names = ['Low', 'Medium', 'High']
    
    model_names = list(models_dict.keys())
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Get probability predictions
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves for each class
        colors_iter = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
        for i, color in zip(range(n_classes), colors_iter):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
        
        # Formatting
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Remove the last empty subplot
    if len(models_dict) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    return fig

# TRAIN STRONGLY REGULARIZED MODELS

def train_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, target_name):
    print(f"\n{'='*80}")
    print(f"TRAINING: {target_name}")
    print(f"{'='*80}")
    
    models = {
        'Logistic Regression': LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=42,
            C=0.5,
            penalty='l2'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'SVM (RBF)': SVC(
            kernel='rbf',
            C=0.3,
            gamma='scale',
            random_state=42,
            probability=True
        )
    }
    
    results = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        test_f1 = f1_score(y_test, model.predict(X_test), average='macro')
        
        results[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'test_precision': precision_score(y_test, model.predict(X_test), average='macro'),
            'test_recall': recall_score(y_test, model.predict(X_test), average='macro')
        }
        
        gap = train_acc - test_acc
        status = "" if gap <= 0.05 else ""
        print(f"{status} Train:{train_acc:.4f} Val:{val_acc:.4f} Test:{test_acc:.4f} F1:{test_f1:.4f} Gap:{gap:.4f}")
        print(confusion_matrix(y_test, model.predict(X_test)))
    
    # ENSEMBLE
    print(f"\nENSEMBLE")
    
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42, C=0.5, penalty='l2')),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1)),
            ('xgb', XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, random_state=42, eval_metric='mlogloss', use_label_encoder=False))
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    trained_models['Ensemble'] = voting_clf
    
    train_acc_ens = accuracy_score(y_train, voting_clf.predict(X_train))
    val_acc_ens = accuracy_score(y_val, voting_clf.predict(X_val))
    test_acc_ens = accuracy_score(y_test, voting_clf.predict(X_test))
    test_f1_ens = f1_score(y_test, voting_clf.predict(X_test), average='macro')
    
    results['Ensemble'] = {
        'train_accuracy': train_acc_ens,
        'val_accuracy': val_acc_ens,
        'test_accuracy': test_acc_ens,
        'test_f1_macro': test_f1_ens,
        'test_precision': precision_score(y_test, voting_clf.predict(X_test), average='macro'),
        'test_recall': recall_score(y_test, voting_clf.predict(X_test), average='macro')
    }
    
    gap_ens = train_acc_ens - test_acc_ens
    status = "" if gap_ens <= 0.05 else ""
    print(f"{status} Train:{train_acc_ens:.4f} Val:{val_acc_ens:.4f} Test:{test_acc_ens:.4f} F1:{test_f1_ens:.4f} Gap:{gap_ens:.4f}")
    print(confusion_matrix(y_test, voting_clf.predict(X_test)))
    
    return results, trained_models

# EXECUTE

X_train_stress, X_val_stress, X_test_stress, y_train_stress, y_val_stress, y_test_stress, p33_stress, p67_stress, pca_stress = prepare_data_with_pca(df, 'stress_level', advanced_features, n_components=0.80)
stress_results, stress_models = train_evaluate_models(X_train_stress, X_val_stress, X_test_stress, y_train_stress, y_val_stress, y_test_stress, "STRESS")

X_train_anxiety, X_val_anxiety, X_test_anxiety, y_train_anxiety, y_val_anxiety, y_test_anxiety, p33_anxiety, p67_anxiety, pca_anxiety = prepare_data_with_pca(df, 'anxiety_level', advanced_features, n_components=0.80)
anxiety_results, anxiety_models = train_evaluate_models(X_train_anxiety, X_val_anxiety, X_test_anxiety, y_train_anxiety, y_val_anxiety, y_test_anxiety, "ANXIETY")

# GENERATE ROC CURVES

print("\n" + "="*80)
print("GENERATING ROC CURVES")

# Create a combined figure with stress and anxiety
fig = plt.figure(figsize=(20, 14))

# Plot Stress ROC curves
print("\nPlotting STRESS ROC curves...")
for idx, (model_name, model) in enumerate(stress_models.items()):
    ax = plt.subplot(2, 5, idx + 1)
    
    y_test_bin = label_binarize(y_test_stress, classes=[0, 1, 2])
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_stress)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_stress)
    else:
        continue
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Low', 'Medium', 'High']
    
    for i, color in zip(range(3), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'STRESS - {model_name}', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

# Plot Anxiety ROC curves
print("Plotting ANXIETY ROC curves...")
for idx, (model_name, model) in enumerate(anxiety_models.items()):
    ax = plt.subplot(2, 5, idx + 6)
    
    y_test_bin = label_binarize(y_test_anxiety, classes=[0, 1, 2])
    
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_anxiety)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test_anxiety)
    else:
        continue
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Low', 'Medium', 'High']
    
    for i, color in zip(range(3), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
               label=f'{class_names[i]} (AUC={roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'ANXIETY - {model_name}', fontsize=11, fontweight='bold')
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('ROC Curves - Multi-Class Classification (One-vs-Rest)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(ROC_OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nROC curves saved to: {ROC_OUTPUT_PATH}")
plt.close()

# SUMMARY

print("\n" + "="*80)
print("FINAL SUMMARY")

print("\nSTRESS:")
for name, m in stress_results.items():
    gap = m['train_accuracy'] - m['test_accuracy']
    print(f"{name:<25} Train:{m['train_accuracy']:.4f} Val:{m['val_accuracy']:.4f} Test:{m['test_accuracy']:.4f} F1:{m['test_f1_macro']:.4f} Gap:{gap:.4f}")

print("\nANXIETY:")
for name, m in anxiety_results.items():
    gap = m['train_accuracy'] - m['test_accuracy']
    print(f"{name:<25} Train:{m['train_accuracy']:.4f} Val:{m['val_accuracy']:.4f} Test:{m['test_accuracy']:.4f} F1:{m['test_f1_macro']:.4f} Gap:{gap:.4f}")

best_stress = max(stress_results.items(), key=lambda x: x[1]['val_accuracy'])
best_anxiety = max(anxiety_results.items(), key=lambda x: x[1]['val_accuracy'])

print("\n" + "="*80)
print("BEST MODELS (by validation accuracy)")
print(f"\nStress:  {best_stress[0]} - Val:{best_stress[1]['val_accuracy']:.4f} Test:{best_stress[1]['test_accuracy']:.4f}")
print(f"Anxiety: {best_anxiety[0]} - Val:{best_anxiety[1]['val_accuracy']:.4f} Test:{best_anxiety[1]['test_accuracy']:.4f}")

# Save results
results_df = pd.DataFrame({
    'Model': list(stress_results.keys()),
    'Stress_Train': [v['train_accuracy'] for v in stress_results.values()],
    'Stress_Val': [v['val_accuracy'] for v in stress_results.values()],
    'Stress_Test': [v['test_accuracy'] for v in stress_results.values()],
    'Stress_F1': [v['test_f1_macro'] for v in stress_results.values()],
    'Stress_Gap': [v['train_accuracy'] - v['test_accuracy'] for v in stress_results.values()],
    'Anxiety_Train': [v['train_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Val': [v['val_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test': [v['test_accuracy'] for v in anxiety_results.values()],
    'Anxiety_F1': [v['test_f1_macro'] for v in anxiety_results.values()],
    'Anxiety_Gap': [v['train_accuracy'] - v['test_accuracy'] for v in anxiety_results.values()]
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print("\n" + "="*80)