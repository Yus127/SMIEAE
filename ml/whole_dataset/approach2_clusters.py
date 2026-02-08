import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# HARDCODED PATHS
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model2/model_2.csv'
ROC_STRESS_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model2/roc_curve_stress.png'
ROC_ANXIETY_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model2/roc_curve_anxiety.png'
os.makedirs('/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model2/', exist_ok=True)
#os.makedirs(ROC_STRESS_PATH, exist_ok=True)
#os.makedirs(ROC_ANXIETY_PATH, exist_ok=True)

print("="*80)
print("BINARY CLASSIFICATION WITH CLUSTERING AND ADVANCED FEATURES")
print("Train: 70% | Validation: 15% | Test: 15%")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# ============================================================================
# STEP 1: CREATE USER CLUSTERS (K=7) BASED ON PHYSIOLOGICAL PROFILES
# ============================================================================

print("\n" + "="*80)
print("STEP 1: CREATING USER CLUSTERS (K=7)")
print("="*80)

# Features for clustering (physiological only, no stress/anxiety to avoid leakage)
clustering_features = [
    'sleep_global_duration',
    'sleep_global_efficiency',
    'deep_sleep_minutes',
    'rem_sleep_minutes',
    'daily_total_steps',
    'daily_hrv_summary_rmssd',
    'heart_rate_activity_beats per minute_mean',
    'daily_respiratory_rate_daily_respiratory_rate',
    'minute_spo2_value_mean',
    'activity_level_sedentary_count'
]

# Aggregate by user to get their physiological profile
user_profiles = df.groupby('userid')[clustering_features].agg(['mean', 'std']).reset_index()
user_profiles.columns = ['userid'] + [f'{col[0]}_{col[1]}' for col in user_profiles.columns[1:]]

# Handle missing values for clustering
imputer_cluster = SimpleImputer(strategy='median')
X_cluster = imputer_cluster.fit_transform(user_profiles.drop('userid', axis=1))

# Normalize for clustering
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

# K-Means clustering
print("\nPerforming K-Means clustering (K=2)...")
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
user_profiles['user_cluster'] = kmeans.fit_predict(X_cluster_scaled)

print("\nCluster distribution:")
print(user_profiles['user_cluster'].value_counts().sort_index())

# Map clusters back to main dataframe
df = df.merge(user_profiles[['userid', 'user_cluster']], on='userid', how='left')

# ============================================================================
# STEP 2: ADVANCED FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 2: ADVANCED FEATURE ENGINEERING")
print("="*80)

# Calculate personal baselines for each user
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

# Merge baselines
df = df.merge(user_baselines, on='userid', how='left')

# Create deviation features
print("Creating deviation features...")
df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
df['activity_ratio'] = (df['daily_total_steps']) / (df['steps_baseline'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
df['resp_deviation'] = (df['daily_respiratory_rate_daily_respiratory_rate'] - df['resp_baseline']) / (df['resp_baseline'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)

# Exam proximity features
print("Creating exam proximity features...")
df['exam_proximity_inverse'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['post_exam_proximity_inverse'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)

# Interaction features
print("Creating interaction features...")
df['cluster_x_exam'] = df['user_cluster'] * df['is_exam_period']
df['hr_dev_x_exam'] = df['hr_deviation'] * df['is_exam_period']
df['activity_x_exam'] = df['activity_ratio'] * df['is_exam_period']
df['hrv_dev_x_exam'] = df['hrv_deviation'] * df['is_exam_period']

# Sleep quality composite
df['sleep_quality'] = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)

# Activity intensity
df['activity_intensity'] = df['daily_total_steps'] / (df['activity_level_sedentary_count'] + 1)

# HRV/HR ratio (autonomic balance)
df['autonomic_balance'] = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)

# Recovery score (REM + deep sleep normalized)
df['recovery_score'] = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)

# Cardiovascular stress indicator
df['cardio_stress'] = df['heart_rate_activity_beats per minute_mean'] / (df['daily_hrv_summary_rmssd'] + 1)

# Sleep fragmentation
df['sleep_fragmentation'] = df['wake_count'] / (df['sleep_global_duration'] / 60 + 1)

# Respiratory efficiency
df['resp_efficiency'] = df['minute_spo2_value_mean'] / (df['daily_respiratory_rate_daily_respiratory_rate'] + 1)

print(f"\nTotal features after engineering: {df.shape[1]}")

# Define advanced feature set (≈25 features)
advanced_features = [
    # Original physiological features
    'daily_total_steps',
    'activity_level_sedentary_count',
    'daily_respiratory_rate_daily_respiratory_rate',
    'minute_spo2_value_mean',
    'daily_hrv_summary_rmssd',
    'hrv_details_rmssd_min',
    'sleep_global_duration',
    'sleep_global_efficiency',
    'deep_sleep_minutes',
    'rem_sleep_minutes',
    'heart_rate_activity_beats per minute_mean',
    # Cluster
    'user_cluster',
    # Deviation features
    'hr_deviation',
    'activity_ratio',
    'hrv_deviation',
    'sleep_deviation',
    # Exam features
    'is_exam_period',
    'exam_proximity_inverse',
    'post_exam_proximity_inverse',
    # Interaction features
    'cluster_x_exam',
    'hr_dev_x_exam',
    'activity_x_exam',
    'hrv_dev_x_exam',
    # Composite features
    'sleep_quality',
    'activity_intensity',
    'autonomic_balance',
    'recovery_score',
    'cardio_stress',
    'sleep_fragmentation',
    'resp_efficiency'
]

print(f"Using {len(advanced_features)} advanced features")

# ============================================================================
# STEP 3: CREATE BINARY TARGETS (LOW vs HIGH, exclude MEDIUM)
# ============================================================================

def prepare_binary_data(df, target_col, feature_cols):
    """
    Prepare binary data: Low vs High (exclude Medium third)
    """
    print(f"\n{'='*80}")
    print(f"PREPARING BINARY DATA FOR: {target_col}")
    print(f"{'='*80}")
    
    # Remove rows with missing target
    df_clean = df[df[target_col].notna()].copy()
    print(f"\nRows with {target_col}: {len(df_clean)}")
    
    # Calculate percentiles
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    print(f"{target_col} 33rd percentile: {p33:.2f}")
    print(f"{target_col} 67th percentile: {p67:.2f}")
    
    # Create binary target: 0=Low (<p33), 1=High (>p67)
    df_binary = df_clean[
        (df_clean[target_col] < p33) | (df_clean[target_col] > p67)
    ].copy()
    
    df_binary['target_binary'] = (df_binary[target_col] > p67).astype(int)
    
    print(f"\nBinary dataset size: {len(df_binary)} observations")
    print(f"  Low (0): {(df_binary['target_binary']==0).sum()} samples ({(df_binary['target_binary']==0).sum()/len(df_binary)*100:.1f}%)")
    print(f"  High (1): {(df_binary['target_binary']==1).sum()} samples ({(df_binary['target_binary']==1).sum()/len(df_binary)*100:.1f}%)")
    
    # Prepare features
    X = df_binary[feature_cols].copy()
    y = df_binary['target_binary']
    
    # Split: 70-15-15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Imputation with median
    print(f"\nImputing missing values with median...")
    imputer = SimpleImputer(strategy='median')
    
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    # Normalization
    print(f"Normalizing features...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, p33, p67)

# ============================================================================
# STEP 4: TRAIN INDIVIDUAL MODELS AND ENSEMBLE + PLOT ROC CURVES
# ============================================================================

def train_evaluate_binary_models(X_train, X_val, X_test, y_train, y_val, y_test, target_name, roc_output_path):
    """Train and evaluate binary classification models + ensemble + plot ROC curves"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING MODELS FOR: {target_name}")
    print(f"{'='*80}")
    
    # Define base models
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                                n_jobs=-1, class_weight='balanced')
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                       random_state=42, eval_metric='logloss', 
                       use_label_encoder=False, scale_pos_weight=1)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, 
              probability=True, class_weight='balanced')
    nb = GaussianNB()
    
    models = {
        'Logistic Regression': lr,
        'Random Forest': rf,
        'XGBoost': xgb,
        'SVM (RBF)': svm,
        'Naive Bayes': nb
    }
    
    results = {}
    roc_data = {}  # Store ROC curve data for plotting
    
    for model_name, model in models.items():
        print(f"\n{'-'*80}")
        print(f"Training: {model_name}")
        print(f"{'-'*80}")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # Get probabilities for AUC and ROC curve
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_test_proba)
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}
        else:
            auc = np.nan
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        results[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': auc
        }
        
        print(f"\nTrain Accuracy: {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        if not np.isnan(auc):
            print(f"Test AUC: {auc:.4f}")
        
        print("\nTest Set - Classification Report:")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=['Low (0)', 'High (1)'],
                                   digits=4))
        
        print("\nTest Set - Confusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        print(f"  Rows: True labels (Low=0, High=1)")
        print(f"  Cols: Predicted labels")
        
        # Check for overfitting
        overfit_score = train_acc - test_acc
        if overfit_score > 0.1:
            print(f"\n⚠️  Warning: Possible overfitting (train-test gap: {overfit_score:.4f})")
        elif overfit_score > 0.05:
            print(f"\n⚠️  Moderate overfitting detected (train-test gap: {overfit_score:.4f})")
        else:
            print(f"\n✓ Good generalization (train-test gap: {overfit_score:.4f})")
    
    # ============================================================================
    # ENSEMBLE: Soft Voting (XGBoost + Random Forest + Logistic Regression)
    # ============================================================================
    
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MODEL: Soft Voting (XGBoost + RF + LR)")
    print(f"{'='*80}")
    
    # Create new instances for ensemble
    ensemble_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    ensemble_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, 
                                        n_jobs=-1, class_weight='balanced')
    ensemble_xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                random_state=42, eval_metric='logloss', 
                                use_label_encoder=False, scale_pos_weight=1)
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', ensemble_lr),
            ('rf', ensemble_rf),
            ('xgb', ensemble_xgb)
        ],
        voting='soft'
    )
    
    print("\nTraining ensemble...")
    voting_clf.fit(X_train, y_train)
    
    # Predict
    y_train_pred_ens = voting_clf.predict(X_train)
    y_val_pred_ens = voting_clf.predict(X_val)
    y_test_pred_ens = voting_clf.predict(X_test)
    y_test_proba_ens = voting_clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc_ens = accuracy_score(y_train, y_train_pred_ens)
    val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
    test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
    test_f1_ens = f1_score(y_test, y_test_pred_ens)
    test_precision_ens = precision_score(y_test, y_test_pred_ens)
    test_recall_ens = recall_score(y_test, y_test_pred_ens)
    auc_ens = roc_auc_score(y_test, y_test_proba_ens)
    
    # ROC curve for ensemble
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_test_proba_ens)
    roc_data['Ensemble (Soft Voting)'] = {'fpr': fpr_ens, 'tpr': tpr_ens, 'auc': auc_ens}
    
    results['Ensemble (Soft Voting)'] = {
        'train_accuracy': train_acc_ens,
        'val_accuracy': val_acc_ens,
        'test_accuracy': test_acc_ens,
        'test_f1': test_f1_ens,
        'test_precision': test_precision_ens,
        'test_recall': test_recall_ens,
        'test_auc': auc_ens
    }
    
    print(f"\nTrain Accuracy: {train_acc_ens:.4f}")
    print(f"Validation Accuracy: {val_acc_ens:.4f}")
    print(f"Test Accuracy: {test_acc_ens:.4f}")
    print(f"Test F1: {test_f1_ens:.4f}")
    print(f"Test Precision: {test_precision_ens:.4f}")
    print(f"Test Recall: {test_recall_ens:.4f}")
    print(f"Test AUC: {auc_ens:.4f}")
    
    print("\nTest Set - Classification Report:")
    print(classification_report(y_test, y_test_pred_ens, 
                               target_names=['Low (0)', 'High (1)'],
                               digits=4))
    
    print("\nTest Set - Confusion Matrix:")
    cm_ens = confusion_matrix(y_test, y_test_pred_ens)
    print(cm_ens)
    
    overfit_score_ens = train_acc_ens - test_acc_ens
    if overfit_score_ens > 0.1:
        print(f"\n⚠️  Warning: Possible overfitting (train-test gap: {overfit_score_ens:.4f})")
    elif overfit_score_ens > 0.05:
        print(f"\n⚠️  Moderate overfitting detected (train-test gap: {overfit_score_ens:.4f})")
    else:
        print(f"\n✓ Good generalization (train-test gap: {overfit_score_ens:.4f})")
    
    # ============================================================================
    # PLOT ROC CURVES
    # ============================================================================
    
    print(f"\n{'='*80}")
    print(f"GENERATING ROC CURVE PLOT")
    print(f"{'='*80}")
    
    plt.figure(figsize=(10, 8))
    
    # Define colors for each model
    colors = {
        'Logistic Regression': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'SVM (RBF)': '#d62728',
        'Naive Bayes': '#9467bd',
        'Ensemble (Soft Voting)': '#000000'  # Black for ensemble
    }
    
    # Plot ROC curve for each model
    for model_name, data in roc_data.items():
        linestyle = '--' if model_name == 'Ensemble (Soft Voting)' else '-'
        linewidth = 3 if model_name == 'Ensemble (Soft Voting)' else 2
        plt.plot(data['fpr'], data['tpr'], 
                color=colors[model_name],
                linestyle=linestyle,
                linewidth=linewidth,
                label=f"{model_name} (AUC = {data['auc']:.3f})")
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k:', linewidth=1.5, label='Random Classifier (AUC = 0.500)')
    
    # Formatting
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves - {target_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    # Save figure
    plt.savefig(roc_output_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {roc_output_path}")
    plt.close()
    
    return results

# ============================================================================
# EXECUTE PIPELINE FOR STRESS AND ANXIETY
# ============================================================================

# Prepare binary data for stress
X_train_stress, X_val_stress, X_test_stress, y_train_stress, y_val_stress, y_test_stress, p33_stress, p67_stress = prepare_binary_data(
    df, 'stress_level', advanced_features
)

# Train models for stress
stress_results = train_evaluate_binary_models(
    X_train_stress, X_val_stress, X_test_stress,
    y_train_stress, y_val_stress, y_test_stress,
    "STRESS LEVEL (Binary)",
    ROC_STRESS_PATH
)

# Prepare binary data for anxiety
X_train_anxiety, X_val_anxiety, X_test_anxiety, y_train_anxiety, y_val_anxiety, y_test_anxiety, p33_anxiety, p67_anxiety = prepare_binary_data(
    df, 'anxiety_level', advanced_features
)

# Train models for anxiety
anxiety_results = train_evaluate_binary_models(
    X_train_anxiety, X_val_anxiety, X_test_anxiety,
    y_train_anxiety, y_val_anxiety, y_test_anxiety,
    "ANXIETY LEVEL (Binary)",
    ROC_ANXIETY_PATH
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY - BINARY CLASSIFICATION WITH CLUSTERING")
print("="*80)

print("\n" + "-"*80)
print("STRESS LEVEL PREDICTION (Binary: Low vs High)")
print("-"*80)
print(f"{'Model':<35} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12}")
print("-"*80)
for model_name, metrics in stress_results.items():
    auc_str = f"{metrics['test_auc']:.4f}" if not np.isnan(metrics['test_auc']) else "N/A"
    print(f"{model_name:<35} {metrics['val_accuracy']:<12.4f} {metrics['test_accuracy']:<12.4f} "
          f"{metrics['test_f1']:<12.4f} {auc_str:<12}")

print("\n" + "-"*80)
print("ANXIETY LEVEL PREDICTION (Binary: Low vs High)")
print("-"*80)
print(f"{'Model':<35} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUC':<12}")
print("-"*80)
for model_name, metrics in anxiety_results.items():
    auc_str = f"{metrics['test_auc']:.4f}" if not np.isnan(metrics['test_auc']) else "N/A"
    print(f"{model_name:<35} {metrics['val_accuracy']:<12.4f} {metrics['test_accuracy']:<12.4f} "
          f"{metrics['test_f1']:<12.4f} {auc_str:<12}")

# Find best models
best_stress_model = max(stress_results.items(), key=lambda x: x[1]['val_accuracy'])
best_anxiety_model = max(anxiety_results.items(), key=lambda x: x[1]['val_accuracy'])

print("\n" + "="*80)
print("BEST MODELS (based on validation accuracy)")
print("="*80)
print(f"\nStress Level: {best_stress_model[0]}")
print(f"  Validation Accuracy: {best_stress_model[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy: {best_stress_model[1]['test_accuracy']:.4f}")
print(f"  Test F1: {best_stress_model[1]['test_f1']:.4f}")
print(f"  Test AUC: {best_stress_model[1]['test_auc']:.4f}")

print(f"\nAnxiety Level: {best_anxiety_model[0]}")
print(f"  Validation Accuracy: {best_anxiety_model[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy: {best_anxiety_model[1]['test_accuracy']:.4f}")
print(f"  Test F1: {best_anxiety_model[1]['test_f1']:.4f}")
print(f"  Test AUC: {best_anxiety_model[1]['test_auc']:.4f}")

print("\n" + "="*80)
print("ROC CURVES SAVED")
print("="*80)
print(f"Stress ROC curve: {ROC_STRESS_PATH}")
print(f"Anxiety ROC curve: {ROC_ANXIETY_PATH}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results
results_df = pd.DataFrame({
    'Model': list(stress_results.keys()),
    'Stress_Val_Acc': [v['val_accuracy'] for v in stress_results.values()],
    'Stress_Test_Acc': [v['test_accuracy'] for v in stress_results.values()],
    'Stress_Test_F1': [v['test_f1'] for v in stress_results.values()],
    'Stress_Test_Precision': [v['test_precision'] for v in stress_results.values()],
    'Stress_Test_Recall': [v['test_recall'] for v in stress_results.values()],
    'Stress_Test_AUC': [v['test_auc'] for v in stress_results.values()],
    'Anxiety_Val_Acc': [v['val_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_Acc': [v['test_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_F1': [v['test_f1'] for v in anxiety_results.values()],
    'Anxiety_Test_Precision': [v['test_precision'] for v in anxiety_results.values()],
    'Anxiety_Test_Recall': [v['test_recall'] for v in anxiety_results.values()],
    'Anxiety_Test_AUC': [v['test_auc'] for v in anxiety_results.values()]
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")