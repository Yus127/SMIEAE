import pandas as pd
import numpy as np
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
warnings.filterwarnings('ignore')
import os

INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model2/model_results_binary_clustering_timeseries.csv'
ROC_OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model2/'
os.makedirs('/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model2', exist_ok=True)
print("BINARY CLASSIFICATION WITH CLUSTERING AND ADVANCED FEATURES")
print("Train: 70% | Validation: 15% | Test: 15% (TIME SERIES SPLIT)")

df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# CREATE USER CLUSTERS (K=2) BASED ON PHYSIOLOGICAL PROFILES

print("\n" + "="*80)

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

# ADVANCED FEATURE ENGINEERING (NON-LEAKAGE FEATURES ONLY)

print("\n" + "="*80)

# NOTE: Personal baselines will be calculated AFTER time series split to avoid leakage
# Here we only create features that don't require future information

# Exam proximity features
print("Creating exam proximity features...")
df['exam_proximity_inverse'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['post_exam_proximity_inverse'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)

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

# Define base feature set (features that will be available before split)
base_features = [
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
    # Exam features
    'is_exam_period',
    'exam_proximity_inverse',
    'post_exam_proximity_inverse',
    # Composite features
    'sleep_quality',
    'activity_intensity',
    'autonomic_balance',
    'recovery_score',
    'cardio_stress',
    'sleep_fragmentation',
    'resp_efficiency'
]

print(f"Using {len(base_features)} base features (before baseline calculations)")

# CREATE BINARY TARGETS (LOW vs HIGH, exclude MEDIUM)

def prepare_binary_data_timeseries(df, target_col, feature_cols):
    """
    Prepare binary data with time series split: Low vs High (exclude Medium third)
    70-15-15 split maintaining temporal order
    FIXED: Baselines calculated only on training data to prevent leakage
    """
    print(f"\n{'='*80}")
    print(f"PREPARING BINARY DATA FOR: {target_col} (TIME SERIES SPLIT - NO LEAKAGE)")
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
    
    # Prepare base features (without baselines)
    X_base = df_binary[feature_cols].copy()
    y = df_binary['target_binary']
    userid = df_binary['userid']
    
    # Time series split: 70-15-15
    n_samples = len(X_base)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split sequentially (no shuffling for time series)
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size + val_size)
    test_idx = slice(train_size + val_size, None)
    
    X_train_base = X_base.iloc[train_idx]
    X_val_base = X_base.iloc[val_idx]
    X_test_base = X_base.iloc[test_idx]
    
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]
    
    userid_train = userid.iloc[train_idx]
    userid_val = userid.iloc[val_idx]
    userid_test = userid.iloc[test_idx]
    
    print(f"\nData split (time series):")
    print(f"  Train: {len(X_train_base)} samples ({len(X_train_base)/n_samples*100:.1f}%)")
    print(f"  Validation: {len(X_val_base)} samples ({len(X_val_base)/n_samples*100:.1f}%)")
    print(f"  Test: {len(X_test_base)} samples ({len(X_test_base)/n_samples*100:.1f}%)")
    
    print(f"\nClass distribution:")
    print(f"  Train - Low: {(y_train==0).sum()}, High: {(y_train==1).sum()}")
    print(f"  Val - Low: {(y_val==0).sum()}, High: {(y_val==1).sum()}")
    print(f"  Test - Low: {(y_test==0).sum()}, High: {(y_test==1).sum()}")
    
    # CALCULATE BASELINES ONLY ON TRAINING DATA (FIX FOR LEAKAGE)
    
    print(f"\n{'='*40}")
    print(f"CALCULATING BASELINES (TRAIN ONLY)")
    print(f"{'='*40}")
    
    # Get original indices for training data
    train_original_idx = df_binary.iloc[train_idx].index
    
    # Calculate baselines using ONLY training data
    user_baselines = df.loc[train_original_idx].groupby('userid').agg({
        'heart_rate_activity_beats per minute_mean': 'mean',
        'daily_total_steps': 'mean',
        'daily_hrv_summary_rmssd': 'mean',
        'daily_respiratory_rate_daily_respiratory_rate': 'mean',
        'sleep_global_duration': 'mean'
    }).reset_index()
    
    user_baselines.columns = ['userid', 'hr_baseline', 'steps_baseline', 'hrv_baseline', 
                              'resp_baseline', 'sleep_baseline']
    
    print(f"Baselines calculated for {len(user_baselines)} users from training data")
    
    # Create function to add baseline features
    def add_baseline_features(X_base_df, userid_series, user_baselines):
        X_with_baselines = X_base_df.copy()
        
        # Merge baselines (users not in train will get NaN, filled with global mean)
        X_with_baselines['userid'] = userid_series.values
        X_with_baselines = X_with_baselines.merge(user_baselines, on='userid', how='left')
        
        # Fill missing baselines with global train baselines
        for col in ['hr_baseline', 'steps_baseline', 'hrv_baseline', 'resp_baseline', 'sleep_baseline']:
            X_with_baselines[col].fillna(user_baselines[col].mean(), inplace=True)
        
        # Create deviation features
        X_with_baselines['hr_deviation'] = (X_with_baselines['heart_rate_activity_beats per minute_mean'] - X_with_baselines['hr_baseline']) / (X_with_baselines['hr_baseline'] + 1e-6)
        X_with_baselines['activity_ratio'] = (X_with_baselines['daily_total_steps']) / (X_with_baselines['steps_baseline'] + 1)
        X_with_baselines['hrv_deviation'] = (X_with_baselines['daily_hrv_summary_rmssd'] - X_with_baselines['hrv_baseline']) / (X_with_baselines['hrv_baseline'] + 1e-6)
        X_with_baselines['resp_deviation'] = (X_with_baselines['daily_respiratory_rate_daily_respiratory_rate'] - X_with_baselines['resp_baseline']) / (X_with_baselines['resp_baseline'] + 1e-6)
        X_with_baselines['sleep_deviation'] = (X_with_baselines['sleep_global_duration'] - X_with_baselines['sleep_baseline']) / (X_with_baselines['sleep_baseline'] + 1e-6)
        
        # Create interaction features
        X_with_baselines['cluster_x_exam'] = X_with_baselines['user_cluster'] * X_with_baselines['is_exam_period']
        X_with_baselines['hr_dev_x_exam'] = X_with_baselines['hr_deviation'] * X_with_baselines['is_exam_period']
        X_with_baselines['activity_x_exam'] = X_with_baselines['activity_ratio'] * X_with_baselines['is_exam_period']
        X_with_baselines['hrv_dev_x_exam'] = X_with_baselines['hrv_deviation'] * X_with_baselines['is_exam_period']
        
        # Drop userid and baseline columns, keep only features
        X_with_baselines = X_with_baselines.drop(['userid', 'hr_baseline', 'steps_baseline', 
                                                   'hrv_baseline', 'resp_baseline', 'sleep_baseline'], axis=1)
        
        return X_with_baselines
    
    # Add baseline features to all sets
    print("Adding baseline-derived features to train/val/test sets...")
    X_train = add_baseline_features(X_train_base, userid_train, user_baselines)
    X_val = add_baseline_features(X_val_base, userid_val, user_baselines)
    X_test = add_baseline_features(X_test_base, userid_test, user_baselines)
    
    print(f"Final feature count: {X_train.shape[1]}")
    
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

# TRAIN INDIVIDUAL MODELS AND ENSEMBLE

def train_evaluate_binary_models(X_train, X_val, X_test, y_train, y_val, y_test, target_name):
    """Train and evaluate binary classification models + ensemble"""
    
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
        
        # Get probabilities for AUC
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_val_proba = model.predict_proba(X_val)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, y_train_proba)
            val_auc = roc_auc_score(y_val, y_val_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            # Store ROC curve data for test set
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_data[model_name] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}
        else:
            train_auc = val_auc = test_auc = np.nan
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_f1 = f1_score(y_train, y_train_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        train_precision = precision_score(y_train, y_train_pred)
        val_precision = precision_score(y_val, y_val_pred)
        test_precision = precision_score(y_test, y_test_pred)
        
        train_recall = recall_score(y_train, y_train_pred)
        val_recall = recall_score(y_val, y_val_pred)
        test_recall = recall_score(y_test, y_test_pred)
        
        results[model_name] = {
            'train_accuracy': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'train_auc': train_auc,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_auc': val_auc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc
        }
        
        print(f"\nTrain Metrics:")
        print(f"  Accuracy:  {train_acc:.4f}")
        print(f"  Precision: {train_precision:.4f}")
        print(f"  Recall:    {train_recall:.4f}")
        print(f"  F1 Score:  {train_f1:.4f}")
        if not np.isnan(train_auc):
            print(f"  ROC AUC:   {train_auc:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {val_acc:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1 Score:  {val_f1:.4f}")
        if not np.isnan(val_auc):
            print(f"  ROC AUC:   {val_auc:.4f}")
        
        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {test_acc:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  F1 Score:  {test_f1:.4f}")
        if not np.isnan(test_auc):
            print(f"  ROC AUC:   {test_auc:.4f}")
        
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
            print(f"\n  Warning: Possible overfitting (train-test gap: {overfit_score:.4f})")
        elif overfit_score > 0.05:
            print(f"\n  Moderate overfitting detected (train-test gap: {overfit_score:.4f})")
        else:
            print(f"\n Good generalization (train-test gap: {overfit_score:.4f})")
    
    # ENSEMBLE: Soft Voting (XGBoost + Random Forest + Logistic Regression)
    
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
    
    voting_clf.fit(X_train, y_train)
    
    # Predict
    y_train_pred_ens = voting_clf.predict(X_train)
    y_val_pred_ens = voting_clf.predict(X_val)
    y_test_pred_ens = voting_clf.predict(X_test)
    
    y_train_proba_ens = voting_clf.predict_proba(X_train)[:, 1]
    y_val_proba_ens = voting_clf.predict_proba(X_val)[:, 1]
    y_test_proba_ens = voting_clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    train_acc_ens = accuracy_score(y_train, y_train_pred_ens)
    val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
    test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
    
    train_f1_ens = f1_score(y_train, y_train_pred_ens)
    val_f1_ens = f1_score(y_val, y_val_pred_ens)
    test_f1_ens = f1_score(y_test, y_test_pred_ens)
    
    train_precision_ens = precision_score(y_train, y_train_pred_ens)
    val_precision_ens = precision_score(y_val, y_val_pred_ens)
    test_precision_ens = precision_score(y_test, y_test_pred_ens)
    
    train_recall_ens = recall_score(y_train, y_train_pred_ens)
    val_recall_ens = recall_score(y_val, y_val_pred_ens)
    test_recall_ens = recall_score(y_test, y_test_pred_ens)
    
    train_auc_ens = roc_auc_score(y_train, y_train_proba_ens)
    val_auc_ens = roc_auc_score(y_val, y_val_proba_ens)
    test_auc_ens = roc_auc_score(y_test, y_test_proba_ens)
    
    # Store ROC curve data for ensemble
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_test_proba_ens)
    roc_data['Ensemble (Soft Voting)'] = {'fpr': fpr_ens, 'tpr': tpr_ens, 'auc': test_auc_ens}
    
    results['Ensemble (Soft Voting)'] = {
        'train_accuracy': train_acc_ens,
        'train_precision': train_precision_ens,
        'train_recall': train_recall_ens,
        'train_f1': train_f1_ens,
        'train_auc': train_auc_ens,
        'val_accuracy': val_acc_ens,
        'val_precision': val_precision_ens,
        'val_recall': val_recall_ens,
        'val_f1': val_f1_ens,
        'val_auc': val_auc_ens,
        'test_accuracy': test_acc_ens,
        'test_precision': test_precision_ens,
        'test_recall': test_recall_ens,
        'test_f1': test_f1_ens,
        'test_auc': test_auc_ens
    }
    
    print(f"\nTrain Metrics:")
    print(f"  Accuracy:  {train_acc_ens:.4f}")
    print(f"  Precision: {train_precision_ens:.4f}")
    print(f"  Recall:    {train_recall_ens:.4f}")
    print(f"  F1 Score:  {train_f1_ens:.4f}")
    print(f"  ROC AUC:   {train_auc_ens:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_acc_ens:.4f}")
    print(f"  Precision: {val_precision_ens:.4f}")
    print(f"  Recall:    {val_recall_ens:.4f}")
    print(f"  F1 Score:  {val_f1_ens:.4f}")
    print(f"  ROC AUC:   {val_auc_ens:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_acc_ens:.4f}")
    print(f"  Precision: {test_precision_ens:.4f}")
    print(f"  Recall:    {test_recall_ens:.4f}")
    print(f"  F1 Score:  {test_f1_ens:.4f}")
    print(f"  ROC AUC:   {test_auc_ens:.4f}")
    
    print("\nTest Set - Classification Report:")
    print(classification_report(y_test, y_test_pred_ens, 
                               target_names=['Low (0)', 'High (1)'],
                               digits=4))
    
    print("\nTest Set - Confusion Matrix:")
    cm_ens = confusion_matrix(y_test, y_test_pred_ens)
    print(cm_ens)
    
    overfit_score_ens = train_acc_ens - test_acc_ens
    if overfit_score_ens > 0.1:
        print(f"\n  Warning: Possible overfitting (train-test gap: {overfit_score_ens:.4f})")
    elif overfit_score_ens > 0.05:
        print(f"\n  Moderate overfitting detected (train-test gap: {overfit_score_ens:.4f})")
    else:
        print(f"\n Good generalization (train-test gap: {overfit_score_ens:.4f})")
    
    return results, roc_data

# FUNCTION TO PLOT ROC CURVES

def plot_roc_curves(roc_data, target_name, save_path):
    """Plot ROC curves for all models"""
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for each model
    colors = {
        'Logistic Regression': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'XGBoost': '#2ca02c',
        'SVM (RBF)': '#d62728',
        'Naive Bayes': '#9467bd',
        'Ensemble (Soft Voting)': '#e377c2'
    }
    
    # Plot ROC curve for each model
    for model_name, data in roc_data.items():
        plt.plot(data['fpr'], data['tpr'], 
                color=colors.get(model_name, 'black'),
                lw=2.5 if 'Ensemble' in model_name else 2,
                linestyle='-' if 'Ensemble' in model_name else '--' if model_name == 'Naive Bayes' else '-',
                label=f"{model_name} (AUC = {data['auc']:.3f})",
                alpha=0.9)
    
    # Plot diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random Classifier (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves - {target_name}', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Add text box with info
    textstr = 'Time Series Split (70-15-15)\nBinary Classification: Low vs High'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"roc_curve_{target_name.lower().replace(' ', '_')}.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"\n ROC curve saved to: {full_path}")
    
    plt.close()

# EXECUTE PIPELINE FOR STRESS AND ANXIETY

# Prepare binary data for stress with time series split (NO LEAKAGE)
X_train_stress, X_val_stress, X_test_stress, y_train_stress, y_val_stress, y_test_stress, p33_stress, p67_stress = prepare_binary_data_timeseries(
    df, 'stress_level', base_features
)

# Train models for stress
stress_results, stress_roc_data = train_evaluate_binary_models(
    X_train_stress, X_val_stress, X_test_stress,
    y_train_stress, y_val_stress, y_test_stress,
    "STRESS LEVEL (Binary)"
)

# Plot ROC curves for stress
plot_roc_curves(stress_roc_data, "Stress Level", ROC_OUTPUT_PATH)

# Prepare binary data for anxiety with time series split (NO LEAKAGE)
X_train_anxiety, X_val_anxiety, X_test_anxiety, y_train_anxiety, y_val_anxiety, y_test_anxiety, p33_anxiety, p67_anxiety = prepare_binary_data_timeseries(
    df, 'anxiety_level', base_features
)

# Train models for anxiety
anxiety_results, anxiety_roc_data = train_evaluate_binary_models(
    X_train_anxiety, X_val_anxiety, X_test_anxiety,
    y_train_anxiety, y_val_anxiety, y_test_anxiety,
    "ANXIETY LEVEL (Binary)"
)

# Plot ROC curves for anxiety
plot_roc_curves(anxiety_roc_data, "Anxiety Level", ROC_OUTPUT_PATH)

# FINAL SUMMARY

print("\n" + "="*80)
print("FINAL SUMMARY - BINARY CLASSIFICATION WITH CLUSTERING (TIME SERIES - NO LEAKAGE)")

print("\n" + "-"*80)
print("STRESS LEVEL PREDICTION (Binary: Low vs High)")
print("-"*80)
print(f"{'Model':<35} {'Val Acc':<10} {'Test Acc':<10} {'Test Prec':<10} {'Test Rec':<10} {'Test F1':<10} {'Test AUC':<10}")
print("-"*80)
for model_name, metrics in stress_results.items():
    auc_str = f"{metrics['test_auc']:.4f}" if not np.isnan(metrics['test_auc']) else "N/A"
    print(f"{model_name:<35} {metrics['val_accuracy']:<10.4f} {metrics['test_accuracy']:<10.4f} "
          f"{metrics['test_precision']:<10.4f} {metrics['test_recall']:<10.4f} "
          f"{metrics['test_f1']:<10.4f} {auc_str:<10}")

print("\n" + "-"*80)
print("ANXIETY LEVEL PREDICTION (Binary: Low vs High)")
print("-"*80)
print(f"{'Model':<35} {'Val Acc':<10} {'Test Acc':<10} {'Test Prec':<10} {'Test Rec':<10} {'Test F1':<10} {'Test AUC':<10}")
print("-"*80)
for model_name, metrics in anxiety_results.items():
    auc_str = f"{metrics['test_auc']:.4f}" if not np.isnan(metrics['test_auc']) else "N/A"
    print(f"{model_name:<35} {metrics['val_accuracy']:<10.4f} {metrics['test_accuracy']:<10.4f} "
          f"{metrics['test_precision']:<10.4f} {metrics['test_recall']:<10.4f} "
          f"{metrics['test_f1']:<10.4f} {auc_str:<10}")

# Find best models
best_stress_model = max(stress_results.items(), key=lambda x: x[1]['val_accuracy'])
best_anxiety_model = max(anxiety_results.items(), key=lambda x: x[1]['val_accuracy'])

print("\n" + "="*80)
print("BEST MODELS (based on validation accuracy)")
print(f"\nStress Level: {best_stress_model[0]}")
print(f"  Train Accuracy:      {best_stress_model[1]['train_accuracy']:.4f}")
print(f"  Validation Accuracy: {best_stress_model[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy:       {best_stress_model[1]['test_accuracy']:.4f}")
print(f"  Test Precision:      {best_stress_model[1]['test_precision']:.4f}")
print(f"  Test Recall:         {best_stress_model[1]['test_recall']:.4f}")
print(f"  Test F1:             {best_stress_model[1]['test_f1']:.4f}")
print(f"  Test AUC:            {best_stress_model[1]['test_auc']:.4f}")

print(f"\nAnxiety Level: {best_anxiety_model[0]}")
print(f"  Train Accuracy:      {best_anxiety_model[1]['train_accuracy']:.4f}")
print(f"  Validation Accuracy: {best_anxiety_model[1]['val_accuracy']:.4f}")
print(f"  Test Accuracy:       {best_anxiety_model[1]['test_accuracy']:.4f}")
print(f"  Test Precision:      {best_anxiety_model[1]['test_precision']:.4f}")
print(f"  Test Recall:         {best_anxiety_model[1]['test_recall']:.4f}")
print(f"  Test F1:             {best_anxiety_model[1]['test_f1']:.4f}")
print(f"  Test AUC:            {best_anxiety_model[1]['test_auc']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")

results_df = pd.DataFrame({
    'Model': list(stress_results.keys()),
    'Stress_Train_Acc': [v['train_accuracy'] for v in stress_results.values()],
    'Stress_Val_Acc': [v['val_accuracy'] for v in stress_results.values()],
    'Stress_Test_Acc': [v['test_accuracy'] for v in stress_results.values()],
    'Stress_Test_Precision': [v['test_precision'] for v in stress_results.values()],
    'Stress_Test_Recall': [v['test_recall'] for v in stress_results.values()],
    'Stress_Test_F1': [v['test_f1'] for v in stress_results.values()],
    'Stress_Test_AUC': [v['test_auc'] for v in stress_results.values()],
    'Anxiety_Train_Acc': [v['train_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Val_Acc': [v['val_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_Acc': [v['test_accuracy'] for v in anxiety_results.values()],
    'Anxiety_Test_Precision': [v['test_precision'] for v in anxiety_results.values()],
    'Anxiety_Test_Recall': [v['test_recall'] for v in anxiety_results.values()],
    'Anxiety_Test_F1': [v['test_f1'] for v in anxiety_results.values()],
    'Anxiety_Test_AUC': [v['test_auc'] for v in anxiety_results.values()]
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print(f"ROC curves saved to: {ROC_OUTPUT_PATH}")