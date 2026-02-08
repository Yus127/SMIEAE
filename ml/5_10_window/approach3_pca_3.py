import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, roc_auc_score, roc_curve, auc,
                            confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ML PIPELINE: PCA + 3-CLASS CLASSIFICATION")
print("STRESS & ANXIETY - TIME SERIES SPLIT (70-15-15)")
print("SEPARATE ANALYSIS FOR 5MIN AND 10MIN WINDOWS")
print("SIMPLIFIED FEATURE ENGINEERING (NO USER CLUSTERING)")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/5_10_dataset/timeseries/model3"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Define files to process
files = [
    "enriched/ml_ready_5min_window_enriched.csv",
    "enriched/ml_ready_10min_window_enriched.csv"
]

# Store overall results
all_results = {}

# Process each window
for file_idx, file in enumerate(files):
    window_type = "5min" if "5min" in file else "10min"
    window_prefix = "w5" if "5min" in file else "w10"
    
    print("\n" + "="*80)
    print(f"PROCESSING: {window_type.upper()} WINDOW")
    print("="*80)
    
    # Create window-specific output directory
    window_output_dir = os.path.join(output_dir, f"{window_type}_window")
    os.makedirs(window_output_dir, exist_ok=True)
    os.makedirs(os.path.join(window_output_dir, "plots"), exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print(f"\n1. LOADING DATA ({window_type})")
    print("-"*80)
    
    data_file = os.path.join(ml_dir, file)
    if not os.path.exists(data_file):
        print(f"✗ File not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} observations")
    
    # Define base features
    base_feature_columns = [
        'is_exam_period', 'days_until_exam', 'is_pre_exam_week', 'is_easter_break',
        'daily_total_steps',
    ]
    
    # Add window-specific heart rate features
    hr_features = [
        f'{window_prefix}_heart_rate_activity_beats per minute_mean',
        f'{window_prefix}_heart_rate_activity_beats per minute_std',
        f'{window_prefix}_heart_rate_activity_beats per minute_min',
        f'{window_prefix}_heart_rate_activity_beats per minute_max',
        f'{window_prefix}_heart_rate_activity_beats per minute_median',
    ]
    base_feature_columns.extend(hr_features)
    
    # Add other physiological features if available
    optional_features = [
        'daily_hrv_summary_rmssd',
        'daily_respiratory_rate_daily_respiratory_rate',
        'sleep_global_duration',
        'sleep_global_efficiency',
        'deep_sleep_minutes',
        'rem_sleep_minutes',
        'wake_count',
        'minute_spo2_value_mean',
        'activity_level_sedentary_count',
        'hrv_details_rmssd_min'
    ]
    
    available_base_features = [col for col in base_feature_columns if col in df.columns]
    available_optional_features = [col for col in optional_features if col in df.columns]
    
    # Identify targets
    target_columns = [col for col in df.columns if col.startswith('q_i_stress') or col.startswith('q_i_anxiety')]
    
    # Stress target
    if 'q_i_stress_sliderNeutralPos' in df.columns:
        stress_target = 'q_i_stress_sliderNeutralPos'
    else:
        stress_cols = [col for col in target_columns if 'stress' in col.lower()]
        stress_target = stress_cols[0] if stress_cols else None
    
    # Anxiety target
    if 'q_i_anxiety_sliderNeutralPos' in df.columns:
        anxiety_target = 'q_i_anxiety_sliderNeutralPos'
    else:
        anxiety_cols = [col for col in target_columns if 'anxiety' in col.lower()]
        anxiety_target = anxiety_cols[0] if anxiety_cols else None
    
    # Create targets list
    targets_to_process = []
    if stress_target:
        targets_to_process.append(('Stress', stress_target))
        print(f"✓ Stress target: {stress_target}")
    if anxiety_target:
        targets_to_process.append(('Anxiety', anxiety_target))
        print(f"✓ Anxiety target: {anxiety_target}")
    
    if not targets_to_process:
        print("✗ No stress or anxiety targets found!")
        continue
    
    print(f"\n✓ Will train models for {len(targets_to_process)} target(s): {', '.join([t[0] for t in targets_to_process])}")
    
    # Ensure userid column exists
    if 'userid' not in df.columns:
        df['userid'] = df['user_id'] if 'user_id' in df.columns else 0
    
    # Store results for this window
    all_results[window_type] = {}
    
    # Loop through each target
    for target_name, primary_target in targets_to_process:
        
        print("\n" + "#"*80)
        print(f"# TARGET: {target_name.upper()} ({window_type} window)")
        print("#"*80)
        
        # Create target-specific output directory
        target_output_dir = os.path.join(window_output_dir, target_name.lower())
        os.makedirs(target_output_dir, exist_ok=True)
        os.makedirs(os.path.join(target_output_dir, "plots"), exist_ok=True)
        
        # ====================================================================
        # 2. SORT BY TIME AND PREPARE DATA
        # ====================================================================
        print(f"\n2. DATA PREPARATION ({target_name} - {window_type})")
        print("-"*80)
        
        # Sort by timestamp (critical for time series split)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            print("✓ Data sorted by timestamp (chronological order)")
        elif 'date' in df.columns:
            df_sorted = df.sort_values('date').reset_index(drop=True)
            print("✓ Data sorted by date (chronological order)")
        elif 'datetime' in df.columns:
            df_sorted = df.sort_values('datetime').reset_index(drop=True)
            print("✓ Data sorted by datetime (chronological order)")
        else:
            df_sorted = df.copy()
            print("⚠ Warning: No timestamp column found. Assuming chronological order.")
        
        # Select base features and target
        df_clean = df_sorted[available_base_features + available_optional_features + [primary_target, 'userid']].copy()
        df_clean = df_clean.dropna(subset=[primary_target])
        
        print(f"✓ Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")
        
        # ====================================================================
        # 3. SIMPLIFIED FEATURE ENGINEERING (PERSONAL BASELINES)
        # ====================================================================
        print(f"\n3. FEATURE ENGINEERING - PERSONAL BASELINES ({target_name})")
        print("-"*80)
        
        # Calculate user baselines
        print("Calculating personal baselines...")
        
        baseline_features = {}
        
        # HR baseline
        hr_mean_feat = None
        for feat in hr_features:
            if 'mean' in feat and feat in df_clean.columns:
                hr_mean_feat = feat
                break
        
        if hr_mean_feat:
            baseline_features['hr_baseline'] = df_clean.groupby('userid')[hr_mean_feat].transform('mean')
        
        # Steps baseline
        if 'daily_total_steps' in df_clean.columns:
            baseline_features['steps_baseline'] = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
        
        # HRV baseline
        if 'daily_hrv_summary_rmssd' in df_clean.columns:
            baseline_features['hrv_baseline'] = df_clean.groupby('userid')['daily_hrv_summary_rmssd'].transform('mean')
        
        # Sleep baseline
        if 'sleep_global_duration' in df_clean.columns:
            baseline_features['sleep_baseline'] = df_clean.groupby('userid')['sleep_global_duration'].transform('mean')
        
        # Respiratory baseline
        if 'daily_respiratory_rate_daily_respiratory_rate' in df_clean.columns:
            baseline_features['resp_baseline'] = df_clean.groupby('userid')['daily_respiratory_rate_daily_respiratory_rate'].transform('mean')
        
        print(f"✓ Created {len(baseline_features)} baseline features")
        
        # ====================================================================
        # 4. CREATE DEVIATION AND INTERACTION FEATURES
        # ====================================================================
        print(f"\n4. CREATING DEVIATION AND INTERACTION FEATURES ({target_name})")
        print("-"*80)
        
        engineered_features = []
        
        # HR deviation
        if hr_mean_feat and 'hr_baseline' in baseline_features:
            df_clean['hr_deviation'] = (df_clean[hr_mean_feat] - baseline_features['hr_baseline']) / (baseline_features['hr_baseline'] + 1e-6)
            engineered_features.append('hr_deviation')
        
        # Activity ratio
        if 'steps_baseline' in baseline_features:
            df_clean['activity_ratio'] = df_clean['daily_total_steps'] / (baseline_features['steps_baseline'] + 1)
            engineered_features.append('activity_ratio')
        
        # HRV deviation
        if 'hrv_baseline' in baseline_features:
            df_clean['hrv_deviation'] = (df_clean['daily_hrv_summary_rmssd'] - baseline_features['hrv_baseline']) / (baseline_features['hrv_baseline'] + 1e-6)
            engineered_features.append('hrv_deviation')
        
        # Sleep deviation
        if 'sleep_baseline' in baseline_features:
            df_clean['sleep_deviation'] = (df_clean['sleep_global_duration'] - baseline_features['sleep_baseline']) / (baseline_features['sleep_baseline'] + 1e-6)
            engineered_features.append('sleep_deviation')
        
        # Exam proximity features
        if 'days_until_exam' in df_clean.columns:
            df_clean['exam_proximity_inverse'] = 1 / (df_clean['days_until_exam'].fillna(365).clip(lower=1) + 1)
            engineered_features.append('exam_proximity_inverse')
        
        # Interaction features with exam period
        if 'is_exam_period' in df_clean.columns:
            if 'hr_deviation' in engineered_features:
                df_clean['hr_dev_x_exam'] = df_clean['hr_deviation'] * df_clean['is_exam_period']
                engineered_features.append('hr_dev_x_exam')
            
            if 'activity_ratio' in engineered_features:
                df_clean['activity_x_exam'] = df_clean['activity_ratio'] * df_clean['is_exam_period']
                engineered_features.append('activity_x_exam')
            
            if 'hrv_deviation' in engineered_features:
                df_clean['hrv_dev_x_exam'] = df_clean['hrv_deviation'] * df_clean['is_exam_period']
                engineered_features.append('hrv_dev_x_exam')
            
            if 'exam_proximity_inverse' in engineered_features:
                if 'activity_ratio' in engineered_features:
                    df_clean['steps_x_exam_prox'] = df_clean['activity_ratio'] * df_clean['exam_proximity_inverse']
                    engineered_features.append('steps_x_exam_prox')
        
        # Advanced physiological features
        if 'sleep_global_efficiency' in df_clean.columns and 'deep_sleep_minutes' in df_clean.columns and 'sleep_global_duration' in df_clean.columns:
            df_clean['sleep_quality'] = (df_clean['sleep_global_efficiency'] * df_clean['deep_sleep_minutes']) / (df_clean['sleep_global_duration'] + 1)
            engineered_features.append('sleep_quality')
        
        if 'daily_total_steps' in df_clean.columns and 'activity_level_sedentary_count' in df_clean.columns:
            df_clean['activity_intensity'] = df_clean['daily_total_steps'] / (df_clean['activity_level_sedentary_count'] + 1)
            engineered_features.append('activity_intensity')
        
        if 'daily_hrv_summary_rmssd' in df_clean.columns and hr_mean_feat:
            df_clean['autonomic_balance'] = df_clean['daily_hrv_summary_rmssd'] / (df_clean[hr_mean_feat] + 1)
            engineered_features.append('autonomic_balance')
        
        if 'rem_sleep_minutes' in df_clean.columns and 'deep_sleep_minutes' in df_clean.columns and 'sleep_global_duration' in df_clean.columns:
            df_clean['recovery_score'] = (df_clean['rem_sleep_minutes'] + df_clean['deep_sleep_minutes']) / (df_clean['sleep_global_duration'] + 1)
            engineered_features.append('recovery_score')
        
        if hr_mean_feat and 'daily_hrv_summary_rmssd' in df_clean.columns:
            df_clean['cardio_stress'] = df_clean[hr_mean_feat] / (df_clean['daily_hrv_summary_rmssd'] + 1)
            engineered_features.append('cardio_stress')
        
        if 'wake_count' in df_clean.columns and 'sleep_global_duration' in df_clean.columns:
            df_clean['sleep_fragmentation'] = df_clean['wake_count'] / (df_clean['sleep_global_duration'] / 60 + 1)
            engineered_features.append('sleep_fragmentation')
        
        if 'minute_spo2_value_mean' in df_clean.columns and 'daily_respiratory_rate_daily_respiratory_rate' in df_clean.columns:
            df_clean['resp_efficiency'] = df_clean['minute_spo2_value_mean'] / (df_clean['daily_respiratory_rate_daily_respiratory_rate'] + 1)
            engineered_features.append('resp_efficiency')
        
        print(f"✓ Created {len(engineered_features)} engineered features")
        
        # ====================================================================
        # 5. PREPARE FINAL FEATURE SET
        # ====================================================================
        print(f"\n5. PREPARING FINAL FEATURE SET ({target_name})")
        print("-"*80)
        
        # Combine all features
        all_features = available_base_features + available_optional_features + engineered_features
        
        print(f"  Total features before PCA: {len(all_features)}")
        print(f"    Base features: {len(available_base_features)}")
        print(f"    Optional features: {len(available_optional_features)}")
        print(f"    Engineered features: {len(engineered_features)}")
        
        df_model = df_clean[all_features + [primary_target]].copy()
        df_model = df_model.dropna()
        
        # Impute any remaining missing values with median
        for col in all_features:
            if df_model[col].isna().sum() > 0:
                median_val = df_model[col].median()
                df_model.loc[:, col] = df_model[col].fillna(median_val)
        
        # Clean extreme values (clip at 0.1% and 99.9% percentiles)
        for col in all_features:
            if df_model[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                upper = df_model[col].quantile(0.999)
                lower = df_model[col].quantile(0.001)
                df_model[col] = df_model[col].clip(lower=lower, upper=upper)
        
        print(f"✓ Clean dataset: {len(df_model)} observations")
        
        # ====================================================================
        # 6. CREATE 3-CLASS TARGET
        # ====================================================================
        print(f"\n6. CREATING 3-CLASS TARGET ({target_name})")
        print("-"*80)
        
        y_continuous = df_model[primary_target]
        p33 = np.percentile(y_continuous, 33.33)
        p67 = np.percentile(y_continuous, 66.67)
        
        y_3class = pd.cut(y_continuous,
                          bins=[-np.inf, p33, p67, np.inf],
                          labels=[0, 1, 2],
                          include_lowest=True).astype(int)
        
        print(f"  Class 0 (Low):    {(y_3class == 0).sum()} observations (≤{p33:.1f})")
        print(f"  Class 1 (Medium): {(y_3class == 1).sum()} observations ({p33:.1f}-{p67:.1f})")
        print(f"  Class 2 (High):   {(y_3class == 2).sum()} observations (≥{p67:.1f})")
        
        X = df_model[all_features]
        
        # ====================================================================
        # 7. TIME SERIES SPLIT (70-15-15)
        # ====================================================================
        print(f"\n7. TIME SERIES SPLIT: 70% TRAIN - 15% VAL - 15% TEST ({target_name})")
        print("-"*80)
        print("Using chronological split (no shuffling) to respect temporal order")
        
        # Calculate split indices
        n_samples = len(X)
        train_size = int(0.70 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Split chronologically
        X_train = X.iloc[:train_size]
        y_train = y_3class.iloc[:train_size]
        
        X_val = X.iloc[train_size:train_size+val_size]
        y_val = y_3class.iloc[train_size:train_size+val_size]
        
        X_test = X.iloc[train_size+val_size:]
        y_test = y_3class.iloc[train_size+val_size:]
        
        print(f"  Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_train==0).sum()}, Medium: {(y_train==1).sum()}, High: {(y_train==2).sum()}")
        print(f"  Val:   {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_val==0).sum()}, Medium: {(y_val==1).sum()}, High: {(y_val==2).sum()}")
        print(f"  Test:  {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_test==0).sum()}, Medium: {(y_test==1).sum()}, High: {(y_test==2).sum()}")
        
        # ====================================================================
        # 8. APPLY PCA TO FEATURES (SINGLE PCA STEP)
        # ====================================================================
        print(f"\n8. APPLYING PCA TO FEATURE SET ({target_name})")
        print("-"*80)
        
        # First normalize
        scaler_features = StandardScaler()
        X_train_scaled = scaler_features.fit_transform(X_train)
        X_val_scaled = scaler_features.transform(X_val)
        X_test_scaled = scaler_features.transform(X_test)
        
        # Exploratory PCA to see variance
        pca_explore = PCA(n_components=None)
        pca_explore.fit(X_train_scaled)
        
        print(f"\n  Explained Variance (first 10 components):")
        for i in range(min(10, len(pca_explore.explained_variance_ratio_))):
            cumsum = pca_explore.explained_variance_ratio_[:i+1].sum()
            print(f"    PC{i+1}: {pca_explore.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumsum*100:.1f}%)")
        
        # Choose components to retain 80% variance (like Code 2)
        n_components = np.argmax(pca_explore.explained_variance_ratio_.cumsum() >= 0.80) + 1
        print(f"\n  → Using {n_components} components (≥80% variance)")
        
        # Apply PCA with selected components
        pca_final = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca_final.fit_transform(X_train_scaled)
        X_val_pca = pca_final.transform(X_val_scaled)
        X_test_pca = pca_final.transform(X_test_scaled)
        
        print(f"\n✓ Dimensionality reduction: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} features")
        print(f"  Retained variance: {pca_final.explained_variance_ratio_.sum()*100:.1f}%")
        
        # ====================================================================
        # 9. TRAIN MODELS ON PCA FEATURES
        # ====================================================================
        print(f"\n9. TRAINING MODELS - 3-CLASS {target_name.upper()} ({window_type})")
        print("="*80)
        
        models = {
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                          random_state=42, objective='multi:softmax', num_class=3),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, 
                                                   random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, 
                                                      multi_class='multinomial'),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, 
                       decision_function_shape='ovr')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name}...")
            model.fit(X_train_pca, y_train)
            
            # Validation
            y_val_pred = model.predict(X_val_pca)
            y_val_proba = model.predict_proba(X_val_pca)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1_per_class = f1_score(y_val, y_val_pred, average=None, zero_division=0)
            
            # Validation AUC
            try:
                y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
                val_auc = roc_auc_score(y_val_bin, y_val_proba, average='weighted', multi_class='ovr')
            except:
                val_auc = np.nan
            
            # Test
            y_test_pred = model.predict(X_test_pca)
            y_test_proba = model.predict_proba(X_test_pca)
            
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0)
            
            # Multi-class AUC
            try:
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                test_auc = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
            except:
                test_auc = np.nan
            
            print(f"  Val:  F1={val_f1:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
            print(f"        F1 per class: Low={val_f1_per_class[0]:.3f}, Med={val_f1_per_class[1]:.3f}, High={val_f1_per_class[2]:.3f}")
            print(f"  Test: F1={test_f1:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")
            print(f"        F1 per class: Low={test_f1_per_class[0]:.3f}, Med={test_f1_per_class[1]:.3f}, High={test_f1_per_class[2]:.3f}")
            
            results[name] = {
                'val_f1': val_f1, 'val_acc': val_acc, 'val_auc': val_auc,
                'test_f1': test_f1, 'test_acc': test_acc, 'test_auc': test_auc,
                'test_f1_per_class': test_f1_per_class,
                'y_val_pred': y_val_pred, 'y_val_proba': y_val_proba,
                'y_test_pred': y_test_pred, 'y_test_proba': y_test_proba
            }
        
        # ====================================================================
        # 10. ENSEMBLE MODEL
        # ====================================================================
        print(f"\n10. ENSEMBLE MODEL ({target_name})")
        print("-"*80)
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', models['XGBoost']),
                ('rf', models['Random Forest']),
                ('lr', models['Logistic Regression'])
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train_pca, y_train)
        
        y_val_pred_ens = ensemble.predict(X_val_pca)
        y_val_proba_ens = ensemble.predict_proba(X_val_pca)
        y_test_pred_ens = ensemble.predict(X_test_pca)
        y_test_proba_ens = ensemble.predict_proba(X_test_pca)
        
        val_f1_ens = f1_score(y_val, y_val_pred_ens, average='weighted', zero_division=0)
        val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
        
        try:
            y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
            val_auc_ens = roc_auc_score(y_val_bin, y_val_proba_ens, average='weighted', multi_class='ovr')
        except:
            val_auc_ens = np.nan
        
        test_f1_ens = f1_score(y_test, y_test_pred_ens, average='weighted', zero_division=0)
        test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
        test_f1_per_class_ens = f1_score(y_test, y_test_pred_ens, average=None, zero_division=0)
        
        try:
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            test_auc_ens = roc_auc_score(y_test_bin, y_test_proba_ens, average='weighted', multi_class='ovr')
        except:
            test_auc_ens = np.nan
        
        print(f"  Val:  F1={val_f1_ens:.4f}, Acc={val_acc_ens:.4f}, AUC={val_auc_ens:.4f}")
        print(f"  Test: F1={test_f1_ens:.4f}, Acc={test_acc_ens:.4f}, AUC={test_auc_ens:.4f}")
        print(f"        F1 per class: Low={test_f1_per_class_ens[0]:.3f}, Med={test_f1_per_class_ens[1]:.3f}, High={test_f1_per_class_ens[2]:.3f}")
        
        results['Ensemble'] = {
            'val_f1': val_f1_ens, 'val_acc': val_acc_ens, 'val_auc': val_auc_ens,
            'test_f1': test_f1_ens, 'test_acc': test_acc_ens, 'test_auc': test_auc_ens,
            'test_f1_per_class': test_f1_per_class_ens,
            'y_val_pred': y_val_pred_ens, 'y_val_proba': y_val_proba_ens,
            'y_test_pred': y_test_pred_ens, 'y_test_proba': y_test_proba_ens
        }
        
        # ====================================================================
        # 11. GENERATE ROC CURVES
        # ====================================================================
        print(f"\n11. GENERATING ROC CURVES ({target_name} - {window_type})")
        print("="*80)
        
        # Binarize the test labels for ROC curve
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
        n_classes = 3
        
        # Define colors for each class
        class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        class_names = ['Low', 'Medium', 'High']
        
        # Define colors for models
        model_colors = {
            'XGBoost': '#FF6B6B',
            'Random Forest': '#4ECDC4',
            'Logistic Regression': '#45B7D1',
            'SVM': '#95E1D3',
            'Ensemble': '#F38181'
        }
        
        # Create individual ROC curve for each model (showing all 3 classes)
        for model_name, result in results.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            y_score = result['y_test_proba']
            
            # Compute ROC curve and AUC for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot ROC curve for each class
            for i, color in zip(range(n_classes), class_colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=2.5,
                       label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=13)
            ax.set_ylabel('True Positive Rate', fontsize=13)
            ax.set_title(f'ROC Curves - {model_name}\n3-Class {target_name} ({window_type} window)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_model_name = model_name.replace(' ', '_').lower()
            plt.savefig(os.path.join(target_output_dir, 'plots', f'roc_curve_{safe_model_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved ROC curve for {model_name}")
        
        # Create comparison ROC curve for each class
        print(f"\nGenerating comparison ROC curves (one per class)...")
        
        for class_idx, class_name in enumerate(class_names):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for model_name, result in results.items():
                y_score = result['y_test_proba']
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
                roc_auc_val = auc(fpr, tpr)
                
                color = model_colors.get(model_name, '#333333')
                ax.plot(fpr, tpr, color=color, lw=2.5,
                       label=f'{model_name} (AUC = {roc_auc_val:.3f})')
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=13)
            ax.set_ylabel('True Positive Rate', fontsize=13)
            ax.set_title(f'ROC Curve Comparison - All Models\n{class_name} {target_name} ({window_type} window)', 
                        fontsize=15, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(target_output_dir, 'plots', f'roc_curve_comparison_{class_name.lower()}_class.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved comparison ROC curve for {class_name} class")
        
        # Create micro-average and macro-average ROC curve for best model
        print(f"\nGenerating micro/macro-average ROC curves for best model...")
        
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best_result = results[best_model_name]
        y_score_best = best_result['y_test_proba']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Compute micro-average ROC curve and AUC
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score_best.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(fpr_micro, tpr_micro,
               label=f'Micro-average (AUC = {roc_auc_micro:.3f})',
               color='deeppink', linestyle=':', linewidth=3)
        
        # Compute macro-average ROC curve and AUC
        all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_score_best[:, i])[0] 
                                            for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_best[:, i])
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= n_classes
        roc_auc_macro = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr,
               label=f'Macro-average (AUC = {roc_auc_macro:.3f})',
               color='navy', linestyle=':', linewidth=3)
        
        # Plot individual class curves
        for i, color in zip(range(n_classes), class_colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_best[:, i])
            roc_auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'Multi-Class ROC - {best_model_name}\n{target_name} ({window_type} window)', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'roc_curve_multiclass_best_model.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved multi-class ROC curve (micro/macro averages)")
        
        # Validation vs Test ROC comparison for best model (High class)
        print(f"\nGenerating validation vs test ROC comparison...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        high_class_idx = 2  # High class
        
        # Validation ROC
        y_val_score = best_result['y_val_proba']
        fpr_val, tpr_val, _ = roc_curve(y_val_bin[:, high_class_idx], y_val_score[:, high_class_idx])
        auc_val = auc(fpr_val, tpr_val)
        ax.plot(fpr_val, tpr_val, color='#4ECDC4', lw=2.5,
               label=f'Validation (AUC = {auc_val:.3f})', linestyle='--')
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, high_class_idx], y_score_best[:, high_class_idx])
        auc_test = auc(fpr_test, tpr_test)
        ax.plot(fpr_test, tpr_test, color='#FF6B6B', lw=2.5,
               label=f'Test (AUC = {auc_test:.3f})')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'ROC Curve: Validation vs Test - {best_model_name}\nHigh {target_name} ({window_type} window)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'roc_curve_validation_vs_test.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved validation vs test ROC curve")
        
        # ====================================================================
        # 12. RESULTS
        # ====================================================================
        print(f"\n12. FINAL RESULTS ({target_name} - {window_type})")
        print("="*80)
        
        best_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best = results[best_name]
        
        print(f"\n🏆 BEST MODEL: {best_name}")
        print(f"   Validation F1:  {results[best_name]['val_f1']:.4f} (weighted)")
        print(f"   Validation AUC: {results[best_name]['val_auc']:.4f}")
        print(f"   Test F1:        {best['test_f1']:.4f} (weighted)")
        print(f"   Test Accuracy:  {best['test_acc']:.4f}")
        if 'test_auc' in best and not np.isnan(best['test_auc']):
            print(f"   Test AUC:       {best['test_auc']:.4f}")
        
        print(f"\n   Per-Class Performance (Test):")
        print(f"     Low (Class 0):    F1 = {best['test_f1_per_class'][0]:.4f}")
        print(f"     Medium (Class 1): F1 = {best['test_f1_per_class'][1]:.4f}")
        print(f"     High (Class 2):   F1 = {best['test_f1_per_class'][2]:.4f}")
        
        print(f"\n📊 PERFORMANCE WITH PCA:")
        print(f"   Original features:  {len(all_features)}")
        print(f"   PCA components:     {n_components} (retained {pca_final.explained_variance_ratio_.sum()*100:.1f}% variance)")
        print(f"   Test F1 (3-class):  {best['test_f1']:.4f}")
        
        # Store for overall comparison
        all_results[window_type][target_name] = {
            'best_model': best_name,
            'best_val_f1': results[best_name]['val_f1'],
            'best_val_auc': results[best_name]['val_auc'],
            'best_test_f1': best['test_f1'],
            'best_test_acc': best['test_acc'],
            'best_test_auc': best['test_auc'],
            'test_f1_per_class': best['test_f1_per_class'],
            'n_samples': len(df_model),
            'n_components': n_components,
            'n_features_original': len(all_features)
        }
        
        # Detailed classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT ({best_name}):")
        print(classification_report(y_test, best['y_test_pred'], 
                                   target_names=[f'Low {target_name}', f'Medium {target_name}', f'High {target_name}'],
                                   zero_division=0))
        
        # Save results
        comparison_df = pd.DataFrame([
            {'Model': name, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
            for name, res in results.items()
        ])
        comparison_df.to_csv(os.path.join(target_output_dir, 'results_pca_3class_simplified.csv'), index=False)
        print(f"\n✓ Saved: results_pca_3class_simplified.csv")
        
        # ====================================================================
        # 13. OTHER VISUALIZATIONS
        # ====================================================================
        print(f"\n13. CREATING OTHER VISUALIZATIONS ({target_name})")
        print("-"*80)
        
        # PCA Variance Explained
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        axes[0].bar(range(1, len(pca_explore.explained_variance_ratio_)+1), 
                   pca_explore.explained_variance_ratio_*100)
        axes[0].axvline(x=n_components, color='r', linestyle='--', label=f'{n_components} components')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Variance Explained (%)')
        axes[0].set_title(f'PCA Scree Plot - {target_name}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Cumulative variance
        cumsum = np.cumsum(pca_explore.explained_variance_ratio_)*100
        axes[1].plot(range(1, len(cumsum)+1), cumsum, 'bo-')
        axes[1].axhline(y=80, color='r', linestyle='--', label='80% threshold')
        axes[1].axvline(x=n_components, color='r', linestyle='--', label=f'{n_components} components')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained (%)')
        axes[1].set_title(f'Cumulative Variance - {target_name}')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'PCA Analysis - {target_name} ({window_type})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'pca_variance.png'), dpi=300)
        print("✓ Saved: pca_variance.png")
        plt.close()
        
        # Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute counts
        cm = confusion_matrix(y_test, best['y_test_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['Low', 'Medium', 'High'])
        disp.plot(ax=axes[0], cmap='Blues', values_format='d')
        axes[0].set_title(f'Confusion Matrix - {best_name}\n{target_name} ({window_type}) - Counts', fontweight='bold')
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, 
                                           display_labels=['Low', 'Medium', 'High'])
        disp_norm.plot(ax=axes[1], cmap='RdYlGn', values_format='.2f')
        axes[1].set_title(f'Confusion Matrix - {best_name}\n{target_name} ({window_type}) - Normalized', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'confusion_matrix_3class.png'), dpi=300)
        print("✓ Saved: confusion_matrix_3class.png")
        plt.close()
        
        # Model comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        models_list = list(results.keys())
        test_f1_list = [results[m]['test_f1'] for m in models_list]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax.barh(models_list, test_f1_list, color=colors[:len(models_list)])
        ax.set_xlabel('Test F1 Score (Weighted)', fontsize=12)
        ax.set_title(f'Model Comparison - 3-Class {target_name} ({window_type})', fontsize=14, fontweight='bold')
        ax.axvline(x=0.333, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_xlim(0, max(test_f1_list)*1.1)
        ax.legend()
        ax.grid(alpha=0.3, axis='x')
        
        for i, (model, f1) in enumerate(zip(models_list, test_f1_list)):
            ax.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'model_comparison.png'), dpi=300)
        print("✓ Saved: model_comparison.png")
        plt.close()
        
        # ====================================================================
        # 14. SUMMARY
        # ====================================================================
        print(f"\n14. SUMMARY ({target_name} - {window_type})")
        print("="*80)
        
        print(f"\n📊 ROC Curves Generated:")
        print(f"   • Individual curves for each model (5 plots, 3 classes each)")
        print(f"   • Comparison curves per class (3 plots)")
        print(f"   • Multi-class ROC with micro/macro averages (1 plot)")
        print(f"   • Validation vs Test comparison (1 plot)")
        print(f"   • Total: 10 ROC curve visualizations")
        
        print("\n" + "#"*80)
        print(f"# ✅ COMPLETE: {target_name} ({window_type})")
        print("#"*80)

# ============================================================================
# OVERALL COMPARISON
# ============================================================================
print("\n\n" + "="*80)
print("OVERALL COMPARISON: STRESS vs ANXIETY (5MIN vs 10MIN)")
print("="*80)

if all_results:
    print("\n" + "-"*80)
    print("BEST MODELS SUMMARY:")
    print("-"*80)
    
    for window_type in ['5min', '10min']:
        if window_type in all_results:
            print(f"\n{window_type.upper()} Window:")
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    res = all_results[window_type][target_name]
                    print(f"  {target_name}:")
                    print(f"    Best Model:        {res['best_model']}")
                    print(f"    Validation F1:     {res['best_val_f1']:.4f}")
                    print(f"    Validation AUC:    {res['best_val_auc']:.4f}")
                    print(f"    Test F1:           {res['best_test_f1']:.4f}")
                    print(f"    Test Accuracy:     {res['best_test_acc']:.4f}")
                    print(f"    Test AUC:          {res['best_test_auc']:.4f}")
                    print(f"    F1 per class:      Low={res['test_f1_per_class'][0]:.3f}, Med={res['test_f1_per_class'][1]:.3f}, High={res['test_f1_per_class'][2]:.3f}")
                    print(f"    Original features: {res['n_features_original']}")
                    print(f"    PCA components:    {res['n_components']}")
                    print(f"    Dataset Size:      {res['n_samples']} observations")
    
    # Create overall summary CSV
    summary_data = []
    for window_type in ['5min', '10min']:
        if window_type in all_results:
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    res = all_results[window_type][target_name]
                    summary_data.append({
                        'Window': window_type,
                        'Target': target_name,
                        'Best_Model': res['best_model'],
                        'Val_F1': res['best_val_f1'],
                        'Val_AUC': res['best_val_auc'],
                        'Test_F1': res['best_test_f1'],
                        'Test_Acc': res['best_test_acc'],
                        'Test_AUC': res['best_test_auc'],
                        'F1_Low': res['test_f1_per_class'][0],
                        'F1_Medium': res['test_f1_per_class'][1],
                        'F1_High': res['test_f1_per_class'][2],
                        'N_Features_Original': res['n_features_original'],
                        'N_Components': res['n_components'],
                        'N_Samples': res['n_samples']
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'overall_summary_pca_3class_simplified.csv'), index=False)
        print("\n✓ Saved: overall_summary_pca_3class_simplified.csv")

print("\n✅ COMPLETE! All windows and targets processed.")
print("\nKEY CHANGES FROM ORIGINAL:")
print("  • Removed user-based clustering (no PCA on user profiles)")
print("  • Simplified feature engineering with personal baselines")
print("  • Single PCA step at 80% variance (like Code 2)")
print("  • Maintained time series split and all visualizations")