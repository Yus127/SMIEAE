import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, roc_auc_score, roc_curve, auc, confusion_matrix,
                            ConfusionMatrixDisplay)
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("ML PIPELINE: STRESS & ANXIETY - BINARY CLASSIFICATION")
print("USER FEATURES + TIME SERIES SPLIT (70-15-15) + NO TEMPORAL LEAKAGE")
print("SEPARATE ANALYSIS FOR 30MIN AND 60MIN WINDOWS")

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/30_60_dataset/timeseries/model2"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Define files to process
files = [
    "enriched/ml_ready_30min_window_enriched.csv",
    "enriched/ml_ready_60min_window_enriched.csv"
]

# Store overall results
all_results = {}

# Process each window
for file_idx, file in enumerate(files):
    window_type = "30min" if "30min" in file else "60min"
    window_prefix = "w30" if "30min" in file else "w60"
    
    print("\n" + "="*80)
    print(f"PROCESSING: {window_type.upper()} WINDOW")
    
    # Create window-specific output directory
    window_output_dir = os.path.join(output_dir, f"{window_type}_window")
    os.makedirs(window_output_dir, exist_ok=True)
    os.makedirs(os.path.join(window_output_dir, "plots"), exist_ok=True)
    
    # 1. LOAD DATA
    print(f"\n1. LOADING DATA ({window_type})")
    print("-"*80)
    
    data_file = os.path.join(ml_dir, file)
    if not os.path.exists(data_file):
        print(f" File not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    print(f" Loaded {len(df)} observations")
    
    # Define features
    feature_columns = [
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
    feature_columns.extend(hr_features)
    
    available_features = [col for col in feature_columns if col in df.columns]
    
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
        print(f" Stress target: {stress_target}")
    if anxiety_target:
        targets_to_process.append(('Anxiety', anxiety_target))
        print(f" Anxiety target: {anxiety_target}")
    
    if not targets_to_process:
        print(" No stress or anxiety targets found!")
        continue
    
    print(f"\n Will train models for {len(targets_to_process)} target(s): {', '.join([t[0] for t in targets_to_process])}")
    
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
        
        # 2. SORT BY TIME AND PREPARE DATA
        print(f"\n2. DATA PREPARATION ({target_name} - {window_type})")
        print("-"*80)
        
        # Sort by timestamp (critical for time series split)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            print(" Data sorted by timestamp (chronological order)")
        elif 'date' in df.columns:
            df_sorted = df.sort_values('date').reset_index(drop=True)
            print(" Data sorted by date (chronological order)")
        elif 'datetime' in df.columns:
            df_sorted = df.sort_values('datetime').reset_index(drop=True)
            print(" Data sorted by datetime (chronological order)")
        else:
            df_sorted = df.copy()
            print(" Warning: No timestamp column found. Assuming chronological order.")
        
        df_clean = df_sorted[available_features + [primary_target, 'userid']].copy()
        df_clean = df_clean.dropna(subset=[primary_target])
        
        # Impute missing features
        for col in available_features:
            if df_clean[col].isna().sum() > 0:
                median_val = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(median_val)
                print(f"  Imputed {col}: median={median_val:.2f}")
        
        print(f"\n Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")
        
        # 3. USER-BASED FEATURES (NO LEAKAGE)
        print(f"\n3. CREATING USER-BASED FEATURES ({target_name})")
        print("-"*80)
        print("Note: User profiles calculated WITHOUT using target variable")
        
        # Calculate user profiles ONLY from physiological/contextual features
        user_agg_dict = {
            'daily_total_steps': 'mean',
            'is_exam_period': 'mean'
        }
        
        # Add HR features that exist
        for feat in hr_features:
            if feat in df_clean.columns:
                user_agg_dict[feat] = 'mean'
        
        user_profiles = df_clean.groupby('userid').agg(user_agg_dict).reset_index()
        
        # Rename columns
        col_mapping = {'userid': 'userid'}
        if 'daily_total_steps' in user_agg_dict:
            col_mapping['daily_total_steps'] = 'avg_steps'
        if 'is_exam_period' in user_agg_dict:
            col_mapping['is_exam_period'] = 'exam_exposure'
        
        # Find HR mean feature
        hr_mean_feat = None
        hr_std_feat = None
        for feat in hr_features:
            if 'mean' in feat and feat in user_profiles.columns:
                hr_mean_feat = feat
                col_mapping[feat] = 'avg_hr'
            if 'std' in feat and feat in user_profiles.columns:
                hr_std_feat = feat
                col_mapping[feat] = 'avg_hr_std'
        
        user_profiles = user_profiles.rename(columns=col_mapping)
        
        # Cluster users based ONLY on physiological/contextual features (NOT stress/anxiety!)
        user_feature_cols = [c for c in ['avg_hr', 'avg_hr_std', 'avg_steps', 'exam_exposure'] 
                            if c in user_profiles.columns]
        
        if len(user_feature_cols) > 0:
            X_users = user_profiles[user_feature_cols].fillna(0)
            scaler_user = StandardScaler()
            X_users_scaled = scaler_user.fit_transform(X_users)
            
            kmeans_users = KMeans(n_clusters=3, random_state=42, n_init=20)
            user_clusters = kmeans_users.fit_predict(X_users_scaled)
            user_profiles['user_cluster'] = user_clusters
            
            print(f" Created 3 user clusters (based on physiology, NOT {target_name.lower()})")
        else:
            user_profiles['user_cluster'] = 0
            print(" Limited features for clustering")
        
        # Add to data
        df_clean = df_clean.merge(user_profiles[['userid', 'user_cluster']], 
                                 on='userid', how='left')
        
        # User baseline deviations (NO TARGET LEAKAGE)
        if hr_mean_feat:
            user_hr_baseline = df_clean.groupby('userid')[hr_mean_feat].transform('mean')
            df_clean['hr_deviation'] = df_clean[hr_mean_feat] - user_hr_baseline
        else:
            df_clean['hr_deviation'] = 0
        
        user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
        df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline.replace(0, 1) + 1)
        
        print(f" Added user features (NO temporal leakage)")
        
        # 4. INTERACTION FEATURES
        print(f"\n4. CREATING INTERACTION FEATURES ({target_name})")
        print("-"*80)
        
        df_clean['cluster_x_exam'] = df_clean['user_cluster'] * df_clean['is_exam_period']
        df_clean['hr_dev_x_exam'] = df_clean['hr_deviation'] * df_clean['is_exam_period']
        df_clean['exam_proximity'] = 1 / (df_clean['days_until_exam'].clip(lower=1) + 1)
        df_clean['steps_x_exam_prox'] = df_clean['steps_ratio'] * df_clean['exam_proximity']
        
        if hr_std_feat:
            df_clean['hr_var_x_exam'] = df_clean[hr_std_feat] * df_clean['is_exam_period']
        else:
            df_clean['hr_var_x_exam'] = 0
        
        print(f" Added 5 interaction features")
        
        # 5. FINAL FEATURE SET
        print(f"\n5. PREPARING FEATURE SET ({target_name})")
        print("-"*80)
        
        enhanced_features = available_features + [
            'user_cluster', 'hr_deviation', 'steps_ratio'
        ]
        
        interaction_features = [
            'cluster_x_exam', 'hr_dev_x_exam', 'exam_proximity', 
            'steps_x_exam_prox', 'hr_var_x_exam'
        ]
        
        full_features = enhanced_features + interaction_features
        
        print(f"  Base features: {len(available_features)}")
        print(f"  User features: {len(enhanced_features) - len(available_features)}")
        print(f"  Interactions:  {len(interaction_features)}")
        print(f"  Total:         {len(full_features)}")
        
        df_model = df_clean[full_features + [primary_target]].copy()
        df_model = df_model.dropna()
        
        # Clean extreme values
        for col in full_features:
            if df_model[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                upper = df_model[col].quantile(0.999)
                lower = df_model[col].quantile(0.001)
                df_model[col] = df_model[col].clip(lower=lower, upper=upper)
        
        print(f"\n Final dataset: {len(df_model)} observations")
        
        # 6. BINARY CLASSIFICATION SETUP
        print(f"\n6. BINARY CLASSIFICATION SETUP ({target_name})")
        print("-"*80)
        
        y_continuous = df_model[primary_target]
        p33 = np.percentile(y_continuous, 33.33)
        p67 = np.percentile(y_continuous, 66.67)
        
        # Keep only low and high (remove middle tercile)
        mask_binary = (y_continuous <= p33) | (y_continuous >= p67)
        df_binary = df_model[mask_binary].copy()
        
        y_binary = (df_binary[primary_target] > p67).astype(int)
        X_binary = df_binary[full_features]
        
        print(f"  Low {target_name.lower()} (0):  {(y_binary == 0).sum()} (≤{p33:.1f})")
        print(f"  High {target_name.lower()} (1): {(y_binary == 1).sum()} (≥{p67:.1f})")
        print(f"  Total binary:    {len(y_binary)}")
        
        # 7. TIME SERIES SPLIT (70-15-15)
        print(f"\n7. TIME SERIES SPLIT: 70% TRAIN - 15% VAL - 15% TEST ({target_name})")
        print("-"*80)
        print("Using chronological split (no shuffling) to respect temporal order")
        
        # Calculate split indices
        n_samples = len(X_binary)
        train_size = int(0.70 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Split chronologically
        X_train = X_binary.iloc[:train_size]
        y_train = y_binary.iloc[:train_size]
        
        X_val = X_binary.iloc[train_size:train_size+val_size]
        y_val = y_binary.iloc[train_size:train_size+val_size]
        
        X_test = X_binary.iloc[train_size+val_size:]
        y_test = y_binary.iloc[train_size+val_size:]
        
        print(f"  Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_train==0).sum()}, High: {(y_train==1).sum()}")
        print(f"  Val:   {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_val==0).sum()}, High: {(y_val==1).sum()}")
        print(f"  Test:  {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")
        print(f"    Low: {(y_test==0).sum()}, High: {(y_test==1).sum()}")
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        print(" Data normalized (fit on train only)")
        
        # 8. TRAIN MODELS
        print(f"\n8. TRAINING MODELS - BINARY {target_name.upper()} ({window_type})")
        
        models = {
            'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                          random_state=42, objective='binary:logistic'),
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, 
                                                   random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name}...")
            model.fit(X_train_scaled, y_train)
            
            # Validation predictions
            y_val_pred = model.predict(X_val_scaled)
            y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_auc = roc_auc_score(y_val, y_val_proba)
            
            # Test predictions
            y_test_pred = model.predict(X_test_scaled)
            y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_auc = roc_auc_score(y_test, y_test_proba)
            
            print(f"  Val:  F1={val_f1:.4f}, Acc={val_acc:.4f}, AUC={val_auc:.4f}")
            print(f"  Test: F1={test_f1:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")
            print(f"        Precision={test_precision:.4f}, Recall={test_recall:.4f}")
            
            results[name] = {
                'val_f1': val_f1, 'val_acc': val_acc, 'val_auc': val_auc,
                'test_f1': test_f1, 'test_acc': test_acc,
                'test_precision': test_precision, 'test_recall': test_recall, 
                'test_auc': test_auc,
                'y_val_pred': y_val_pred, 'y_val_proba': y_val_proba,
                'y_test_pred': y_test_pred, 'y_test_proba': y_test_proba
            }
        
        # 9. ENSEMBLE MODEL
        print(f"\n9. ENSEMBLE MODEL ({target_name})")
        print("-"*80)
        
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', models['XGBoost']),
                ('rf', models['Random Forest']),
                ('lr', models['Logistic Regression'])
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        
        y_val_pred_ens = ensemble.predict(X_val_scaled)
        y_val_proba_ens = ensemble.predict_proba(X_val_scaled)[:, 1]
        
        y_test_pred_ens = ensemble.predict(X_test_scaled)
        y_test_proba_ens = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        val_f1_ens = f1_score(y_val, y_val_pred_ens, zero_division=0)
        val_acc_ens = accuracy_score(y_val, y_val_pred_ens)
        val_auc_ens = roc_auc_score(y_val, y_val_proba_ens)
        
        test_f1_ens = f1_score(y_test, y_test_pred_ens, zero_division=0)
        test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
        test_auc_ens = roc_auc_score(y_test, y_test_proba_ens)
        test_precision_ens = precision_score(y_test, y_test_pred_ens, zero_division=0)
        test_recall_ens = recall_score(y_test, y_test_pred_ens, zero_division=0)
        
        print(f"  Val:  F1={val_f1_ens:.4f}, Acc={val_acc_ens:.4f}, AUC={val_auc_ens:.4f}")
        print(f"  Test: F1={test_f1_ens:.4f}, Acc={test_acc_ens:.4f}, AUC={test_auc_ens:.4f}")
        
        results['Ensemble'] = {
            'val_f1': val_f1_ens, 'val_acc': val_acc_ens, 'val_auc': val_auc_ens,
            'test_f1': test_f1_ens, 'test_acc': test_acc_ens, 
            'test_precision': test_precision_ens, 'test_recall': test_recall_ens,
            'test_auc': test_auc_ens,
            'y_val_pred': y_val_pred_ens, 'y_val_proba': y_val_proba_ens,
            'y_test_pred': y_test_pred_ens, 'y_test_proba': y_test_proba_ens
        }
        
        # 10. GENERATE ROC CURVES
        print(f"\n10. GENERATING ROC CURVES ({target_name} - {window_type})")
        
        # Define colors for different models
        model_colors = {
            'XGBoost': '#FF6B6B',
            'Random Forest': '#4ECDC4',
            'Logistic Regression': '#45B7D1',
            'SVM': '#95E1D3',
            'Ensemble': '#F38181'
        }
        
        # Create individual ROC curve for each model
        for model_name, result in results.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
            roc_auc_val = auc(fpr, tpr)
            
            # Plot ROC curve
            color = model_colors.get(model_name, '#333333')
            ax.plot(fpr, tpr, color=color, lw=3,
                   label=f'{model_name} (AUC = {roc_auc_val:.3f})')
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=13)
            ax.set_ylabel('True Positive Rate', fontsize=13)
            ax.set_title(f'ROC Curve - {model_name}\nBinary {target_name} ({window_type} window)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_model_name = model_name.replace(' ', '_').lower()
            plt.savefig(os.path.join(target_output_dir, 'plots', f'roc_curve_{safe_model_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f" Saved ROC curve for {model_name}")
        
        # Create comparison ROC curve (all models on one plot)
        print(f"\nGenerating comparison ROC curve with all models...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_test_proba'])
            roc_auc_val = auc(fpr, tpr)
            
            color = model_colors.get(model_name, '#333333')
            ax.plot(fpr, tpr, color=color, lw=2.5,
                   label=f'{model_name} (AUC = {roc_auc_val:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'ROC Curve Comparison - All Models\nBinary {target_name} ({window_type} window)', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'roc_curve_comparison_all_models.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved comparison ROC curve (all models)")
        
        # 11. VALIDATION vs TEST ROC COMPARISON
        print(f"\nGenerating validation vs test ROC comparison...")
        
        # Create a comparison plot for the best model showing validation and test performance
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best_result = results[best_model_name]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Validation ROC
        fpr_val, tpr_val, _ = roc_curve(y_val, best_result['y_val_proba'])
        auc_val = auc(fpr_val, tpr_val)
        ax.plot(fpr_val, tpr_val, color='#4ECDC4', lw=2.5,
               label=f'Validation (AUC = {auc_val:.3f})', linestyle='--')
        
        # Test ROC
        fpr_test, tpr_test, _ = roc_curve(y_test, best_result['y_test_proba'])
        auc_test = auc(fpr_test, tpr_test)
        ax.plot(fpr_test, tpr_test, color='#FF6B6B', lw=2.5,
               label=f'Test (AUC = {auc_test:.3f})')
        
        # Diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'ROC Curve: Validation vs Test - {best_model_name}\nBinary {target_name} ({window_type} window)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'roc_curve_validation_vs_test.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Saved validation vs test ROC curve")
        
        # 12. RESULTS & FINAL VISUALIZATIONS
        print(f"\n12. FINAL RESULTS ({target_name} - {window_type})")
        
        best_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
        best = results[best_name]
        
        print(f"\n BEST MODEL: {best_name}")
        print(f"   Validation F1:  {results[best_name]['val_f1']:.4f}")
        print(f"   Validation AUC: {results[best_name]['val_auc']:.4f}")
        print(f"   Test F1:        {best['test_f1']:.4f}")
        print(f"   Test Accuracy:  {best['test_acc']:.4f}")
        print(f"   Test Precision: {best['test_precision']:.4f}")
        print(f"   Test Recall:    {best['test_recall']:.4f}")
        print(f"   Test AUC:       {best['test_auc']:.4f}")
        
        # Check if results are realistic
        if best['test_f1'] > 0.90:
            print("\n  WARNING: F1 > 0.90 is suspiciously high!")
            print("   Possible issues:")
            print("   • Small dataset with consistent patterns")
            print("   • Binary split too extreme (very separable classes)")
            print("   • Consider user-stratified CV to validate properly")
        
        # Store for overall comparison
        all_results[window_type][target_name] = {
            'best_model': best_name,
            'best_val_f1': results[best_name]['val_f1'],
            'best_val_auc': results[best_name]['val_auc'],
            'best_test_f1': best['test_f1'],
            'best_test_acc': best['test_acc'],
            'best_test_auc': best['test_auc'],
            'n_samples': len(df_binary)
        }
        
        # Save results CSV
        comparison_df = pd.DataFrame([
            {'Model': name, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
            for name, res in results.items()
        ])
        comparison_df.to_csv(os.path.join(target_output_dir, 'results_no_leakage.csv'), index=False)
        print(f"\n Saved: results_no_leakage.csv")
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, best['y_test_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'Confusion Matrix - {best_name}\n{target_name} ({window_type})', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'confusion_matrix.png'), dpi=300)
        print(f" Saved: confusion_matrix.png")
        plt.close()
        
        # 13. SUMMARY
        print(f"\n13. SUMMARY ({target_name} - {window_type})")
        
        print(f"\n ROC Curves Generated:")
        print(f"   • Individual curves for each model (5 plots)")
        print(f"   • Comparison curve with all models (1 plot)")
        print(f"   • Validation vs Test comparison for best model (1 plot)")
        print(f"   • Total: 3 ROC curve visualizations")
        
        print("\n" + "#"*80)
        print(f"#  COMPLETE: {target_name} ({window_type})")
        print("#"*80)

# OVERALL COMPARISON
print("\n\n" + "="*80)
print("OVERALL COMPARISON: STRESS vs ANXIETY (30MIN vs 60MIN)")

if all_results:
    print("\n" + "-"*80)
    print("BEST MODELS SUMMARY:")
    print("-"*80)
    
    for window_type in ['30min', '60min']:
        if window_type in all_results:
            print(f"\n{window_type.upper()} Window:")
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    res = all_results[window_type][target_name]
                    print(f"  {target_name}:")
                    print(f"    Best Model:    {res['best_model']}")
                    print(f"    Validation F1: {res['best_val_f1']:.4f}")
                    print(f"    Validation AUC: {res['best_val_auc']:.4f}")
                    print(f"    Test F1:       {res['best_test_f1']:.4f}")
                    print(f"    Test Accuracy: {res['best_test_acc']:.4f}")
                    print(f"    Test AUC:      {res['best_test_auc']:.4f}")
                    print(f"    Dataset Size:  {res['n_samples']} observations")
    
    # Create overall summary CSV
    summary_data = []
    for window_type in ['30min', '60min']:
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
                        'N_Samples': res['n_samples']
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'overall_summary.csv'), index=False)
        print("\n Saved: overall_summary.csv")

print("\n" + "="*80)
print("   Time Series Split: 70% Train - 15% Val - 15% Test (chronological)")
print("   Binary Classification: Low vs High (middle tercile removed)")
print("   User Features: NO temporal leakage")
print("   ROC Curves: Comprehensive visualizations generated")
print(f"\nResults saved to: {output_dir}")
print("  - 30min_window/")
print("    - stress/")
print("      - plots/")
print("        - roc_curve_*.png (individual model curves)")
print("        - roc_curve_comparison_all_models.png")
print("        - roc_curve_validation_vs_test.png")
print("        - confusion_matrix.png")
print("    - anxiety/")
print("      - plots/ (same structure)")
print("  - 60min_window/")
print("    - stress/")
print("      - plots/ (same structure)")
print("    - anxiety/")
print("      - plots/ (same structure)")
print("  - overall_summary.csv")
