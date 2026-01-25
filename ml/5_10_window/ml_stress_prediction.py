import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STRESS AND ANXIETY PREDICTION WITH PROPER DATA NORMALIZATION")
print("SEPARATE ANALYSIS FOR 30MIN AND 60MIN WINDOWS")
print("TIME-SERIES SPLIT: 70% TRAIN - 15% VALIDATION - 15% TEST")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/5_10_dataset/timeseries/model1"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "shap"), exist_ok=True)

# Define files to process
files = [
    "enriched/ml_ready_5min_window_enriched.csv",
    "enriched/ml_ready_10min_window_enriched.csv"
]

# Store overall results for comparison
all_results = {}

# Process each file separately
for file_idx, file in enumerate(files):
    window_type = "30min" if "30min" in file else "60min"
    window_prefix = "w30" if "30min" in file else "w60"
    
    print("\n" + "="*80)
    print(f"PROCESSING: {window_type} WINDOW")
    print("="*80)
    
    # Create window-specific output directory
    window_output_dir = os.path.join(output_dir, f"{window_type}_window")
    os.makedirs(window_output_dir, exist_ok=True)
    os.makedirs(os.path.join(window_output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(window_output_dir, "shap"), exist_ok=True)
    
    # Load data
    print(f"\n1. LOADING DATA ({window_type})")
    print("-"*80)
    
    data_file = os.path.join(ml_dir, file)
    print(f"Loading: {data_file}")
    
    if not os.path.exists(data_file):
        print(f"✗ File not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} observations")
    
    # Define feature columns for this window
    feature_columns = [
        'is_exam_period',
        'days_until_exam',
        'is_pre_exam_week',
        'is_easter_break',
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
    print(f"\n✓ Using {len(available_features)} features")
    
    # Identify target variables - both stress and anxiety
    target_columns = [col for col in df.columns if col.startswith('q_i_stress') or col.startswith('q_i_anxiety')]
    
    # Identify stress target
    if 'q_i_stress_sliderNeutralPos' in df.columns:
        stress_target = 'q_i_stress_sliderNeutralPos'
    else:
        stress_cols = [col for col in target_columns if 'stress' in col.lower()]
        stress_target = stress_cols[0] if stress_cols else None
    
    # Identify anxiety target
    if 'q_i_anxiety_sliderNeutralPos' in df.columns:
        anxiety_target = 'q_i_anxiety_sliderNeutralPos'
    else:
        anxiety_cols = [col for col in target_columns if 'anxiety' in col.lower()]
        anxiety_target = anxiety_cols[0] if anxiety_cols else None
    
    # Create list of targets to process
    targets_to_process = []
    if stress_target:
        targets_to_process.append(('Stress', stress_target))
        print(f"✓ Stress target: {stress_target}")
    if anxiety_target:
        targets_to_process.append(('Anxiety', anxiety_target))
        print(f"✓ Anxiety target: {anxiety_target}")
    
    if not targets_to_process:
        print("✗ No stress or anxiety targets found in dataset!")
        continue
    
    print(f"\n✓ Will train models for {len(targets_to_process)} target(s): {', '.join([t[0] for t in targets_to_process])}")
    
    # Store results for this window
    all_results[window_type] = {}
    
    # Loop through each target (Stress and/or Anxiety)
    for target_name, primary_target in targets_to_process:
        
        print("\n" + "#"*80)
        print(f"# TARGET: {target_name.upper()} ({window_type} window)")
        print("#"*80)
        
        # Create target-specific output directory
        target_output_dir = os.path.join(window_output_dir, target_name.lower())
        os.makedirs(target_output_dir, exist_ok=True)
        os.makedirs(os.path.join(target_output_dir, "plots"), exist_ok=True)
        
        # 2. DATA PREPROCESSING
        print(f"\n2. DATA PREPROCESSING ({target_name} - {window_type})")
        print("-"*80)
        
        # Sort by timestamp if available (important for time series split)
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
            print("⚠ Warning: No timestamp column found. Assuming data is already in chronological order.")
        
        df_clean = df_sorted[available_features + [primary_target]].dropna(subset=[primary_target])
        
        # Impute missing features with median
        for col in available_features:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                missing_count = df_clean[col].isna().sum()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  Imputed {col}: {missing_count} missing values with median {median_val:.2f}")
        
        print(f"\n✓ Clean dataset: {len(df_clean)} observations")
        
        # 3. EXPLORATORY DATA ANALYSIS
        print(f"\n3. EXPLORATORY DATA ANALYSIS ({target_name} - {window_type})")
        print("-"*80)
        
        X_raw = df_clean[available_features].copy()
        y_continuous = df_clean[primary_target].copy()
        
        # Show target statistics
        print(f"\n{target_name} Target Statistics:")
        print(f"  Mean: {y_continuous.mean():.2f}")
        print(f"  Std:  {y_continuous.std():.2f}")
        print(f"  Min:  {y_continuous.min():.2f}")
        print(f"  Max:  {y_continuous.max():.2f}")
        
        # 4. CREATE TARGET VARIABLE (3-CLASS)
        print(f"\n4. TARGET VARIABLE CREATION ({target_name} - {window_type})")
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
        
        # 5. TIME SERIES SPLIT (80-10-10)
        print(f"\n5. TIME SERIES DATA SPLIT: 70% TRAIN - 15% VAL - 15% TEST ({target_name})")
        print("-"*80)
        print("Using chronological split (no shuffling) to respect temporal order")
        
        # Calculate split indices
        n_samples = len(X_raw)
        train_size = int(0.70 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Split chronologically
        X_train_raw = X_raw.iloc[:train_size]
        y_train = y.iloc[:train_size]
        
        X_val_raw = X_raw.iloc[train_size:train_size+val_size]
        y_val = y.iloc[train_size:train_size+val_size]
        
        X_test_raw = X_raw.iloc[train_size+val_size:]
        y_test = y.iloc[train_size+val_size:]
        
        print(f"Total: {len(X_raw)} observations")
        print(f"\nTraining:   {len(X_train_raw)} ({len(X_train_raw)/len(X_raw)*100:.1f}%)")
        print(f"  Low: {(y_train==0).sum()}, Medium: {(y_train==1).sum()}, High: {(y_train==2).sum()}")
        print(f"Validation: {len(X_val_raw)} ({len(X_val_raw)/len(X_raw)*100:.1f}%)")
        print(f"  Low: {(y_val==0).sum()}, Medium: {(y_val==1).sum()}, High: {(y_val==2).sum()}")
        print(f"Test:       {len(X_test_raw)} ({len(X_test_raw)/len(X_raw)*100:.1f}%)")
        print(f"  Low: {(y_test==0).sum()}, Medium: {(y_test==1).sum()}, High: {(y_test==2).sum()}")
        
        # 6. NORMALIZATION
        print(f"\n6. DATA NORMALIZATION ({target_name} - {window_type})")
        print("-"*80)
        print("Applying StandardScaler (Z-score normalization)")
        
        scaler_standard = StandardScaler()
        X_train_standard = scaler_standard.fit_transform(X_train_raw)
        X_val_standard = scaler_standard.transform(X_val_raw)
        X_test_standard = scaler_standard.transform(X_test_raw)
        
        X_train = pd.DataFrame(X_train_standard, columns=available_features, index=X_train_raw.index)
        X_val = pd.DataFrame(X_val_standard, columns=available_features, index=X_val_raw.index)
        X_test = pd.DataFrame(X_test_standard, columns=available_features, index=X_test_raw.index)
        
        print("✓ Normalization complete")
        
        # 7. TRAIN MODELS
        print(f"\n7. TRAINING MODELS ({target_name} - {window_type})")
        print("="*80)
        print("Note: Using chronological train-val-test split (not cross-validation)")
        print("to preserve temporal order in time series data")
        
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
        
        # 8. GENERATE ROC CURVES
        print(f"\n8. GENERATING ROC CURVES ({target_name} - {window_type})")
        print("="*80)
        
        # Binarize the test labels for ROC curve
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        n_classes = 3
        
        # Define colors for each class
        class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        class_names = ['Low', 'Medium', 'High']
        
        # Create individual ROC curve for each model
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
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'ROC Curves - {model_name}\n{target_name} ({window_type} window)', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_model_name = model_name.replace(' ', '_').lower()
            plt.savefig(os.path.join(target_output_dir, 'plots', f'roc_curve_{safe_model_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved ROC curve for {model_name}")
        
        # Create comparison ROC curve (all models on one plot)
        print(f"\nGenerating comparison ROC curve with all models...")
        
        # For comparison, we'll show one class (e.g., High class - index 2)
        comparison_class = 2  # High class
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different models
        model_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#F38181']
        
        for (model_name, result), color in zip(results.items(), model_colors):
            y_score = result['y_test_proba']
            fpr, tpr, _ = roc_curve(y_test_bin[:, comparison_class], y_score[:, comparison_class])
            roc_auc_val = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2.5,
                   label=f'{model_name} (AUC = {roc_auc_val:.3f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title(f'ROC Curve Comparison - All Models\n{target_name} - {class_names[comparison_class]} Class ({window_type} window)', 
                    fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', f'roc_curve_comparison_all_models.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison ROC curve (all models)")
        
        # 9. COMPARE RESULTS
        print(f"\n9. MODEL COMPARISON ({target_name} - {window_type})")
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
        
        comparison_df.to_csv(os.path.join(target_output_dir, "model_comparison_normalized.csv"), index=False)
        
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model for {target_name}: {best_model_name}")
        print(f"   Validation F1: {comparison_df.iloc[0]['Val F1']:.4f}")
        print(f"   Test F1:       {comparison_df.iloc[0]['Test F1']:.4f}")
        
        # Store results for overall comparison
        all_results[window_type][target_name] = {
            'comparison_df': comparison_df,
            'best_model': best_model_name,
            'best_val_f1': comparison_df.iloc[0]['Val F1'],
            'best_test_f1': comparison_df.iloc[0]['Test F1'],
            'n_samples': len(df_clean)
        }
        
        # 10. SAVE NORMALIZED DATA
        print(f"\n10. SAVING NORMALIZED DATASETS ({target_name} - {window_type})")
        print("-"*80)
        
        normalized_train = X_train.copy()
        normalized_train['target'] = y_train.values
        normalized_train.to_csv(os.path.join(target_output_dir, 'train_normalized.csv'), index=False)
        
        normalized_val = X_val.copy()
        normalized_val['target'] = y_val.values
        normalized_val.to_csv(os.path.join(target_output_dir, 'validation_normalized.csv'), index=False)
        
        normalized_test = X_test.copy()
        normalized_test['target'] = y_test.values
        normalized_test.to_csv(os.path.join(target_output_dir, 'test_normalized.csv'), index=False)
        
        print("✓ Saved: train_normalized.csv")
        print("✓ Saved: validation_normalized.csv")
        print("✓ Saved: test_normalized.csv")
        
        # Save scaler
        import joblib
        joblib.dump(scaler_standard, os.path.join(target_output_dir, 'scaler_standard.pkl'))
        print("✓ Saved: scaler_standard.pkl")
        
        # 11. SUMMARY
        print(f"\n11. SUMMARY ({target_name} - {window_type})")
        print("="*80)
        
        print(f"\n✅ DATA PROPERLY NORMALIZED using StandardScaler")
        print(f"   • Time series split: chronological order preserved")
        print(f"   • Train: {len(X_train)} (70%), Val: {len(X_val)} (15%), Test: {len(X_test)} (15%)")
        
        print(f"\n🎯 Best Model: {best_model_name}")
        print(f"   • Validation F1: {results[best_model_name]['val_f1']:.4f}")
        print(f"   • Test F1:       {results[best_model_name]['test_f1']:.4f}")
        print(f"   • Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
        
        print(f"\n📊 ROC Curves Generated:")
        print(f"   • Individual curves for each model (5 plots)")
        print(f"   • Comparison curve with all models (1 plot)")
        
        print(f"\n📁 Results saved to: {target_output_dir}")
        
        print("\n" + "#"*80)
        print(f"# ✅ COMPLETE for {target_name} ({window_type})!")
        print("#"*80)

# OVERALL COMPARISON
print("\n\n" + "="*80)
print("OVERALL COMPARISON: STRESS vs ANXIETY (30MIN vs 60MIN WINDOWS)")
print("="*80)

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
                    print(f"    Test F1:       {res['best_test_f1']:.4f}")
                    print(f"    Dataset Size:  {res['n_samples']} observations")
    
    # Create overall comparison CSV
    print("\n" + "-"*80)
    print("Saving overall comparison...")
    print("-"*80)
    
    overall_comparison = []
    for window_type in ['30min', '60min']:
        if window_type in all_results:
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    comp_df = all_results[window_type][target_name]['comparison_df'].copy()
                    comp_df['Window'] = window_type
                    comp_df['Target'] = target_name
                    overall_comparison.append(comp_df)
    
    if overall_comparison:
        overall_df = pd.concat(overall_comparison, ignore_index=True)
        overall_df = overall_df[['Window', 'Target', 'Model', 'Val Acc', 'Val F1', 'Test Acc', 'Test F1', 'Test AUC', 'F1 Low', 'F1 Med', 'F1 High']]
        overall_df.to_csv(os.path.join(output_dir, 'overall_comparison_all_targets_models.csv'), index=False)
        print("✓ Saved: overall_comparison_all_targets_models.csv")

print("\n" + "="*80)
print("✅ COMPLETE! All windows and targets processed.")
print("   Time Series Split: 70% Train - 15% Val - 15% Test (chronological)")
print("   ROC Curves: Generated for all models and targets")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print("  - 30min_window/")
print("    - stress/")
print("      - plots/")
print("        - roc_curve_*.png (individual model curves)")
print("        - roc_curve_comparison_all_models.png")
print("    - anxiety/")
print("      - plots/")
print("        - roc_curve_*.png (individual model curves)")
print("        - roc_curve_comparison_all_models.png")
print("  - 60min_window/")
print("    - stress/")
print("      - plots/ (same structure)")
print("    - anxiety/")
print("      - plots/ (same structure)")
print("  - overall_comparison_all_targets_models.csv")
print("\nNote: Data split preserves temporal order for time series analysis")
print("="*80)