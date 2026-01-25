import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED LSTM MODEL FOR STRESS & ANXIETY PREDICTION")
print("TIME SERIES SPLIT (70-15-15) - SEPARATE 5MIN AND 10MIN WINDOWS")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/5_10_dataset/timeseries/model4"
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
    
    # Create timestamp if needed
    if 'response_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['response_timestamp'])
    elif 'timestamp' not in df.columns:
        df['timestamp'] = df.groupby('userid').cumcount()
    
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
        # 2. PREPARE DATA
        # ====================================================================
        print(f"\n2. DATA PREPARATION ({target_name} - {window_type})")
        print("-"*80)
        
        # Sort by timestamp (critical for time series)
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values(['userid', 'timestamp']).reset_index(drop=True)
            print("✓ Data sorted by user and timestamp (chronological order)")
        else:
            df_sorted = df.sort_values('userid').reset_index(drop=True)
            print("⚠ Warning: No timestamp column, sorted by userid only")
        
        df_clean = df_sorted[available_features + [primary_target, 'userid', 'timestamp']].copy()
        df_clean = df_clean.dropna(subset=[primary_target])
        
        # Impute missing features
        for col in available_features:
            if df_clean[col].isna().sum() > 0:
                median_val = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(median_val)
        
        print(f"✓ Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")
        
        # ====================================================================
        # 3. USER-SPECIFIC FEATURES
        # ====================================================================
        print(f"\n3. CREATING USER-SPECIFIC FEATURES ({target_name})")
        print("-"*80)
        
        # Find HR mean feature
        hr_mean_feat = None
        for feat in hr_features:
            if 'mean' in feat and feat in df_clean.columns:
                hr_mean_feat = feat
                break
        
        if hr_mean_feat:
            user_hr_baseline = df_clean.groupby('userid')[hr_mean_feat].transform('mean')
            df_clean['hr_deviation'] = df_clean[hr_mean_feat] - user_hr_baseline
        else:
            df_clean['hr_deviation'] = 0
        
        user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
        df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline.replace(0, 1) + 1)
        
        enhanced_features = available_features + ['hr_deviation', 'steps_ratio']
        
        print(f"✓ Enhanced features: {len(enhanced_features)}")
        
        # ====================================================================
        # 4. BINARY CLASSIFICATION (EASIER FOR LSTM)
        # ====================================================================
        print(f"\n4. CREATING BINARY TARGET (LOW vs HIGH {target_name})")
        print("-"*80)
        
        y_continuous = df_clean[primary_target]
        p33 = np.percentile(y_continuous, 33.33)
        p67 = np.percentile(y_continuous, 66.67)
        
        # Binary: Remove medium, keep only Low and High
        mask_binary = (y_continuous <= p33) | (y_continuous >= p67)
        df_binary = df_clean[mask_binary].copy()
        
        df_binary['target_class'] = (df_binary[primary_target] > p67).astype(int)
        
        print(f"  Low (0):  {(df_binary['target_class'] == 0).sum()} observations (≤{p33:.1f})")
        print(f"  High (1): {(df_binary['target_class'] == 1).sum()} observations (≥{p67:.1f})")
        print(f"  Removed:  {(~mask_binary).sum()} medium observations")
        
        # ====================================================================
        # 5. CREATE SEQUENCES
        # ====================================================================
        print(f"\n5. CREATING SEQUENCES ({target_name})")
        print("-"*80)
        
        seq_length = 3  # Shorter sequences
        
        print(f"  Sequence length: {seq_length} timesteps")
        
        def create_sequences_all_users(df, user_col, feature_cols, target_col, seq_length):
            """Create sequences from ALL users together"""
            X_sequences = []
            y_sequences = []
            user_ids = []
            timestamps = []
            
            for user in df[user_col].unique():
                user_data = df[df[user_col] == user].copy()
                
                if len(user_data) < seq_length + 1:
                    continue
                
                features = user_data[feature_cols].values
                targets = user_data[target_col].values
                user_timestamps = user_data['timestamp'].values
                
                for i in range(len(user_data) - seq_length):
                    X_sequences.append(features[i:i+seq_length])
                    y_sequences.append(targets[i+seq_length])
                    user_ids.append(user)
                    timestamps.append(user_timestamps[i+seq_length])
            
            return np.array(X_sequences), np.array(y_sequences), np.array(user_ids), np.array(timestamps)
        
        X_sequences, y_sequences, seq_users, seq_timestamps = create_sequences_all_users(
            df_binary, 'userid', enhanced_features, 'target_class', seq_length
        )
        
        print(f"✓ Created {len(X_sequences)} sequences")
        print(f"  Shape: {X_sequences.shape}")
        
        # ====================================================================
        # 6. TIME SERIES SPLIT (70-15-15)
        # ====================================================================
        print(f"\n6. TIME SERIES SPLIT: 70% TRAIN - 15% VAL - 15% TEST ({target_name})")
        print("-"*80)
        print("Using chronological split (no shuffling) to respect temporal order")
        
        # Sort by timestamp to ensure chronological order
        sort_idx = np.argsort(seq_timestamps)
        X_sequences = X_sequences[sort_idx]
        y_sequences = y_sequences[sort_idx]
        
        # Calculate split indices
        n_samples = len(X_sequences)
        train_size = int(0.70 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Split chronologically
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        
        X_val = X_sequences[train_size:train_size+val_size]
        y_val = y_sequences[train_size:train_size+val_size]
        
        X_test = X_sequences[train_size+val_size:]
        y_test = y_sequences[train_size+val_size:]
        
        print(f"  Train: {len(X_train)} sequences ({len(X_train)/n_samples*100:.1f}%)")
        print(f"    Low: {np.sum(y_train==0)}, High: {np.sum(y_train==1)}")
        print(f"  Val:   {len(X_val)} sequences ({len(X_val)/n_samples*100:.1f}%)")
        print(f"    Low: {np.sum(y_val==0)}, High: {np.sum(y_val==1)}")
        print(f"  Test:  {len(X_test)} sequences ({len(X_test)/n_samples*100:.1f}%)")
        print(f"    Low: {np.sum(y_test==0)}, High: {np.sum(y_test==1)}")
        
        # ====================================================================
        # 7. NORMALIZE
        # ====================================================================
        print(f"\n7. NORMALIZING DATA ({target_name})")
        print("-"*80)
        
        n_samples_train, n_timesteps, n_features = X_train.shape
        
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        print(f"✓ Data normalized (fit on train only)")
        
        # Convert to one-hot for binary
        y_train_cat = to_categorical(y_train, num_classes=2)
        y_val_cat = to_categorical(y_val, num_classes=2)
        y_test_cat = to_categorical(y_test, num_classes=2)
        
        # ====================================================================
        # 8. BUILD LSTM MODEL
        # ====================================================================
        print(f"\n8. BUILDING LSTM MODEL ({target_name})")
        print("-"*80)
        
        # Compute class weights for imbalanced data
        class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        
        print(f"  Class weights: {class_weights}")
        
        model = Sequential([
            # Bidirectional LSTM (captures patterns in both directions)
            Bidirectional(LSTM(32, return_sequences=True, 
                               input_shape=(seq_length, n_features))),
            Dropout(0.4),
            
            # Second LSTM layer
            Bidirectional(LSTM(16, return_sequences=False)),
            Dropout(0.4),
            
            # Dense layers
            Dense(8, activation='relu'),
            Dropout(0.3),
            
            # Output (binary)
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✓ LSTM Model Architecture:")
        model.summary()
        
        # ====================================================================
        # 9. TRAIN WITH CLASS WEIGHTS
        # ====================================================================
        print(f"\n9. TRAINING LSTM MODEL ({target_name})")
        print("-"*80)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-6,
            verbose=1
        )
        
        print("\nTraining with class weights to handle imbalance...")
        history = model.fit(
            X_train_scaled, y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=150,
            batch_size=16,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\n✓ Training complete!")
        
        # ====================================================================
        # 10. EVALUATE
        # ====================================================================
        print(f"\n10. EVALUATING MODEL ({target_name})")
        print("-"*80)
        
        # Get probability predictions for ROC curve
        y_train_proba = model.predict(X_train_scaled, verbose=0)
        y_val_proba = model.predict(X_val_scaled, verbose=0)
        y_test_proba = model.predict(X_test_scaled, verbose=0)
        
        # Get class predictions
        y_train_pred = np.argmax(y_train_proba, axis=1)
        y_val_pred = np.argmax(y_val_proba, axis=1)
        y_test_pred = np.argmax(y_test_proba, axis=1)
        
        train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nPerformance Summary:")
        print(f"  Train: F1={train_f1:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   F1={val_f1:.4f}, Acc={val_acc:.4f}")
        print(f"  Test:  F1={test_f1:.4f}, Acc={test_acc:.4f}")
        
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION REPORT (TEST SET) - {target_name}")
        print(f"{'='*80}")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=[f'Low {target_name}', f'High {target_name}'],
                                   zero_division=0))
        
        # Store for overall comparison
        all_results[window_type][target_name] = {
            'model_type': 'LSTM_Binary',
            'sequence_length': seq_length,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'epochs_trained': len(history.history['loss']),
            'total_params': model.count_params(),
            'n_sequences': n_samples
        }
        
        # ====================================================================
        # 11. VISUALIZATIONS
        # ====================================================================
        print(f"\n11. CREATING VISUALIZATIONS ({target_name})")
        print("-"*80)
        
        # Training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].set_title(f'Model Loss - Binary LSTM\n{target_name} ({window_type})', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title(f'Model Accuracy - Binary LSTM\n{target_name} ({window_type})', fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'training_history.png'), dpi=300)
        print("✓ Saved: training_history.png")
        plt.close()
        
        # Confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title(f'LSTM Confusion Matrix - Binary Classification\n{target_name} ({window_type})', 
                    fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'confusion_matrix.png'), dpi=300)
        print("✓ Saved: confusion_matrix.png")
        plt.close()
        
        # ====================================================================
        # NEW: ROC CURVE
        # ====================================================================
        print("\n  Creating ROC Curve...")
        
        # Calculate ROC curve for each dataset
        # For binary classification, we use the probability of class 1 (High stress/anxiety)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba[:, 1])
        roc_auc_train = auc(fpr_train, tpr_train)
        
        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_proba[:, 1])
        roc_auc_val = auc(fpr_val, tpr_val)
        
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba[:, 1])
        roc_auc_test = auc(fpr_test, tpr_test)
        
        # Plot ROC curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier', alpha=0.5)
        
        # Plot ROC curves for each dataset
        ax.plot(fpr_train, tpr_train, linewidth=2.5, 
                label=f'Train (AUC = {roc_auc_train:.3f})', color='#2E86AB')
        ax.plot(fpr_val, tpr_val, linewidth=2.5, 
                label=f'Validation (AUC = {roc_auc_val:.3f})', color='#A23B72')
        ax.plot(fpr_test, tpr_test, linewidth=2.5, 
                label=f'Test (AUC = {roc_auc_test:.3f})', color='#F18F01')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curve - Binary LSTM\n{target_name} ({window_type})', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        plt.savefig(os.path.join(target_output_dir, 'plots', 'roc_curve.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: roc_curve.png")
        plt.close()
        
        # Print AUC scores
        print(f"\n  ROC AUC Scores:")
        print(f"    Train:      {roc_auc_train:.4f}")
        print(f"    Validation: {roc_auc_val:.4f}")
        print(f"    Test:       {roc_auc_test:.4f}")
        
        # ====================================================================
        # 12. SAVE
        # ====================================================================
        print(f"\n12. SAVING RESULTS ({target_name})")
        print("-"*80)
        
        model.save(os.path.join(target_output_dir, 'lstm_binary_model.keras'))
        print("✓ Saved: lstm_binary_model.keras")
        
        import joblib
        joblib.dump(scaler, os.path.join(target_output_dir, 'scaler_lstm.pkl'))
        print("✓ Saved: scaler_lstm.pkl")
        
        results = {
            'Model': 'LSTM_Binary',
            'Target': target_name,
            'Window': window_type,
            'Sequence_Length': seq_length,
            'Train_F1': train_f1,
            'Val_F1': val_f1,
            'Test_F1': test_f1,
            'Test_Acc': test_acc,
            'Train_AUC': roc_auc_train,
            'Val_AUC': roc_auc_val,
            'Test_AUC': roc_auc_test,
            'Epochs': len(history.history['loss']),
            'Parameters': model.count_params(),
            'N_Sequences': n_samples
        }
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(os.path.join(target_output_dir, 'lstm_results.csv'), index=False)
        print("✓ Saved: lstm_results.csv")
        
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
    print("LSTM MODEL SUMMARY:")
    print("-"*80)
    
    for window_type in ['5min', '10min']:
        if window_type in all_results:
            print(f"\n{window_type.upper()} Window:")
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    res = all_results[window_type][target_name]
                    print(f"  {target_name}:")
                    print(f"    Validation F1:  {res['val_f1']:.4f}")
                    print(f"    Test F1:        {res['test_f1']:.4f}")
                    print(f"    Test Accuracy:  {res['test_acc']:.4f}")
                    print(f"    Epochs Trained: {res['epochs_trained']}")
                    print(f"    Sequences:      {res['n_sequences']}")
    
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
                        'Model': res['model_type'],
                        'Seq_Length': res['sequence_length'],
                        'Train_F1': res['train_f1'],
                        'Val_F1': res['val_f1'],
                        'Test_F1': res['test_f1'],
                        'Test_Acc': res['test_acc'],
                        'Epochs': res['epochs_trained'],
                        'Params': res['total_params'],
                        'N_Sequences': res['n_sequences']
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'overall_lstm_summary.csv'), index=False)
        print("\n✓ Saved: overall_lstm_summary.csv")

print("\n" + "="*80)
print("✅ COMPLETE! All windows and targets processed.")
print("   Time Series Split: 70% Train - 15% Val - 15% Test (chronological)")
print("   Architecture: Bidirectional LSTM (32→16 units)")
print("   Binary Classification: Low vs High")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print("  - 5min_window/")
print("    - stress/")
print("      - lstm_binary_model.keras")
print("      - scaler_lstm.pkl")
print("      - plots/")
print("        - training_history.png")
print("        - confusion_matrix.png")
print("        - roc_curve.png  ← NEW!")
print("    - anxiety/")
print("  - 10min_window/")
print("    - stress/")
print("    - anxiety/")
print("  - overall_lstm_summary.csv")
print("\n💡 Key Features:")
print("   • Bidirectional LSTM (learns temporal patterns both ways)")
print("   • Class weights (handles imbalanced data)")
print("   • Time series split (respects chronological order)")
print("   • User-specific features (HR deviation, step ratio)")
print("   • Binary classification (clearer Low/High distinction)")
print("   • ROC Curve with AUC scores (model discrimination ability)")
print("="*80)