import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, auc)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LSTM MODEL WITH SLIDING WINDOWS (3 TIMESTEPS) - BINARY CLASSIFICATION")
print("TIME SERIES SPLIT (70-15-15)")
print("Proper temporal sequence modeling for stress/anxiety prediction")
print("Low vs High (2 classes)")
print("="*80)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/timeseries/model4'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. LOADING DATA")
print("-"*80)

df = pd.read_csv(INPUT_PATH)
print(f"✓ Loaded {len(df)} observations, {len(df.columns)} columns")

# Features - use the same features from previous models
feature_columns = [
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
    'sleep_stage_transitions',
    'daily_hrv_summary_entropy',
    'heart_rate_activity_beats per minute_mean',
    'is_exam_period',
    'is_semana_santa',
    'days_to_next_exam',
    'days_since_last_exam',
    'weeks_to_next_exam',
    'weeks_since_last_exam'
]

# Check which features are available
available_features = [col for col in feature_columns if col in df.columns]
print(f"✓ Using {len(available_features)} available features")

# Ensure userid exists
if 'userid' not in df.columns:
    if 'user_id' in df.columns:
        df['userid'] = df['user_id']
    else:
        print("  ⚠ No userid column found, creating sequential user IDs")
        df['userid'] = 0

# Create timestamp if not exists (for proper ordering)
if 'timestamp' not in df.columns:
    if 'created_at' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_at'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    else:
        print("  ⚠ No timestamp column found, using sequential order per user")
        df['timestamp'] = df.groupby('userid').cumcount()

print(f"✓ Dataset has {df['userid'].nunique()} unique users")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n2. FEATURE ENGINEERING")
print("-"*80)

# Calculate personal baselines for each user (using only past data to avoid leakage)
print("Calculating user-specific features...")

# Sort by user and time first
df = df.sort_values(['userid', 'timestamp']).reset_index(drop=True)

# For each user, calculate expanding mean (only using past data)
df['hr_user_mean'] = df.groupby('userid')['heart_rate_activity_beats per minute_mean'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.mean())
)
df['steps_user_mean'] = df.groupby('userid')['daily_total_steps'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.mean())
)
df['hrv_user_mean'] = df.groupby('userid')['daily_hrv_summary_rmssd'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.mean())
)
df['sleep_user_mean'] = df.groupby('userid')['sleep_global_duration'].transform(
    lambda x: x.expanding().mean().shift(1).fillna(x.mean())
)

# Create deviation features
df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_user_mean']) / (df['hr_user_mean'] + 1e-6)
df['activity_ratio'] = (df['daily_total_steps']) / (df['steps_user_mean'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_user_mean']) / (df['hrv_user_mean'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_user_mean']) / (df['sleep_user_mean'] + 1e-6)

# Temporal features
df['exam_proximity_inverse'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['post_exam_proximity_inverse'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)

# Interaction features
df['hr_dev_x_exam'] = df['hr_deviation'] * df['is_exam_period']
df['activity_x_exam'] = df['activity_ratio'] * df['is_exam_period']
df['hrv_dev_x_exam'] = df['hrv_deviation'] * df['is_exam_period']

# Composite features
df['sleep_quality'] = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['activity_intensity'] = df['daily_total_steps'] / (df['activity_level_sedentary_count'] + 1)
df['autonomic_balance'] = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)
df['recovery_score'] = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['cardio_stress'] = df['heart_rate_activity_beats per minute_mean'] / (df['daily_hrv_summary_rmssd'] + 1)

# Enhanced feature set
enhanced_features = available_features + [
    'hr_deviation', 'activity_ratio', 'hrv_deviation', 'sleep_deviation',
    'exam_proximity_inverse', 'post_exam_proximity_inverse',
    'hr_dev_x_exam', 'activity_x_exam', 'hrv_dev_x_exam',
    'sleep_quality', 'activity_intensity', 'autonomic_balance',
    'recovery_score', 'cardio_stress'
]

print(f"✓ Enhanced features: {len(enhanced_features)}")

# ============================================================================
# 3. CREATE SLIDING WINDOW SEQUENCES - BINARY + TIME SERIES SPLIT
# ============================================================================
def create_target_classes_binary(series):
    """
    Create 2-class target from continuous values
    Class 0: Low (below median)
    Class 1: High (at or above median)
    """
    median_val = series.median()
    classes = (series >= median_val).astype(int)
    return classes, median_val

def create_sliding_windows_timeseries(df, target_col, feature_cols, window_size=3):
    """
    Create sliding window sequences for LSTM - BINARY CLASSIFICATION
    WITH TIME SERIES SPLIT (70-15-15)
    
    Maintains temporal order and splits sequentially
    """
    print(f"\n{'='*80}")
    print(f"CREATING SLIDING WINDOWS FOR: {target_col} (BINARY + TIME SERIES SPLIT)")
    print(f"{'='*80}")
    
    # Clean data
    df_clean = df[df[target_col].notna()].copy()
    print(f"✓ Rows with {target_col}: {len(df_clean)}")
    
    # Impute missing values in features
    for col in feature_cols:
        if col in df_clean.columns and df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Create BINARY target classes (Low vs High)
    df_clean['target_class'], median_val = create_target_classes_binary(df_clean[target_col])
    
    print(f"\nBinary Target distribution (overall):")
    print(f"  Class 0 (Low):  {(df_clean['target_class']==0).sum()} samples (<{median_val:.2f})")
    print(f"  Class 1 (High): {(df_clean['target_class']==1).sum()} samples (≥{median_val:.2f})")
    print(f"  Split point (median): {median_val:.2f}")
    
    # Sort by timestamp to maintain temporal order
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    # Create sliding windows across ALL data (maintaining temporal order)
    print(f"\nCreating sliding windows (window_size={window_size} timesteps)...")
    
    X_sequences = []
    y_sequences = []
    
    # Get features and target as arrays
    features = df_clean[feature_cols].values
    targets = df_clean['target_class'].values
    
    # Create sliding windows
    for i in range(window_size, len(df_clean)):
        X_sequences.append(features[i-window_size:i])
        y_sequences.append(targets[i])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    
    print(f"✓ Created {len(X)} sliding window sequences")
    print(f"  Sequence shape: {X.shape}")
    
    # TIME SERIES SPLIT: 70-15-15 (sequential, no shuffling)
    print(f"\n{'='*80}")
    print("TIME SERIES SPLIT (70-15-15)")
    print(f"{'='*80}")
    
    n_samples = len(X)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split sequentially (no shuffling for time series)
    train_idx = slice(0, train_size)
    val_idx = slice(train_size, train_size + val_size)
    test_idx = slice(train_size + val_size, None)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
    
    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]
    
    print(f"\n✓ Data split completed (sequential time series split):")
    print(f"  Train: {len(X_train)} sequences ({len(X_train)/len(X)*100:.1f}%) - earliest data")
    print(f"  Val:   {len(X_val)} sequences ({len(X_val)/len(X)*100:.1f}%) - middle data")
    print(f"  Test:  {len(X_test)} sequences ({len(X_test)/len(X)*100:.1f}%) - most recent data")
    
    # Check class distribution in each split
    print(f"\nClass distribution:")
    print(f"  Train: Low={np.sum(y_train==0)} ({np.sum(y_train==0)/len(y_train)*100:.1f}%), "
          f"High={np.sum(y_train==1)} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
    print(f"  Val:   Low={np.sum(y_val==0)} ({np.sum(y_val==0)/len(y_val)*100:.1f}%), "
          f"High={np.sum(y_val==1)} ({np.sum(y_val==1)/len(y_val)*100:.1f}%)")
    print(f"  Test:  Low={np.sum(y_test==0)} ({np.sum(y_test==0)/len(y_test)*100:.1f}%), "
          f"High={np.sum(y_test==1)} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, median_val

# ============================================================================
# 4. NORMALIZE SEQUENCES
# ============================================================================
def normalize_sequences(X_train, X_val, X_test):
    """Normalize sequence data (fit on train only)"""
    print(f"\n{'='*80}")
    print("NORMALIZING SEQUENCES")
    print(f"{'='*80}")
    
    n_samples, n_timesteps, n_features = X_train.shape
    
    # Reshape for normalization: (samples * timesteps, features)
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape)
    
    print(f"✓ Normalized using StandardScaler (fit on train only)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================================================
# 5. BUILD BINARY LSTM MODEL
# ============================================================================
def build_lstm_model_binary(window_size, n_features):
    """Build LSTM model for binary classification"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(16, return_sequences=False),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')  # Binary output
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# 6. CALCULATE METRICS
# ============================================================================
def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate binary classification metrics"""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    except:
        metrics['roc_auc'] = np.nan
    
    return metrics

# ============================================================================
# 7. TRAIN AND EVALUATE
# ============================================================================
def train_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, 
                   target_name, output_dir, window_size):
    """Train and evaluate binary LSTM"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING BINARY LSTM FOR: {target_name}")
    print(f"{'='*80}")
    
    model = build_lstm_model_binary(window_size, X_train.shape[2])
    print("\nModel Architecture:")
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(os.path.join(output_dir, f'best_{target_name.lower()}_binary.keras'),
                       monitor='val_loss', save_best_only=True)
    ]
    
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✓ Training complete!")
    
    # Predictions
    y_train_proba = model.predict(X_train, verbose=0).flatten()
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # Metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    print("\n" + "="*80)
    print(f"{target_name} RESULTS - BINARY CLASSIFICATION")
    print("="*80)
    
    print(f"\nTrain Metrics:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print(f"  ROC AUC:   {train_metrics['roc_auc']:.4f}")
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    print(f"  ROC AUC:   {val_metrics['roc_auc']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Low', 'High']))
    
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix (Test):")
    print(cm)
    
    # Visualizations
    create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, output_dir)
    
    model.save(os.path.join(output_dir, f'{target_name.lower()}_binary_final.keras'))
    
    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'model': model,
        'history': history,
        'cm': cm
    }

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
def create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, output_dir):
    """Create visualizations"""
    
    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{target_name} - Loss (Binary, Time Series Split)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{target_name} - Accuracy (Binary, Time Series Split)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'history_{target_name.lower()}_binary_ts.png'), dpi=300)
    plt.close()
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    ax.set_title(f'{target_name} - Confusion Matrix (Binary, Time Series Split)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'cm_{target_name.lower()}_binary_ts.png'), dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})', color='#1f77b4')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{target_name} - ROC Curve (Binary, Time Series Split)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'roc_{target_name.lower()}_binary_ts.png'), dpi=300)
    plt.close()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

window_size = 3

print(f"\n{'='*80}")
print("CONFIGURATION")
print(f"{'='*80}")
print(f"  Window: {window_size} timesteps")
print(f"  Classification: BINARY (Low vs High)")
print(f"  Split: Median")
print(f"  Split method: TIME SERIES (70-15-15, sequential)")
print(f"  Train: earliest 70%, Val: middle 15%, Test: most recent 15%")

# ============================================================================
# STRESS PREDICTION
# ============================================================================
print(f"\n{'='*80}")
print("STRESS PREDICTION")
print(f"{'='*80}")

X_tr_s, X_v_s, X_te_s, y_tr_s, y_v_s, y_te_s, med_s = create_sliding_windows_timeseries(
    df, 'stress_level', enhanced_features, window_size
)

X_tr_s, X_v_s, X_te_s, scaler_s = normalize_sequences(X_tr_s, X_v_s, X_te_s)

stress_results = train_evaluate(X_tr_s, X_v_s, X_te_s, y_tr_s, y_v_s, y_te_s, 
                                "STRESS", OUTPUT_DIR, window_size)

import joblib
joblib.dump(scaler_s, os.path.join(OUTPUT_DIR, 'scaler_stress_binary_ts.pkl'))

with open(os.path.join(OUTPUT_DIR, 'stress_median_threshold.txt'), 'w') as f:
    f.write(f"Stress median threshold: {med_s}\n")
    f.write(f"Class 0 (Low): values < {med_s}\n")
    f.write(f"Class 1 (High): values >= {med_s}\n")

# ============================================================================
# ANXIETY PREDICTION
# ============================================================================
print(f"\n{'='*80}")
print("ANXIETY PREDICTION")
print(f"{'='*80}")

X_tr_a, X_v_a, X_te_a, y_tr_a, y_v_a, y_te_a, med_a = create_sliding_windows_timeseries(
    df, 'anxiety_level', enhanced_features, window_size
)

X_tr_a, X_v_a, X_te_a, scaler_a = normalize_sequences(X_tr_a, X_v_a, X_te_a)

anxiety_results = train_evaluate(X_tr_a, X_v_a, X_te_a, y_tr_a, y_v_a, y_te_a,
                                 "ANXIETY", OUTPUT_DIR, window_size)

joblib.dump(scaler_a, os.path.join(OUTPUT_DIR, 'scaler_anxiety_binary_ts.pkl'))

with open(os.path.join(OUTPUT_DIR, 'anxiety_median_threshold.txt'), 'w') as f:
    f.write(f"Anxiety median threshold: {med_a}\n")
    f.write(f"Class 0 (Low): values < {med_a}\n")
    f.write(f"Class 1 (High): values >= {med_a}\n")

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - BINARY CLASSIFICATION WITH TIME SERIES SPLIT")
print("="*80)

print("\n" + "-"*80)
print("STRESS PREDICTION (BINARY: Low vs High)")
print("-"*80)
print(f"Median threshold: {med_s:.2f}")
print(f"Train:      Acc={stress_results['train']['accuracy']:.4f}, Prec={stress_results['train']['precision']:.4f}, "
      f"Rec={stress_results['train']['recall']:.4f}, F1={stress_results['train']['f1']:.4f}, "
      f"AUC={stress_results['train']['roc_auc']:.4f}")
print(f"Validation: Acc={stress_results['val']['accuracy']:.4f}, Prec={stress_results['val']['precision']:.4f}, "
      f"Rec={stress_results['val']['recall']:.4f}, F1={stress_results['val']['f1']:.4f}, "
      f"AUC={stress_results['val']['roc_auc']:.4f}")
print(f"Test:       Acc={stress_results['test']['accuracy']:.4f}, Prec={stress_results['test']['precision']:.4f}, "
      f"Rec={stress_results['test']['recall']:.4f}, F1={stress_results['test']['f1']:.4f}, "
      f"AUC={stress_results['test']['roc_auc']:.4f}")

print("\n" + "-"*80)
print("ANXIETY PREDICTION (BINARY: Low vs High)")
print("-"*80)
print(f"Median threshold: {med_a:.2f}")
print(f"Train:      Acc={anxiety_results['train']['accuracy']:.4f}, Prec={anxiety_results['train']['precision']:.4f}, "
      f"Rec={anxiety_results['train']['recall']:.4f}, F1={anxiety_results['train']['f1']:.4f}, "
      f"AUC={anxiety_results['train']['roc_auc']:.4f}")
print(f"Validation: Acc={anxiety_results['val']['accuracy']:.4f}, Prec={anxiety_results['val']['precision']:.4f}, "
      f"Rec={anxiety_results['val']['recall']:.4f}, F1={anxiety_results['val']['f1']:.4f}, "
      f"AUC={anxiety_results['val']['roc_auc']:.4f}")
print(f"Test:       Acc={anxiety_results['test']['accuracy']:.4f}, Prec={anxiety_results['test']['precision']:.4f}, "
      f"Rec={anxiety_results['test']['recall']:.4f}, F1={anxiety_results['test']['f1']:.4f}, "
      f"AUC={anxiety_results['test']['roc_auc']:.4f}")

# Save summary
summary_data = {
    'Model': ['Stress_Binary_TS', 'Anxiety_Binary_TS'],
    'Split_Method': ['Time_Series_70-15-15', 'Time_Series_70-15-15'],
    'Median_Threshold': [med_s, med_a],
    'Train_Acc': [stress_results['train']['accuracy'], anxiety_results['train']['accuracy']],
    'Val_Acc': [stress_results['val']['accuracy'], anxiety_results['val']['accuracy']],
    'Test_Acc': [stress_results['test']['accuracy'], anxiety_results['test']['accuracy']],
    'Test_Precision': [stress_results['test']['precision'], anxiety_results['test']['precision']],
    'Test_Recall': [stress_results['test']['recall'], anxiety_results['test']['recall']],
    'Test_F1': [stress_results['test']['f1'], anxiety_results['test']['f1']],
    'Test_AUC': [stress_results['test']['roc_auc'], anxiety_results['test']['roc_auc']]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'binary_timeseries_summary.csv'), index=False)

print("\n" + "="*80)
print("KEY FEATURES OF THIS BINARY LSTM IMPLEMENTATION:")
print("="*80)
print(f"✓ Sliding windows of {window_size} timesteps")
print(f"✓ BINARY classification (Low vs High) using median split")
print(f"✓ TIME SERIES SPLIT: 70-15-15 (sequential, no shuffling)")
print(f"  - Train: earliest 70% of data")
print(f"  - Val: middle 15% of data")
print(f"  - Test: most recent 15% of data")
print(f"✓ Proper temporal ordering maintained")
print(f"✓ User-specific baseline features (expanding mean to avoid leakage)")
print(f"✓ Normalization fit only on training data")
print(f"✓ Binary crossentropy loss with sigmoid activation")
print(f"✓ Early stopping with learning rate reduction")
print(f"✓ Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC AUC")

print("\n" + "="*80)
print(f"✅ BINARY LSTM TRAINING COMPLETE!")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print("="*80)