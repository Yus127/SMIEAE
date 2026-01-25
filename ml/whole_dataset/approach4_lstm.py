import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LSTM MODEL WITH SLIDING WINDOWS (3 TIMESTEPS)")
print("Proper temporal sequence modeling for stress/anxiety prediction")
print("="*80)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model4'

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
# 3. CREATE SLIDING WINDOW SEQUENCES
# ============================================================================
def create_target_classes(series, p33, p67):
    """Create 3-class target from continuous values"""
    classes = pd.cut(series, bins=[-np.inf, p33, p67, np.inf], labels=[0, 1, 2])
    return classes.astype(int)

def create_sliding_windows(df, target_col, feature_cols, window_size=3):
    """
    Create sliding window sequences for LSTM
    Uses window_size previous timesteps to predict current state
    Maintains temporal order within each user
    """
    print(f"\n{'='*80}")
    print(f"CREATING SLIDING WINDOWS FOR: {target_col}")
    print(f"{'='*80}")
    
    # Clean data
    df_clean = df[df[target_col].notna()].copy()
    print(f"✓ Rows with {target_col}: {len(df_clean)}")
    
    # Impute missing values in features
    for col in feature_cols:
        if col in df_clean.columns and df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Create target classes
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    
    df_clean['target_class'] = create_target_classes(df_clean[target_col], p33, p67)
    
    print(f"\nTarget distribution:")
    print(f"  Class 0 (Low):    {(df_clean['target_class']==0).sum()} samples (≤{p33:.2f})")
    print(f"  Class 1 (Medium): {(df_clean['target_class']==1).sum()} samples ({p33:.2f}-{p67:.2f})")
    print(f"  Class 2 (High):   {(df_clean['target_class']==2).sum()} samples (≥{p67:.2f})")
    
    # Sort by user and timestamp
    df_clean = df_clean.sort_values(['userid', 'timestamp']).reset_index(drop=True)
    
    # Create sliding windows per user
    print(f"\nCreating sliding windows (window_size={window_size} timesteps)...")
    
    X_sequences = []
    y_sequences = []
    user_sequences = []
    
    for user in df_clean['userid'].unique():
        user_data = df_clean[df_clean['userid'] == user].copy()
        
        if len(user_data) < window_size:
            continue  # Skip users with too few observations
        
        # Get features and target
        features = user_data[feature_cols].values
        targets = user_data['target_class'].values
        
        # Create sliding windows
        # For each position i, use previous window_size timesteps to predict current state
        for i in range(window_size, len(user_data)):
            X_sequences.append(features[i-window_size:i])  # Previous window_size steps
            y_sequences.append(targets[i])  # Current state
            user_sequences.append(user)
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    users = np.array(user_sequences)
    
    print(f"✓ Created {len(X)} sliding window sequences")
    print(f"  Sequence shape: {X.shape}")  # (samples, window_size, features)
    print(f"  Each sequence uses {window_size} past timesteps to predict current state")
    
    return X, y, users, p33, p67

# ============================================================================
# 4. SPLIT DATA (USER-STRATIFIED TEMPORAL SPLIT)
# ============================================================================
def split_sequences_by_users(X, y, users, train_ratio=0.7, val_ratio=0.15):
    """
    Split sequences by users to prevent data leakage
    Each user appears in only one set (train, val, or test)
    """
    print(f"\nSplitting data by users (train={train_ratio:.0%}, val={val_ratio:.0%}, test={1-train_ratio-val_ratio:.0%})...")
    
    # Get unique users
    unique_users = np.unique(users)
    n_users = len(unique_users)
    
    # Shuffle users randomly
    np.random.shuffle(unique_users)
    
    # Split users
    n_train = int(train_ratio * n_users)
    n_val = int(val_ratio * n_users)
    
    train_users = unique_users[:n_train]
    val_users = unique_users[n_train:n_train + n_val]
    test_users = unique_users[n_train + n_val:]
    
    # Create masks
    train_mask = np.isin(users, train_users)
    val_mask = np.isin(users, val_users)
    test_mask = np.isin(users, test_users)
    
    # Split data
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"✓ Data split completed:")
    print(f"  Train: {len(X_train)} sequences from {len(train_users)} users ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} sequences from {len(val_users)} users ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} sequences from {len(test_users)} users ({len(X_test)/len(X)*100:.1f}%)")
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(f"  Train: Low={np.sum(y_train==0)}, Med={np.sum(y_train==1)}, High={np.sum(y_train==2)}")
    print(f"  Val:   Low={np.sum(y_val==0)}, Med={np.sum(y_val==1)}, High={np.sum(y_val==2)}")
    print(f"  Test:  Low={np.sum(y_test==0)}, Med={np.sum(y_test==1)}, High={np.sum(y_test==2)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# 5. NORMALIZE SEQUENCES
# ============================================================================
def normalize_sequences(X_train, X_val, X_test):
    """Normalize sequence data (fit on train only)"""
    print("\nNormalizing sequences...")
    
    n_samples_train, n_timesteps, n_features = X_train.shape
    
    # Reshape for normalization: (samples * timesteps, features)
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to sequences
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    print(f"✓ Normalized using StandardScaler (fit on train only)")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# ============================================================================
# 6. BUILD LSTM MODEL (OPTIMIZED FOR 3 TIMESTEPS)
# ============================================================================
def build_lstm_model(window_size, n_features, n_classes=3):
    """
    Build LSTM model optimized for short sequences (3 timesteps)
    """
    model = Sequential([
        # First LSTM layer - fewer units for short sequences
        LSTM(32, return_sequences=True, input_shape=(window_size, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(16, return_sequences=False),
        Dropout(0.3),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================================================
# 7. CALCULATE COMPREHENSIVE METRICS
# ============================================================================
def calculate_lstm_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive metrics for LSTM"""
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ROC AUC (one-vs-rest)
    try:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
    except:
        metrics['roc_auc'] = np.nan
    
    return metrics

# ============================================================================
# 8. TRAIN AND EVALUATE FUNCTION
# ============================================================================
def train_evaluate_lstm(X_train, X_val, X_test, y_train, y_val, y_test, 
                       target_name, output_dir, window_size):
    """Train and evaluate LSTM model"""
    
    print(f"\n{'='*80}")
    print(f"TRAINING LSTM FOR: {target_name}")
    print(f"{'='*80}")
    
    # Convert to one-hot
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    # Build model
    n_features = X_train.shape[2]
    model = build_lstm_model(window_size, n_features)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, f'best_lstm_{target_name.lower()}.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    print("\n✓ Training complete!")
    
    # Predictions
    y_train_pred_proba = model.predict(X_train, verbose=0)
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    
    y_val_pred_proba = model.predict(X_val, verbose=0)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    
    y_test_pred_proba = model.predict(X_test, verbose=0)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    # Calculate metrics
    train_metrics = calculate_lstm_metrics(y_train, y_train_pred, y_train_pred_proba)
    val_metrics = calculate_lstm_metrics(y_val, y_val_pred, y_val_pred_proba)
    test_metrics = calculate_lstm_metrics(y_test, y_test_pred, y_test_pred_proba)
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
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
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Low', 'Medium', 'High']))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Visualizations
    create_visualizations(history, y_test, y_test_pred, y_test_pred_proba, cm, target_name, output_dir)
    
    # Save model
    model.save(os.path.join(output_dir, f'lstm_{target_name.lower()}_final.keras'))
    
    results = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'model': model,
        'history': history,
        'confusion_matrix': cm
    }
    
    return results

# ============================================================================
# 9. VISUALIZATION FUNCTION
# ============================================================================
def create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, output_dir):
    """Create and save visualizations"""
    
    # Training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{target_name} - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{target_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'training_history_{target_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Med', 'High'],
                yticklabels=['Low', 'Med', 'High'])
    ax.set_title(f'{target_name} - Confusion Matrix (Test)', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'confusion_matrix_{target_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve (One-vs-Rest for multiclass)
    from itertools import cycle
    
    # Binarize the labels for ROC curve
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_test_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each class
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])
    class_names = ['Low (Class 0)', 'Medium (Class 1)', 'High (Class 2)']
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average ROC curve
    ax.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
            color='deeppink', linestyle=':', linewidth=3)
    
    # Plot macro-average ROC curve
    ax.plot(fpr["macro"], tpr["macro"],
            label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
            color='navy', linestyle=':', linewidth=3)
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{target_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'roc_curve_{target_name.lower()}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: roc_curve_{target_name.lower()}.png")
    plt.close()

# ============================================================================
# 10. MAIN EXECUTION
# ============================================================================

window_size = 3  # Sliding window size (3 timesteps)

print(f"\n{'='*80}")
print(f"CONFIGURATION")
print(f"{'='*80}")
print(f"  Window size: {window_size} timesteps")
print(f"  Split method: User-stratified (no user in multiple sets)")
print(f"  Normalization: StandardScaler (fit on train only)")

# ============================================================================
# STRESS PREDICTION
# ============================================================================

# Create sliding windows
X_stress, y_stress, users_stress, p33_s, p67_s = create_sliding_windows(
    df, 'stress_level', enhanced_features, window_size
)

# Split by users
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = split_sequences_by_users(
    X_stress, y_stress, users_stress
)

# Normalize
X_train_s_scaled, X_val_s_scaled, X_test_s_scaled, scaler_s = normalize_sequences(
    X_train_s, X_val_s, X_test_s
)

# Train
stress_results = train_evaluate_lstm(
    X_train_s_scaled, X_val_s_scaled, X_test_s_scaled,
    y_train_s, y_val_s, y_test_s,
    "STRESS", OUTPUT_DIR, window_size
)

# Save scaler
import joblib
joblib.dump(scaler_s, os.path.join(OUTPUT_DIR, 'scaler_stress.pkl'))

# ============================================================================
# ANXIETY PREDICTION
# ============================================================================

# Create sliding windows
X_anxiety, y_anxiety, users_anxiety, p33_a, p67_a = create_sliding_windows(
    df, 'anxiety_level', enhanced_features, window_size
)

# Split by users
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a = split_sequences_by_users(
    X_anxiety, y_anxiety, users_anxiety
)

# Normalize
X_train_a_scaled, X_val_a_scaled, X_test_a_scaled, scaler_a = normalize_sequences(
    X_train_a, X_val_a, X_test_a
)

# Train
anxiety_results = train_evaluate_lstm(
    X_train_a_scaled, X_val_a_scaled, X_test_a_scaled,
    y_train_a, y_val_a, y_test_a,
    "ANXIETY", OUTPUT_DIR, window_size
)

# Save scaler
joblib.dump(scaler_a, os.path.join(OUTPUT_DIR, 'scaler_anxiety.pkl'))

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - LSTM WITH SLIDING WINDOWS")
print("="*80)

print("\n" + "-"*80)
print("STRESS PREDICTION")
print("-"*80)
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
print("ANXIETY PREDICTION")
print("-"*80)
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
    'Model': ['LSTM_Stress', 'LSTM_Anxiety'],
    'Window_Size': [window_size, window_size],
    'Train_Acc': [stress_results['train']['accuracy'], anxiety_results['train']['accuracy']],
    'Val_Acc': [stress_results['val']['accuracy'], anxiety_results['val']['accuracy']],
    'Test_Acc': [stress_results['test']['accuracy'], anxiety_results['test']['accuracy']],
    'Test_Precision': [stress_results['test']['precision'], anxiety_results['test']['precision']],
    'Test_Recall': [stress_results['test']['recall'], anxiety_results['test']['recall']],
    'Test_F1': [stress_results['test']['f1'], anxiety_results['test']['f1']],
    'Test_AUC': [stress_results['test']['roc_auc'], anxiety_results['test']['roc_auc']]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'lstm_sliding_window_summary.csv'), index=False)

print("\n" + "="*80)
print("KEY FEATURES OF THIS LSTM IMPLEMENTATION:")
print("="*80)
print(f"✓ Sliding windows of {window_size} timesteps")
print(f"✓ User-stratified split (prevents data leakage between users)")
print(f"✓ Proper temporal ordering maintained within each user")
print(f"✓ User-specific baseline features (expanding mean to avoid leakage)")
print(f"✓ Normalization fit only on training data")
print(f"✓ Early stopping with learning rate reduction")
print(f"✓ Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC AUC")

print("\n" + "="*80)
print(f"✅ LSTM TRAINING COMPLETE!")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print("="*80)