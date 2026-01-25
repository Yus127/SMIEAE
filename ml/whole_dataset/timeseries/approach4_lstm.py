import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, classification_report, confusion_matrix,
                            roc_auc_score)
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
print("LSTM MODEL FOR STRESS/ANXIETY PREDICTION WITH TEMPORAL SEQUENCES")
print("Train: 70% | Validation: 15% | Test: 15% (TIME SERIES SPLIT)")
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

# Calculate personal baselines for each user
print("Calculating personal baselines...")
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

# Create deviation features
print("Creating deviation features...")
df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
df['activity_ratio'] = (df['daily_total_steps']) / (df['steps_baseline'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)

# Exam proximity features
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
# 3. PREPARE SEQUENCES FOR BOTH TARGETS
# ============================================================================
def create_target_classes(series, p33, p67):
    """Create 3-class target from continuous values"""
    classes = pd.cut(series, bins=[-np.inf, p33, p67, np.inf], labels=[0, 1, 2])
    return classes.astype(int)

def prepare_lstm_data(df, target_col, feature_cols, seq_length=5):
    """
    Prepare data for LSTM with time series split
    """
    print(f"\n{'='*80}")
    print(f"PREPARING LSTM DATA FOR: {target_col}")
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
    
    print(f"  Class 0 (Low):    {(df_clean['target_class']==0).sum()} ({p33:.2f})")
    print(f"  Class 1 (Medium): {(df_clean['target_class']==1).sum()} ({p33:.2f}-{p67:.2f})")
    print(f"  Class 2 (High):   {(df_clean['target_class']==2).sum()} ({p67:.2f})")
    
    # Sort by user and timestamp
    df_clean = df_clean.sort_values(['userid', 'timestamp'])
    
    # Create sequences per user
    print(f"\nCreating sequences (length={seq_length})...")
    
    X_sequences = []
    y_sequences = []
    user_sequences = []
    timestamp_sequences = []
    
    for user in df_clean['userid'].unique():
        user_data = df_clean[df_clean['userid'] == user].copy()
        
        if len(user_data) < seq_length + 1:
            continue  # Skip users with too few observations
        
        # Get features and target
        features = user_data[feature_cols].values
        targets = user_data['target_class'].values
        timestamps = user_data['timestamp'].values
        
        # Create sequences
        for i in range(len(user_data) - seq_length):
            X_sequences.append(features[i:i+seq_length])
            y_sequences.append(targets[i+seq_length])  # Predict next value
            user_sequences.append(user)
            timestamp_sequences.append(timestamps[i+seq_length])
    
    X = np.array(X_sequences)
    y = np.array(y_sequences)
    users = np.array(user_sequences)
    timestamps = np.array(timestamp_sequences)
    
    print(f"✓ Created {len(X)} sequences from {len(df_clean['userid'].unique())} users")
    print(f"  Sequence shape: {X.shape}")  # (samples, timesteps, features)
    
    # Time series split: 70-15-15
    n_samples = len(X)
    train_size = int(0.70 * n_samples)
    val_size = int(0.15 * n_samples)
    
    # Split sequentially by time
    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]
    
    print(f"\nTime series split:")
    print(f"  Train: {len(X_train)} sequences ({len(X_train)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(X_val)} sequences ({len(X_val)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(X_test)} sequences ({len(X_test)/n_samples*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, p33, p67

# ============================================================================
# 4. NORMALIZE AND PREPARE DATA
# ============================================================================
def normalize_sequences(X_train, X_val, X_test):
    """Normalize sequence data"""
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
# 5. BUILD LSTM MODEL
# ============================================================================
def build_lstm_model(seq_length, n_features, n_classes=3):
    """Build LSTM model architecture"""
    model = Sequential([
        # First LSTM layer
        LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        
        # Third LSTM layer
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
# 6. CALCULATE COMPREHENSIVE METRICS
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
# 7. TRAIN AND EVALUATE FOR BOTH TARGETS
# ============================================================================
def train_evaluate_lstm(X_train, X_val, X_test, y_train, y_val, y_test, 
                       target_name, output_dir, seq_length):
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
    model = build_lstm_model(seq_length, n_features)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
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
        epochs=100,
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
    
    print("\nPerformance Summary:")
    print(f"  Train: Acc={train_metrics['accuracy']:.4f}, Prec={train_metrics['precision']:.4f}, "
          f"Rec={train_metrics['recall']:.4f}, F1={train_metrics['f1']:.4f}, AUC={train_metrics['roc_auc']:.4f}")
    print(f"  Val:   Acc={val_metrics['accuracy']:.4f}, Prec={val_metrics['precision']:.4f}, "
          f"Rec={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['roc_auc']:.4f}")
    print(f"  Test:  Acc={test_metrics['accuracy']:.4f}, Prec={test_metrics['precision']:.4f}, "
          f"Rec={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['roc_auc']:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                               target_names=['Low', 'Medium', 'High']))
    
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Visualizations
    create_visualizations(history, y_test, y_test_pred, cm, target_name, output_dir)
    
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
# 8. VISUALIZATION FUNCTION
# ============================================================================
def create_visualizations(history, y_test, y_test_pred, cm, target_name, output_dir):
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

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

seq_length = 5  # Sequence length for LSTM

# Prepare data for stress
X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, p33_s, p67_s = prepare_lstm_data(
    df, 'stress_level', enhanced_features, seq_length
)

# Normalize stress data
X_train_s_scaled, X_val_s_scaled, X_test_s_scaled, scaler_s = normalize_sequences(
    X_train_s, X_val_s, X_test_s
)

# Train stress model
stress_results = train_evaluate_lstm(
    X_train_s_scaled, X_val_s_scaled, X_test_s_scaled,
    y_train_s, y_val_s, y_test_s,
    "STRESS", OUTPUT_DIR, seq_length
)

# Prepare data for anxiety
X_train_a, X_val_a, X_test_a, y_train_a, y_val_a, y_test_a, p33_a, p67_a = prepare_lstm_data(
    df, 'anxiety_level', enhanced_features, seq_length
)

# Normalize anxiety data
X_train_a_scaled, X_val_a_scaled, X_test_a_scaled, scaler_a = normalize_sequences(
    X_train_a, X_val_a, X_test_a
)

# Train anxiety model
anxiety_results = train_evaluate_lstm(
    X_train_a_scaled, X_val_a_scaled, X_test_a_scaled,
    y_train_a, y_val_a, y_test_a,
    "ANXIETY", OUTPUT_DIR, seq_length
)

# ============================================================================
# 10. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - LSTM MODELS (TIME SERIES SPLIT 70-15-15)")
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
    'Train_Acc': [stress_results['train']['accuracy'], anxiety_results['train']['accuracy']],
    'Val_Acc': [stress_results['val']['accuracy'], anxiety_results['val']['accuracy']],
    'Test_Acc': [stress_results['test']['accuracy'], anxiety_results['test']['accuracy']],
    'Test_Precision': [stress_results['test']['precision'], anxiety_results['test']['precision']],
    'Test_Recall': [stress_results['test']['recall'], anxiety_results['test']['recall']],
    'Test_F1': [stress_results['test']['f1'], anxiety_results['test']['f1']],
    'Test_AUC': [stress_results['test']['roc_auc'], anxiety_results['test']['roc_auc']]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'lstm_summary.csv'), index=False)

print("\n" + "="*80)
print(f"✅ LSTM TRAINING COMPLETE!")
print(f"📁 Results saved to: {OUTPUT_DIR}")
print("="*80)