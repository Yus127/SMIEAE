import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/random_split/model4'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

print("BI-LSTM BINARY CLASSIFICATION - MEDIAN SPLIT (ALL DATA)")
print("Window: 3 timesteps | Split: Random 70/15/15 | Target: Median (no exclusions)")
print("Using ALL data points for better generalization")

print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)
print(f" Loaded {len(df)} observations from {df['userid'].nunique()} users")

# FEATURE ENGINEERING

print("\n" + "="*80)
print("FEATURE ENGINEERING")

# Sort by user and date
if 'unified_date' in df.columns:
    df = df.sort_values(['userid', 'unified_date']).reset_index(drop=True)
    print(" Sorted by userid and unified_date")
else:
    df = df.sort_values(['userid']).reset_index(drop=True)
    print(" Sorted by userid")

# User baselines (using all data for each user)
user_baselines = df.groupby('userid').agg({
    'heart_rate_activity_beats per minute_mean': 'mean',
    'daily_total_steps': 'mean',
    'daily_hrv_summary_rmssd': 'mean',
    'sleep_global_duration': 'mean'
}).reset_index()

user_baselines.columns = ['userid', 'hr_baseline', 'steps_baseline', 'hrv_baseline', 'sleep_baseline']
df = df.merge(user_baselines, on='userid', how='left')

# Deviation features (personalized to user baseline)
df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
df['activity_ratio'] = df['daily_total_steps'] / (df['steps_baseline'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)

# Temporal features (exam-related)
df['exam_proximity'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['post_exam_proximity'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)

# Composite physiological features
df['sleep_quality'] = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['activity_intensity'] = df['daily_total_steps'] / (df['activity_level_sedentary_count'] + 1)
df['autonomic_balance'] = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)
df['recovery_score'] = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['cardio_stress'] = df['heart_rate_activity_beats per minute_mean'] / (df['daily_hrv_summary_rmssd'] + 1)

# Complete feature set
temporal_features = [
    # Base physiological features
    'daily_total_steps', 
    'daily_hrv_summary_rmssd', 
    'heart_rate_activity_beats per minute_mean',
    'sleep_global_duration', 
    'sleep_global_efficiency', 
    'deep_sleep_minutes', 
    'rem_sleep_minutes',
    'daily_respiratory_rate_daily_respiratory_rate', 
    'minute_spo2_value_mean',
    'activity_level_sedentary_count',
    'sleep_stage_transitions',
    'hrv_details_rmssd_min',
    'daily_hrv_summary_entropy',
    
    # Exam context features
    'is_exam_period',
    'is_semana_santa', 
    'days_to_next_exam', 
    'days_since_last_exam',
    'weeks_to_next_exam',
    'weeks_since_last_exam',
    
    # Engineered features
    'hr_deviation', 
    'activity_ratio', 
    'hrv_deviation', 
    'sleep_deviation', 
    'exam_proximity', 
    'post_exam_proximity',
    'sleep_quality', 
    'activity_intensity', 
    'autonomic_balance', 
    'recovery_score', 
    'cardio_stress'
]

# Keep only available features
temporal_features = [f for f in temporal_features if f in df.columns]
print(f" Using {len(temporal_features)} temporal features")

# CREATE SEQUENCES WITH MEDIAN SPLIT (KEEPS ALL DATA)

def create_sequences_median(df, target_col, feature_cols, sequence_length=3):
    """
    Create sequences using MEDIAN split
    Low: below median, High: at or above median
    KEEPS ALL DATA (no exclusions)
    """
    print(f"\n{'='*80}")
    print(f"Creating sequences for {target_col.upper()}")
    print(f"{'='*80}")
    
    df_clean = df[df[target_col].notna()].copy()
    print(f" Data with {target_col}: {len(df_clean)} observations")
    
    # MEDIAN split (simple, keeps ALL data)
    median = df_clean[target_col].median()
    df_clean['target_binary'] = (df_clean[target_col] >= median).astype(int)
    
    print(f" MEDIAN split: threshold = {median:.2f}")
    print(f"  Low (< {median:.2f}):  {np.sum(df_clean['target_binary']==0)} samples ({np.sum(df_clean['target_binary']==0)/len(df_clean)*100:.1f}%)")
    print(f"  High (≥ {median:.2f}): {np.sum(df_clean['target_binary']==1)} samples ({np.sum(df_clean['target_binary']==1)/len(df_clean)*100:.1f}%)")
    print(f" KEEPING ALL DATA (no middle exclusion)")
    
    sequences = []
    targets = []
    
    for user_id in df_clean['userid'].unique():
        user_data = df_clean[df_clean['userid'] == user_id].copy()
        
        if len(user_data) < sequence_length:
            continue
        
        user_features = user_data[feature_cols].values
        user_targets = user_data['target_binary'].values
        
        # Create overlapping sequences (sliding window)
        for i in range(len(user_data) - sequence_length + 1):
            sequences.append(user_features[i:i+sequence_length])
            targets.append(user_targets[i+sequence_length-1])  # Predict last timestep
    
    X = np.array(sequences)
    y = np.array(targets)
    
    print(f" Created {len(sequences)} sequences from {len(df_clean['userid'].unique())} users")
    print(f"  Sequences per user: {len(sequences) / len(df_clean['userid'].unique()):.1f} avg")
    print(f"  Final distribution: Low={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%), High={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    return X, y, median

# NORMALIZE SEQUENCES

def normalize_sequences(X_train, X_val, X_test):
    """Normalize sequences using StandardScaler"""
    print(f"\n{'='*80}")
    print("NORMALIZATION")
    print(f"{'='*80}")
    
    n_samples, n_timesteps, n_features = X_train.shape
    
    # Flatten for normalization
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_flat = imputer.fit_transform(X_train_flat)
    X_val_flat = imputer.transform(X_val_flat)
    X_test_flat = imputer.transform(X_test_flat)
    
    # Scale
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    # Reshape back to sequences
    X_train = X_train_flat.reshape(-1, n_timesteps, n_features)
    X_val = X_val_flat.reshape(-1, n_timesteps, n_features)
    X_test = X_test_flat.reshape(-1, n_timesteps, n_features)
    
    print(f" Applied median imputation and standard scaling")
    print(f" Final shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return X_train, X_val, X_test, scaler, imputer

# BUILD BI-LSTM MODEL (SAME AS BEFORE)

def build_bilstm_model(n_timesteps, n_features):
    """
    Bidirectional LSTM with regularization
    SAME architecture as percentile version
    """
    model = Sequential([
        # First Bi-LSTM layer (26 units)
        Bidirectional(LSTM(26, return_sequences=True, 
                          recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.008)),
                     input_shape=(n_timesteps, n_features)),
        Dropout(0.48),
        BatchNormalization(),
        
        # Second Bi-LSTM layer (13 units)
        Bidirectional(LSTM(13, return_sequences=False, 
                          recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.008))),
        Dropout(0.48),
        BatchNormalization(),
        
        # Dense layer
        Dense(13, activation='relu', kernel_regularizer=l2(0.008)),
        Dropout(0.38),
        
        # Output layer (binary classification)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00035, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# CALCULATE METRICS

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

# VISUALIZATIONS

def create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, output_dir):
    """Create comprehensive visualizations"""
    
    # 1. Training History
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history.history['loss'], label='Train', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_loss'], label='Val', linewidth=2, color='#A23B72')
    axes[0].set_title(f'{target_name} - Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_accuracy'], label='Val', linewidth=2, color='#A23B72')
    axes[1].set_title(f'{target_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(history.history['auc'], label='Train', linewidth=2, color='#2E86AB')
    axes[2].plot(history.history['val_auc'], label='Val', linewidth=2, color='#A23B72')
    axes[2].set_title(f'{target_name} - AUC', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('AUC', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'{target_name.lower()}_training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={'size': 14},
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    ax.set_title(f'{target_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'{target_name.lower()}_confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc_val = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, lw=3, color='#2E86AB', label=f'Bi-LSTM (AUC = {roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{target_name} - ROC Curve', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'{target_name.lower()}_roc_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, label='True Low', color='#2E86AB')
    axes[0].hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, label='True High', color='#A23B72')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'{target_name} - Prediction Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        low_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        high_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        axes[1].bar(['Low', 'High'], [low_acc, high_acc], color=['#2E86AB', '#A23B72'], alpha=0.7)
        axes[1].set_ylim([0, 1.0])
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title(f'{target_name} - Per-Class Accuracy', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        for i, (label, acc) in enumerate(zip(['Low', 'High'], [low_acc, high_acc])):
            axes[1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', f'{target_name.lower()}_prediction_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# TRAIN AND EVALUATE PIPELINE

def train_and_evaluate(df, target_col, target_name, output_dir):
    """Complete training and evaluation pipeline"""
    
    print(f"\n{'#'*80}")
    print(f"PROCESSING {target_name.upper()}")
    print(f"{'#'*80}")
    
    # 1. Create sequences with MEDIAN split
    X, y, median = create_sequences_median(df, target_col, temporal_features, sequence_length=3)
    
    # 2. Random split (70/15/15)
    print(f"\n{'='*80}")
    print("RANDOM SPLIT (70% Train / 15% Val / 15% Test)")
    print(f"{'='*80}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f" Train: {len(X_train)} sequences (Low={np.sum(y_train==0)}, High={np.sum(y_train==1)})")
    print(f" Val:   {len(X_val)} sequences (Low={np.sum(y_val==0)}, High={np.sum(y_val==1)})")
    print(f" Test:  {len(X_test)} sequences (Low={np.sum(y_test==0)}, High={np.sum(y_test==1)})")
    
    # 3. Normalize
    X_train, X_val, X_test, scaler, imputer = normalize_sequences(X_train, X_val, X_test)
    
    # 4. Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\n Class weights: {class_weight_dict}")
    
    # 5. Build model
    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = build_bilstm_model(n_timesteps, n_features)
    
    print(f"\n{'='*80}")
    print("MODEL ARCHITECTURE (SAME AS PERCENTILE VERSION)")
    print(f"{'='*80}")
    model.summary()
    
    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=40,
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=15,
            min_lr=1e-6, 
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(output_dir, f'best_{target_name.lower()}_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # 7. Train
    print(f"\n{'='*80}")
    print(f"TRAINING {target_name.upper()}")
    print(f"{'='*80}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2
    )
    
    print(f"\n Training complete!")
    
    # 8. Predict
    y_train_proba = model.predict(X_train, verbose=0).flatten()
    y_train_pred = (y_train_proba >= 0.5).astype(int)
    
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    y_val_pred = (y_val_proba >= 0.5).astype(int)
    
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    
    # 9. Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    
    # 10. Print results
    print(f"\n{'='*80}")
    print(f"{target_name.upper()} RESULTS")
    print(f"{'='*80}")
    
    print(f"\nMedian threshold: {median:.2f}")
    print(f"  Low:  < {median:.2f}")
    print(f"  High: ≥ {median:.2f}")
    
    print("\nTrain Set:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall:    {train_metrics['recall']:.4f}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print(f"  AUC:       {train_metrics['roc_auc']:.4f}")
    
    print("\nValidation Set:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    print(f"  AUC:       {val_metrics['roc_auc']:.4f}")
    
    print("\nTest Set:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['roc_auc']:.4f}")
    
    gap = train_metrics['accuracy'] - test_metrics['accuracy']
    print(f"\nTrain-Test Gap: {gap:.4f}")
    
    if gap <= 0.10:
        print("   Excellent generalization (gap ≤ 0.10)")
    elif gap <= 0.15:
        print("   Good generalization (gap ≤ 0.15)")
    else:
        print("   Moderate overfitting (gap > 0.15)")
    
    # 11. Classification report
    print(f"\n{'-'*80}")
    print(f"Classification Report (Test Set):")
    print(f"{'-'*80}")
    print(classification_report(y_test, y_test_pred, target_names=['Low', 'High'], digits=4))
    
    # 12. Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n{'-'*80}")
    print(f"Confusion Matrix (Test Set):")
    print(f"{'-'*80}")
    print(cm)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        low_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        high_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"\nPer-Class Accuracy:")
        print(f"  Low (Class 0):  {low_acc:.4f}")
        print(f"  High (Class 1): {high_acc:.4f}")
    
    # 13. Create visualizations
    print(f"\n Creating visualizations...")
    create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, output_dir)
    
    # 14. Save model and artifacts
    model.save(os.path.join(output_dir, f'{target_name.lower()}_final_model.keras'))
    joblib.dump(scaler, os.path.join(output_dir, f'{target_name.lower()}_scaler.pkl'))
    joblib.dump(imputer, os.path.join(output_dir, f'{target_name.lower()}_imputer.pkl'))
    
    print(f" Saved model, scaler, and imputer")
    
    return {
        'target': target_name,
        'median': median,
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'train_auc': train_metrics['roc_auc'],
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['f1'],
        'val_auc': val_metrics['roc_auc'],
        'test_acc': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_f1': test_metrics['f1'],
        'test_auc': test_metrics['roc_auc'],
        'gap': gap
    }

# MAIN EXECUTION

print(f"\n{'='*80}")
print("STARTING TRAINING PIPELINE")
print(f"{'='*80}")

# Train Stress model
stress_results = train_and_evaluate(df, 'stress_level', 'Stress', OUTPUT_DIR)

# Train Anxiety model
anxiety_results = train_and_evaluate(df, 'anxiety_level', 'Anxiety', OUTPUT_DIR)

# FINAL SUMMARY

print("\n" + "="*80)
print("FINAL SUMMARY - MEDIAN SPLIT (ALL DATA)")

print(f"\nSTRESS:")
print(f"  Median threshold: {stress_results['median']:.2f}")
print(f"  Train: Acc={stress_results['train_acc']:.4f}, F1={stress_results['train_f1']:.4f}, AUC={stress_results['train_auc']:.4f}")
print(f"  Val:   Acc={stress_results['val_acc']:.4f}, F1={stress_results['val_f1']:.4f}, AUC={stress_results['val_auc']:.4f}")
print(f"  Test:  Acc={stress_results['test_acc']:.4f}, F1={stress_results['test_f1']:.4f}, AUC={stress_results['test_auc']:.4f}")
print(f"  Gap:   {stress_results['gap']:.4f}")

print(f"\nANXIETY:")
print(f"  Median threshold: {anxiety_results['median']:.2f}")
print(f"  Train: Acc={anxiety_results['train_acc']:.4f}, F1={anxiety_results['train_f1']:.4f}, AUC={anxiety_results['train_auc']:.4f}")
print(f"  Val:   Acc={anxiety_results['val_acc']:.4f}, F1={anxiety_results['val_f1']:.4f}, AUC={anxiety_results['val_auc']:.4f}")
print(f"  Test:  Acc={anxiety_results['test_acc']:.4f}, F1={anxiety_results['test_f1']:.4f}, AUC={anxiety_results['test_auc']:.4f}")
print(f"  Gap:   {anxiety_results['gap']:.4f}")

# Calculate averages
avg_test_acc = (stress_results['test_acc'] + anxiety_results['test_acc']) / 2
avg_test_f1 = (stress_results['test_f1'] + anxiety_results['test_f1']) / 2
avg_test_auc = (stress_results['test_auc'] + anxiety_results['test_auc']) / 2
avg_gap = (stress_results['gap'] + anxiety_results['gap']) / 2

print(f"\nAVERAGE (Stress + Anxiety):")
print(f"  Test Accuracy:  {avg_test_acc:.4f}")
print(f"  Test F1:        {avg_test_f1:.4f}")
print(f"  Test AUC:       {avg_test_auc:.4f}")
print(f"  Train-Test Gap: {avg_gap:.4f}")

# Save summary
summary_df = pd.DataFrame([stress_results, anxiety_results])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'median_split_results_summary.csv'), index=False)

print(f"\n{'='*80}")
print("COMPARISON: PERCENTILE vs MEDIAN SPLIT")
print(f"{'='*80}")
print("\nPercentile (33/67) approach:")
print("   Better class separation (excludes ambiguous middle)")
print("   Higher performance metrics")
print("   Discards ~34% of data")
print("   May overfit to extreme cases")

print("\nMedian approach (this model):")
print("   Uses ALL data (better for generalization)")
print("   More balanced training")
print("   Better represents full spectrum")
print("   Includes ambiguous boundary cases")
print("   Slightly lower metrics expected")

