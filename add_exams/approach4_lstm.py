import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/lstm_results_binary.csv'

print("="*80)
print("BI-LSTM BINARY TEMPORAL - OPTIMAL REGULARIZATION")
print("Target Gap: <0.10 | Maintaining Performance")
print("="*80)

print("\nLoading data...")
df = pd.read_csv(INPUT_PATH)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

user_baselines = df.groupby('userid').agg({
    'heart_rate_activity_beats per minute_mean': 'mean',
    'daily_total_steps': 'mean',
    'daily_hrv_summary_rmssd': 'mean',
    'sleep_global_duration': 'mean'
}).reset_index()

user_baselines.columns = ['userid', 'hr_baseline', 'steps_baseline', 'hrv_baseline', 'sleep_baseline']
df = df.merge(user_baselines, on='userid', how='left')

df['hr_deviation'] = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
df['activity_ratio'] = df['daily_total_steps'] / (df['steps_baseline'] + 1)
df['hrv_deviation'] = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)
df['exam_proximity'] = 1 / (df['days_to_next_exam'].fillna(365) + 1)
df['sleep_quality'] = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
df['autonomic_balance'] = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)
df['recovery_score'] = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)

temporal_features = [
    'daily_total_steps', 'daily_hrv_summary_rmssd', 'heart_rate_activity_beats per minute_mean',
    'sleep_global_duration', 'sleep_global_efficiency', 'deep_sleep_minutes', 'rem_sleep_minutes',
    'daily_respiratory_rate_daily_respiratory_rate', 'minute_spo2_value_mean', 'is_exam_period',
    'hr_deviation', 'activity_ratio', 'hrv_deviation', 'sleep_deviation', 'exam_proximity',
    'sleep_quality', 'autonomic_balance', 'recovery_score'
]

# ============================================================================
# CREATE SEQUENCES
# ============================================================================

def create_sequences_binary(df, target_col, feature_cols, sequence_length=3):
    df_clean = df[df[target_col].notna()].copy()
    
    p33 = df_clean[target_col].quantile(0.33)
    p67 = df_clean[target_col].quantile(0.67)
    
    df_binary = df_clean[(df_clean[target_col] < p33) | (df_clean[target_col] > p67)].copy()
    df_binary['target_binary'] = (df_binary[target_col] > p67).astype(int)
    
    df_binary = df_binary.sort_values(['userid', 'unified_date']).reset_index(drop=True)
    
    sequences = []
    targets = []
    
    for user_id in df_binary['userid'].unique():
        user_data = df_binary[df_binary['userid'] == user_id].copy()
        
        if len(user_data) < sequence_length:
            continue
        
        user_features = user_data[feature_cols].values
        user_targets = user_data['target_binary'].values
        
        for i in range(len(user_data) - sequence_length + 1):
            sequences.append(user_features[i:i+sequence_length])
            targets.append(user_targets[i+sequence_length-1])
    
    return np.array(sequences), np.array(targets), p33, p67

def prepare_and_train(df, target_col, target_name):
    print(f"\n{'='*80}")
    print(f"{target_name.upper()}")
    print(f"{'='*80}")
    
    # Create sequences
    X, y, p33, p67 = create_sequences_binary(df, target_col, temporal_features)
    print(f"Sequences: {len(X)} | Low: {(y==0).sum()} | High: {(y==1).sum()}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Normalize
    n_samples, n_timesteps, n_features = X_train.shape
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    imputer = SimpleImputer(strategy='median')
    X_train_flat = imputer.fit_transform(X_train_flat)
    X_val_flat = imputer.transform(X_val_flat)
    X_test_flat = imputer.transform(X_test_flat)
    
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)
    X_test_flat = scaler.transform(X_test_flat)
    
    X_train = X_train_flat.reshape(-1, n_timesteps, n_features)
    X_val = X_val_flat.reshape(-1, n_timesteps, n_features)
    X_test = X_test_flat.reshape(-1, n_timesteps, n_features)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Build model - OPTIMAL REGULARIZATION
    model = Sequential([
        # First Bi-LSTM layer - smaller, more dropout
        Bidirectional(LSTM(26, return_sequences=True, 
                          recurrent_dropout=0.2,  # Increased from 0.15
                          kernel_regularizer=l2(0.008)),  # Increased from 0.005
                     input_shape=(n_timesteps, n_features)),
        Dropout(0.48),  # Increased from 0.45
        BatchNormalization(),
        
        # Second Bi-LSTM layer
        Bidirectional(LSTM(13, return_sequences=False, 
                          recurrent_dropout=0.2,  # Increased from 0.15
                          kernel_regularizer=l2(0.008))),  # Increased from 0.005
        Dropout(0.48),  # Increased from 0.45
        BatchNormalization(),
        
        # Dense layer
        Dense(13, activation='relu', kernel_regularizer=l2(0.008)),  # Increased L2
        Dropout(0.38),  # Increased from 0.35
        
        # Output
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.00035, clipnorm=1.0),  # Lower LR + gradient clipping
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Callbacks with more patience
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=40,  # Increased from 35
        restore_best_weights=True, 
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=15,  # Increased from 12
        min_lr=1e-6, 
        verbose=1
    )
    
    # Train
    print(f"\nTraining {target_name}...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=2
    )
    
    # Evaluate
    y_train_pred = (model.predict(X_train, verbose=0) > 0.5).astype(int).flatten()
    y_val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
    y_test_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    gap = train_acc - test_acc
    
    print(f"\n{'='*80}")
    print(f"{target_name.upper()} RESULTS")
    print(f"{'='*80}")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Train-Test Gap: {gap:.4f}")
    
    if gap <= 0.10:
        print("✓ EXCELLENT: Gap <= 0.10")
    elif gap <= 0.12:
        print("✓ GOOD: Gap <= 0.12")
    else:
        print("⚠️  Gap > 0.12")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Low', 'High'], digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Calculate per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    low_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    high_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\nPer-class accuracy:")
    print(f"  Low (0): {low_acc:.4f}")
    print(f"  High (1): {high_acc:.4f}")
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_auc,
        'gap': gap
    }

# ============================================================================
# TRAIN BOTH MODELS
# ============================================================================

stress_results = prepare_and_train(df, 'stress_level', 'Stress')
anxiety_results = prepare_and_train(df, 'anxiety_level', 'Anxiety')

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY - OPTIMIZED BI-LSTM")
print("="*80)

print("\nSTRESS:")
print(f"  Train: {stress_results['train_acc']:.4f} | Val: {stress_results['val_acc']:.4f} | Test: {stress_results['test_acc']:.4f}")
print(f"  F1: {stress_results['test_f1']:.4f} | Prec: {stress_results['test_precision']:.4f} | Rec: {stress_results['test_recall']:.4f}")
print(f"  AUC: {stress_results['test_auc']:.4f} | Gap: {stress_results['gap']:.4f}")

print("\nANXIETY:")
print(f"  Train: {anxiety_results['train_acc']:.4f} | Val: {anxiety_results['val_acc']:.4f} | Test: {anxiety_results['test_acc']:.4f}")
print(f"  F1: {anxiety_results['test_f1']:.4f} | Prec: {anxiety_results['test_precision']:.4f} | Rec: {anxiety_results['test_recall']:.4f}")
print(f"  AUC: {anxiety_results['test_auc']:.4f} | Gap: {anxiety_results['gap']:.4f}")

avg_gap = (stress_results['gap'] + anxiety_results['gap']) / 2
avg_auc = (stress_results['test_auc'] + anxiety_results['test_auc']) / 2

print(f"\nAVERAGE:")
print(f"  Gap: {avg_gap:.4f}")
print(f"  AUC: {avg_auc:.4f}")

if avg_gap <= 0.10:
    print("  Status: ✓ EXCELLENT GENERALIZATION")
elif avg_gap <= 0.12:
    print("  Status: ✓ GOOD GENERALIZATION")
else:
    print("  Status: ⚠️  MODERATE OVERFITTING")

# Save
results_df = pd.DataFrame({
    'Target': ['Stress', 'Anxiety'],
    'Train_Acc': [stress_results['train_acc'], anxiety_results['train_acc']],
    'Val_Acc': [stress_results['val_acc'], anxiety_results['val_acc']],
    'Test_Acc': [stress_results['test_acc'], anxiety_results['test_acc']],
    'Test_F1': [stress_results['test_f1'], anxiety_results['test_f1']],
    'Test_Precision': [stress_results['test_precision'], anxiety_results['test_precision']],
    'Test_Recall': [stress_results['test_recall'], anxiety_results['test_recall']],
    'Test_AUC': [stress_results['test_auc'], anxiety_results['test_auc']],
    'Train_Test_Gap': [stress_results['gap'], anxiety_results['gap']]
})

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")
print("\n" + "="*80)