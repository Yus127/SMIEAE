import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
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
print("IMPROVED LSTM MODEL FOR STRESS PREDICTION")
print("="*80)

np.random.seed(42)
tf.random.set_seed(42)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"
output_dir = "/Users/YusMolina/Downloads/smieae/results/lstm_improved"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n1. LOADING DATA")
print("-"*80)

data_file = f"{ml_dir}/ml_ready_combined_windows_enriched.csv"
df = pd.read_csv(data_file)

feature_columns = [
    'is_exam_period', 'days_until_exam', 'is_pre_exam_week', 'is_easter_break',
    'daily_total_steps',
    'w30_heart_rate_activity_beats per minute_mean',
    'w30_heart_rate_activity_beats per minute_std',
    'w30_heart_rate_activity_beats per minute_min',
    'w30_heart_rate_activity_beats per minute_max',
    'w30_heart_rate_activity_beats per minute_median',
    'w60_heart_rate_activity_beats per minute_mean',
    'w60_heart_rate_activity_beats per minute_std',
    'w60_heart_rate_activity_beats per minute_min',
    'w60_heart_rate_activity_beats per minute_max',
    'w60_heart_rate_activity_beats per minute_median',
]

available_features = [col for col in feature_columns if col in df.columns]
primary_target = 'q_i_stress_sliderNeutralPos'

if 'userid' not in df.columns:
    df['userid'] = df['user_id'] if 'user_id' in df.columns else 0

if 'response_timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['response_timestamp'])
else:
    df['timestamp'] = df.groupby('userid').cumcount()

df_clean = df[available_features + [primary_target, 'userid', 'timestamp']].copy()
df_clean = df_clean.dropna(subset=[primary_target])

for col in available_features:
    if df_clean[col].isna().sum() > 0:
        df_clean.loc[:, col] = df_clean[col].fillna(df_clean[col].median())

print(f"✓ Loaded: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")

# ============================================================================
# 2. USER-SPECIFIC FEATURES
# ============================================================================
print("\n2. CREATING USER-SPECIFIC FEATURES")
print("-"*80)

user_hr_baseline = df_clean.groupby('userid')['w30_heart_rate_activity_beats per minute_mean'].transform('mean')
df_clean['hr_deviation'] = df_clean['w30_heart_rate_activity_beats per minute_mean'] - user_hr_baseline

user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline.replace(0, 1) + 1)

df_clean['hr_change_30_60'] = (df_clean['w30_heart_rate_activity_beats per minute_mean'] - 
                                df_clean['w60_heart_rate_activity_beats per minute_mean'])

enhanced_features = available_features + ['hr_deviation', 'steps_ratio', 'hr_change_30_60']

print(f"✓ Enhanced features: {len(enhanced_features)}")

df_clean = df_clean.sort_values(['userid', 'timestamp'])

# ============================================================================
# 3. BINARY CLASSIFICATION (EASIER FOR LSTM)
# ============================================================================
print("\n3. CREATING BINARY TARGET (LOW vs HIGH)")
print("-"*80)

y_continuous = df_clean[primary_target]
p33 = np.percentile(y_continuous, 33.33)
p67 = np.percentile(y_continuous, 66.67)

# Binary: Remove medium, keep only Low and High
mask_binary = (y_continuous <= p33) | (y_continuous >= p67)
df_binary = df_clean[mask_binary].copy()

df_binary['stress_class'] = (df_binary[primary_target] > p67).astype(int)

print(f"  Low (0):  {(df_binary['stress_class'] == 0).sum()} observations (≤{p33:.1f})")
print(f"  High (1): {(df_binary['stress_class'] == 1).sum()} observations (≥{p67:.1f})")
print(f"  Removed:  {(~mask_binary).sum()} medium observations")

# ============================================================================
# 4. CREATE SEQUENCES - OBSERVATION-LEVEL SPLIT
# ============================================================================
print("\n4. CREATING SEQUENCES (OBSERVATION-LEVEL SPLIT)")
print("-"*80)

seq_length = 3  # Shorter sequences (less data needed per user)

print(f"  Sequence length: {seq_length} timesteps")

def create_sequences_all_users(df, user_col, feature_cols, target_col, seq_length):
    """Create sequences from ALL users together"""
    X_sequences = []
    y_sequences = []
    
    for user in df[user_col].unique():
        user_data = df[df[user_col] == user].copy()
        
        if len(user_data) < seq_length + 1:
            continue
        
        features = user_data[feature_cols].values
        targets = user_data[target_col].values
        
        for i in range(len(user_data) - seq_length):
            X_sequences.append(features[i:i+seq_length])
            y_sequences.append(targets[i+seq_length])
    
    return np.array(X_sequences), np.array(y_sequences)

X_sequences, y_sequences = create_sequences_all_users(
    df_binary, 'userid', enhanced_features, 'stress_class', seq_length
)

print(f"✓ Created {len(X_sequences)} sequences")
print(f"  Shape: {X_sequences.shape}")

# ============================================================================
# 5. OBSERVATION-LEVEL SPLIT (NOT USER-STRATIFIED)
# ============================================================================
print("\n5. TRAIN-VAL-TEST SPLIT (70-15-15)")
print("-"*80)

# Random split at observation level (better for small datasets)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_sequences, y_sequences, test_size=0.30, random_state=42, stratify=y_sequences
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train)} sequences")
print(f"  Val:   {len(X_val)} sequences")
print(f"  Test:  {len(X_test)} sequences")

# Check class distribution
print(f"\n  Class distribution (Train): Low={np.sum(y_train==0)}, High={np.sum(y_train==1)}")
print(f"  Class distribution (Val):   Low={np.sum(y_val==0)}, High={np.sum(y_val==1)}")
print(f"  Class distribution (Test):  Low={np.sum(y_test==0)}, High={np.sum(y_test==1)}")

# ============================================================================
# 6. NORMALIZE
# ============================================================================
print("\n6. NORMALIZING DATA")
print("-"*80)

n_samples_train, n_timesteps, n_features = X_train.shape

X_train_reshaped = X_train.reshape(-1, n_features)
X_val_reshaped = X_val.reshape(-1, n_features)
X_test_reshaped = X_test.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

print(f"✓ Data normalized")

# Convert to one-hot for binary
y_train_cat = to_categorical(y_train, num_classes=2)
y_val_cat = to_categorical(y_val, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# ============================================================================
# 7. BUILD SIMPLER LSTM MODEL
# ============================================================================
print("\n7. BUILDING LSTM MODEL")
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
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ LSTM Model Architecture:")
model.summary()

# ============================================================================
# 8. TRAIN WITH CLASS WEIGHTS
# ============================================================================
print("\n8. TRAINING LSTM MODEL")
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
    batch_size=16,  # Smaller batch size for small dataset
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✓ Training complete!")

# ============================================================================
# 9. EVALUATE
# ============================================================================
print("\n9. EVALUATING MODEL")
print("-"*80)

y_train_pred = np.argmax(model.predict(X_train_scaled, verbose=0), axis=1)
y_val_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
y_test_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
train_acc = accuracy_score(y_train, y_train_pred)

val_f1 = f1_score(y_val, y_val_pred, average='weighted')
val_acc = accuracy_score(y_val, y_val_pred)

test_f1 = f1_score(y_test, y_test_pred, average='weighted')
test_acc = accuracy_score(y_test, y_test_pred)

print("\nPerformance Summary:")
print(f"  Train: F1={train_f1:.4f}, Acc={train_acc:.4f}")
print(f"  Val:   F1={val_f1:.4f}, Acc={val_acc:.4f}")
print(f"  Test:  F1={test_f1:.4f}, Acc={test_acc:.4f}")

print("\n" + "="*80)
print("CLASSIFICATION REPORT (TEST SET)")
print("="*80)
print(classification_report(y_test, y_test_pred, target_names=['Low Stress', 'High Stress']))

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n10. CREATING VISUALIZATIONS")
print("-"*80)

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss - Binary LSTM')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Accuracy - Binary LSTM')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'training_history.png'), dpi=300)
print("✓ Saved: training_history.png")
plt.close()

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title('LSTM Confusion Matrix - Binary Classification', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), dpi=300)
print("✓ Saved: confusion_matrix.png")
plt.close()

# ============================================================================
# 11. SAVE
# ============================================================================
print("\n11. SAVING RESULTS")
print("-"*80)

model.save(os.path.join(output_dir, 'lstm_binary_model.keras'))
print("✓ Saved: lstm_binary_model.keras")

import joblib
joblib.dump(scaler, os.path.join(output_dir, 'scaler_lstm.pkl'))
print("✓ Saved: scaler_lstm.pkl")

results = {
    'Model': 'LSTM_Binary',
    'Sequence_Length': seq_length,
    'Train_F1': train_f1,
    'Val_F1': val_f1,
    'Test_F1': test_f1,
    'Test_Acc': test_acc,
    'Epochs': len(history.history['loss']),
    'Parameters': model.count_params()
}

results_df = pd.DataFrame([results])
results_df.to_csv(os.path.join(output_dir, 'lstm_results.csv'), index=False)
print("✓ Saved: lstm_results.csv")

# ============================================================================
# 12. COMPARISON WITH PREVIOUS METHODS
# ============================================================================
print("\n" + "="*80)
print("LSTM MODEL SUMMARY & COMPARISON")
print("="*80)

print(f"\n📊 Model Configuration:")
print(f"   Architecture: Bidirectional LSTM (32→16 units)")
print(f"   Sequence Length: {seq_length} timesteps")
print(f"   Features: {n_features}")
print(f"   Parameters: {model.count_params():,}")
print(f"   Class Weights: Applied (handles imbalance)")

print(f"\n🎯 Performance (Binary Classification):")
print(f"   Test F1:  {test_f1:.4f}")
print(f"   Test Acc: {test_acc:.4f}")

print(f"\n📈 Comparison with Previous Methods:")
print(f"   Baseline (3-class):      F1 = 0.368")
print(f"   User Features (3-class): F1 = 0.494")
print(f"   Binary (no temporal):    F1 = 0.633")
print(f"   LSTM Binary:             F1 = {test_f1:.4f}")

improvement_vs_baseline = ((test_f1 - 0.368) / 0.368) * 100
print(f"\n   Improvement vs baseline: {improvement_vs_baseline:+.1f}%")

print(f"\n💡 Key Improvements:")
print(f"   ✓ Binary classification (removed ambiguous medium)")
print(f"   ✓ Shorter sequences (3 timesteps - more stable)")
print(f"   ✓ Bidirectional LSTM (learns patterns both ways)")
print(f"   ✓ Class weights (handles imbalance)")
print(f"   ✓ Observation-level split (better for small dataset)")

print(f"\n📁 Outputs: {output_dir}")

print("\n" + "="*80)
print("✅ IMPROVED LSTM COMPLETE!")
print("="*80)