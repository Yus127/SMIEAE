import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

print("BI-LSTM BINARY CLASSIFICATION - MEDIAN SPLIT (TIME-BASED SPLIT)")
print("30MIN & 60MIN WINDOWS - SEPARATE PROCESSING")

np.random.seed(42)
tf.random.set_seed(42)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/30_60_dataset/timeseries/model4"
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

# HELPER FUNCTIONS

def create_sequences_median(df, target_col, feature_cols, sequence_length=3):
    """
    Create sequences using MEDIAN split
    Low: below median, High: at or above median
    KEEPS ALL DATA (no exclusions)
    Returns sequences in CHRONOLOGICAL ORDER (no shuffling)
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
    user_ids = []
    timestamps = []
    
    for user_id in df_clean['userid'].unique():
        user_data = df_clean[df_clean['userid'] == user_id].copy()
        
        if len(user_data) < sequence_length:
            continue
        
        user_features = user_data[feature_cols].values
        user_targets = user_data['target_binary'].values
        user_timestamps = user_data['timestamp'].values if 'timestamp' in user_data.columns else np.arange(len(user_data))
        
        # Create overlapping sequences (sliding window)
        for i in range(len(user_data) - sequence_length + 1):
            sequences.append(user_features[i:i+sequence_length])
            targets.append(user_targets[i+sequence_length-1])  # Predict last timestep
            user_ids.append(user_id)
            timestamps.append(user_timestamps[i+sequence_length-1])
    
    X = np.array(sequences)
    y = np.array(targets)
    user_ids = np.array(user_ids)
    timestamps = np.array(timestamps)
    
    print(f" Created {len(sequences)} sequences from {len(df_clean['userid'].unique())} users")
    print(f"  Sequences per user: {len(sequences) / len(df_clean['userid'].unique()):.1f} avg")
    print(f"  Final distribution: Low={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%), High={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    return X, y, user_ids, timestamps, median

def time_based_split(X, y, timestamps, train_ratio=0.70, val_ratio=0.15):
    """
    Split data chronologically (time-based) without shuffling
    70% train, 15% val, 15% test
    """
    print(f"\n{'='*80}")
    print("TIME-BASED SPLIT (70% Train / 15% Val / 15% Test)")
    print("Chronological order preserved - NO SHUFFLING")
    print(f"{'='*80}")
    
    # Sort by timestamp to ensure chronological order
    sort_idx = np.argsort(timestamps)
    X = X[sort_idx]
    y = y[sort_idx]
    
    n_samples = len(X)
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    # Split sequentially (no shuffling for time series)
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    print(f" Train: {len(X_train)} sequences (Low={np.sum(y_train==0)}, High={np.sum(y_train==1)})")
    print(f"  Class balance: Low={np.sum(y_train==0)/len(y_train)*100:.1f}%, High={np.sum(y_train==1)/len(y_train)*100:.1f}%")
    
    print(f"\n Val:   {len(X_val)} sequences (Low={np.sum(y_val==0)}, High={np.sum(y_val==1)})")
    print(f"  Class balance: Low={np.sum(y_val==0)/len(y_val)*100:.1f}%, High={np.sum(y_val==1)/len(y_val)*100:.1f}%")
    
    print(f"\n Test:  {len(X_test)} sequences (Low={np.sum(y_test==0)}, High={np.sum(y_test==1)})")
    print(f"  Class balance: Low={np.sum(y_test==0)/len(y_test)*100:.1f}%, High={np.sum(y_test==1)/len(y_test)*100:.1f}%")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

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

def build_bilstm_model(n_timesteps, n_features):
    """
    Bidirectional LSTM with regularization
    Architecture optimized for binary classification
    """
    model = Sequential([
        # First Bi-LSTM layer (32 units)
        Bidirectional(LSTM(32, return_sequences=True, 
                          recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.008)),
                     input_shape=(n_timesteps, n_features)),
        Dropout(0.4),
        BatchNormalization(),
        
        # Second Bi-LSTM layer (16 units)
        Bidirectional(LSTM(16, return_sequences=False, 
                          recurrent_dropout=0.2,
                          kernel_regularizer=l2(0.008))),
        Dropout(0.4),
        BatchNormalization(),
        
        # Dense layer
        Dense(8, activation='relu', kernel_regularizer=l2(0.008)),
        Dropout(0.3),
        
        # Output layer (binary classification)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

def create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, target_name, window_type, output_dir):
    """Create comprehensive visualizations"""
    
    # 1. Training History
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(history.history['loss'], label='Train', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_loss'], label='Val', linewidth=2, color='#A23B72')
    axes[0].set_title(f'{target_name} - Loss ({window_type})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Train', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_accuracy'], label='Val', linewidth=2, color='#A23B72')
    axes[1].set_title(f'{target_name} - Accuracy ({window_type})', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(history.history['auc'], label='Train', linewidth=2, color='#2E86AB')
    axes[2].plot(history.history['val_auc'], label='Val', linewidth=2, color='#A23B72')
    axes[2].set_title(f'{target_name} - AUC ({window_type})', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('AUC', fontsize=11)
    axes[2].legend(fontsize=10)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'training_history.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={'size': 14},
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    ax.set_title(f'{target_name} - Confusion Matrix ({window_type})', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), 
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
    ax.set_title(f'{target_name} - ROC Curve ({window_type})', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'roc_curve.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(y_test_proba[y_test == 0], bins=30, alpha=0.7, label='True Low', color='#2E86AB')
    axes[0].hist(y_test_proba[y_test == 1], bins=30, alpha=0.7, label='True High', color='#A23B72')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title(f'{target_name} - Prediction Distribution ({window_type})', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        low_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
        high_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        axes[1].bar(['Low', 'High'], [low_acc, high_acc], color=['#2E86AB', '#A23B72'], alpha=0.7)
        axes[1].set_ylim([0, 1.0])
        axes[1].set_ylabel('Accuracy', fontsize=11)
        axes[1].set_title(f'{target_name} - Per-Class Accuracy ({window_type})', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
        
        for i, (label, acc) in enumerate(zip(['Low', 'High'], [low_acc, high_acc])):
            axes[1].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'prediction_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# MAIN PROCESSING LOOP

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
    
    # Ensure userid column exists
    if 'userid' not in df.columns:
        df['userid'] = df['user_id'] if 'user_id' in df.columns else 0
    
    # Create timestamp if needed
    if 'response_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['response_timestamp'])
    elif 'timestamp' not in df.columns:
        df['timestamp'] = df.groupby('userid').cumcount()
    
    # Sort by timestamp (critical for time series)
    if 'timestamp' in df.columns:
        df = df.sort_values(['userid', 'timestamp']).reset_index(drop=True)
        print(" Data sorted by user and timestamp (chronological order)")
    else:
        df = df.sort_values('userid').reset_index(drop=True)
        print(" Warning: No timestamp column, sorted by userid only")
    
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
    
    # Store results for this window
    all_results[window_type] = {}
    
    # PROCESS EACH TARGET
    
    for target_name, primary_target in targets_to_process:
        
        print("\n" + "#"*80)
        print(f"# TARGET: {target_name.upper()} ({window_type} window)")
        print("#"*80)
        
        # Create target-specific output directory
        target_output_dir = os.path.join(window_output_dir, target_name.lower())
        os.makedirs(target_output_dir, exist_ok=True)
        os.makedirs(os.path.join(target_output_dir, "plots"), exist_ok=True)
        
        # 2. PREPARE DATA
        print(f"\n2. DATA PREPARATION ({target_name} - {window_type})")
        print("-"*80)
        
        df_clean = df[available_features + [primary_target, 'userid', 'timestamp']].copy()
        df_clean = df_clean.dropna(subset=[primary_target])
        
        # Impute missing features
        for col in available_features:
            if df_clean[col].isna().sum() > 0:
                median_val = df_clean[col].median()
                df_clean.loc[:, col] = df_clean[col].fillna(median_val)
        
        print(f" Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")
        
        # 3. USER-SPECIFIC FEATURES
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
        
        print(f" Enhanced features: {len(enhanced_features)}")
        
        # 4. CREATE SEQUENCES WITH MEDIAN SPLIT
        seq_length = 3
        X, y, user_ids, timestamps, median = create_sequences_median(
            df_clean, primary_target, enhanced_features, sequence_length=seq_length
        )
        
        # 5. TIME-BASED SPLIT (70-15-15)
        X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
            X, y, timestamps, train_ratio=0.70, val_ratio=0.15
        )
        
        # 6. NORMALIZE
        X_train, X_val, X_test, scaler, imputer = normalize_sequences(X_train, X_val, X_test)
        
        # 7. CLASS WEIGHTS
        class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: class_weights_array[i] for i in range(len(class_weights_array))}
        
        print(f"\n Class weights: {class_weight_dict}")
        
        # 8. BUILD MODEL
        print(f"\n{'='*80}")
        print(f"BUILDING BI-LSTM MODEL ({target_name})")
        print(f"{'='*80}")
        
        n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
        model = build_bilstm_model(n_timesteps, n_features)
        
        model.summary()
        
        # 9. CALLBACKS
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=30,
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
                os.path.join(target_output_dir, f'best_{target_name.lower()}_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # 10. TRAIN
        print(f"\n{'='*80}")
        print(f"TRAINING {target_name.upper()}")
        print(f"{'='*80}")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=16,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=2
        )
        
        print(f"\n Training complete!")
        
        # 11. PREDICT
        y_train_proba = model.predict(X_train, verbose=0).flatten()
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        
        y_val_proba = model.predict(X_val, verbose=0).flatten()
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        
        y_test_proba = model.predict(X_test, verbose=0).flatten()
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        
        # 12. CALCULATE METRICS
        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # 13. PRINT RESULTS
        print(f"\n{'='*80}")
        print(f"{target_name.upper()} RESULTS ({window_type})")
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
        
        # Classification report
        print(f"\n{'-'*80}")
        print(f"Classification Report (Test Set):")
        print(f"{'-'*80}")
        print(classification_report(y_test, y_test_pred, target_names=['Low', 'High'], digits=4))
        
        # Confusion matrix
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
        
        # 14. CREATE VISUALIZATIONS
        print(f"\n Creating visualizations...")
        create_visualizations(history, y_test, y_test_pred, y_test_proba, cm, 
                            target_name, window_type, target_output_dir)
        
        # 15. SAVE MODEL AND ARTIFACTS
        model.save(os.path.join(target_output_dir, f'{target_name.lower()}_final_model.keras'))
        joblib.dump(scaler, os.path.join(target_output_dir, f'{target_name.lower()}_scaler.pkl'))
        joblib.dump(imputer, os.path.join(target_output_dir, f'{target_name.lower()}_imputer.pkl'))
        
        print(f" Saved model, scaler, and imputer")
        
        # Store results
        all_results[window_type][target_name] = {
            'model_type': 'BiLSTM_Binary',
            'sequence_length': seq_length,
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
            'gap': gap,
            'epochs_trained': len(history.history['loss']),
            'total_params': model.count_params(),
            'n_sequences': len(X)
        }
        
        # Save individual results
        results_df = pd.DataFrame([all_results[window_type][target_name]])
        results_df.to_csv(os.path.join(target_output_dir, 'bilstm_results.csv'), index=False)
        print(" Saved: bilstm_results.csv")
        
        print("\n" + "#"*80)
        print(f"#  COMPLETE: {target_name} ({window_type})")
        print("#"*80)

# OVERALL COMPARISON
print("\n\n" + "="*80)
print("OVERALL COMPARISON: STRESS vs ANXIETY (30MIN vs 60MIN)")

if all_results:
    print("\n" + "-"*80)
    print("BI-LSTM MODEL SUMMARY (TIME-BASED SPLIT):")
    print("-"*80)
    
    for window_type in ['30min', '60min']:
        if window_type in all_results:
            print(f"\n{window_type.upper()} Window:")
            for target_name in ['Stress', 'Anxiety']:
                if target_name in all_results[window_type]:
                    res = all_results[window_type][target_name]
                    print(f"  {target_name}:")
                    print(f"    Median threshold: {res['median']:.2f}")
                    print(f"    Train: Acc={res['train_acc']:.4f}, F1={res['train_f1']:.4f}, AUC={res['train_auc']:.4f}")
                    print(f"    Val:   Acc={res['val_acc']:.4f}, F1={res['val_f1']:.4f}, AUC={res['val_auc']:.4f}")
                    print(f"    Test:  Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}, AUC={res['test_auc']:.4f}")
                    print(f"    Gap:   {res['gap']:.4f}")
                    print(f"    Epochs: {res['epochs_trained']}, Sequences: {res['n_sequences']}")
    
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
                        'Model': res['model_type'],
                        'Seq_Length': res['sequence_length'],
                        'Median': res['median'],
                        'Train_Acc': res['train_acc'],
                        'Train_F1': res['train_f1'],
                        'Train_AUC': res['train_auc'],
                        'Val_Acc': res['val_acc'],
                        'Val_F1': res['val_f1'],
                        'Val_AUC': res['val_auc'],
                        'Test_Acc': res['test_acc'],
                        'Test_F1': res['test_f1'],
                        'Test_AUC': res['test_auc'],
                        'Gap': res['gap'],
                        'Epochs': res['epochs_trained'],
                        'Params': res['total_params'],
                        'N_Sequences': res['n_sequences']
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, 'overall_bilstm_timeseries_summary.csv'), index=False)
        print("\n Saved: overall_bilstm_timeseries_summary.csv")


