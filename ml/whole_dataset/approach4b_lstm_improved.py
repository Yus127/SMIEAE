"""
approach4b_lstm_improved.py  -  Enhanced BiLSTM + Multi-Head Attention
Dataset : whole dataset  (data_with_exam_features.csv  –  daily aggregates)

Key improvements over approach4:
  1. Sequence length   : 3  →  5  (more temporal context)
  2. Architecture      : BiLSTM(64) + BiLSTM(32,seq) + MultiHeadAttention(2h) + GAP
  3. Evaluation        : LOSO cross-validation (Leave One Subject Out)
                         → no user overlap between train / test
  4. Rolling features  : 3-day & 7-day rolling mean/std per user
  5. Regularisation    : dropout 0.30 (was 0.48), l2=0.005 (was 0.008)
  6. Model size        : 64/32 units (was 26/13)
  7. Reports both LOSO metrics and a final held-out model
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report,
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    BatchNormalization, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# ─── PATHS ────────────────────────────────────────────────────────────────────
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/whole_dataset/loso/model4b'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

SEQ_LEN = 5   # was 3 in approach4

print("=" * 80)
print("APPROACH 4B  –  BiLSTM + MULTI-HEAD ATTENTION  |  WHOLE DATASET  (daily)")
print(f"Sequence length: {SEQ_LEN}  |  Eval: LOSO  |  Features: raw + rolling stats")
print("=" * 80)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"\nLoaded {len(df):,} rows  |  {df['userid'].nunique()} users")

if 'unified_date' in df.columns:
    df = df.sort_values(['userid', 'unified_date']).reset_index(drop=True)

# ─── BASELINE FEATURES ────────────────────────────────────────────────────────
_bl_map = {
    'heart_rate_activity_beats per minute_mean': 'hr_baseline',
    'daily_total_steps':                         'steps_baseline',
    'daily_hrv_summary_rmssd':                   'hrv_baseline',
    'sleep_global_duration':                     'sleep_baseline',
}
_available_bl = {k: v for k, v in _bl_map.items() if k in df.columns}
if _available_bl:
    bl = df.groupby('userid')[list(_available_bl)].mean().rename(columns=_available_bl).reset_index()
    df = df.merge(bl, on='userid', how='left')

    if 'hr_baseline' in df.columns:
        df['hr_deviation']    = (df['heart_rate_activity_beats per minute_mean'] - df['hr_baseline']) / (df['hr_baseline'] + 1e-6)
    if 'steps_baseline' in df.columns:
        df['activity_ratio']  = df['daily_total_steps'] / (df['steps_baseline'] + 1)
    if 'hrv_baseline' in df.columns:
        df['hrv_deviation']   = (df['daily_hrv_summary_rmssd'] - df['hrv_baseline']) / (df['hrv_baseline'] + 1e-6)
    if 'sleep_baseline' in df.columns:
        df['sleep_deviation'] = (df['sleep_global_duration'] - df['sleep_baseline']) / (df['sleep_baseline'] + 1e-6)

if 'days_to_next_exam' in df.columns:
    df['exam_proximity']      = 1 / (df['days_to_next_exam'].fillna(365) + 1)
if 'days_since_last_exam' in df.columns:
    df['post_exam_proximity'] = 1 / (df['days_since_last_exam'].fillna(365) + 1)

# Composite
if all(c in df.columns for c in ['sleep_global_efficiency', 'deep_sleep_minutes', 'sleep_global_duration']):
    df['sleep_quality']   = (df['sleep_global_efficiency'] * df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)
if all(c in df.columns for c in ['daily_total_steps', 'activity_level_sedentary_count']):
    df['activity_intensity'] = df['daily_total_steps'] / (df['activity_level_sedentary_count'] + 1)
if all(c in df.columns for c in ['daily_hrv_summary_rmssd', 'heart_rate_activity_beats per minute_mean']):
    df['autonomic_balance']  = df['daily_hrv_summary_rmssd'] / (df['heart_rate_activity_beats per minute_mean'] + 1)
    df['cardio_stress']      = df['heart_rate_activity_beats per minute_mean'] / (df['daily_hrv_summary_rmssd'] + 1)
if all(c in df.columns for c in ['rem_sleep_minutes', 'deep_sleep_minutes', 'sleep_global_duration']):
    df['recovery_score']     = (df['rem_sleep_minutes'] + df['deep_sleep_minutes']) / (df['sleep_global_duration'] + 1)

# ─── ROLLING FEATURES (PER USER, NO LOOKAHEAD) ───────────────────────────────
_rolling_src = {
    'heart_rate_activity_beats per minute_mean': 'hr',
    'daily_hrv_summary_rmssd':                   'hrv',
    'daily_total_steps':                         'steps',
    'sleep_global_duration':                     'sleep',
}
for src_col, alias in _rolling_src.items():
    if src_col not in df.columns:
        continue
    grp = df.groupby('userid')[src_col]
    df[f'{alias}_roll3_mean'] = grp.transform(lambda x: x.rolling(3, min_periods=1).mean())
    df[f'{alias}_roll7_mean'] = grp.transform(lambda x: x.rolling(7, min_periods=1).mean())
    df[f'{alias}_roll3_std']  = grp.transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
print("Added 3-day & 7-day rolling features per user (no lookahead)")

# ─── FEATURE LIST ─────────────────────────────────────────────────────────────
_CANDIDATE_FEATURES = [
    # Base physiological
    'daily_total_steps', 'daily_hrv_summary_rmssd',
    'heart_rate_activity_beats per minute_mean',
    'sleep_global_duration', 'sleep_global_efficiency',
    'deep_sleep_minutes', 'rem_sleep_minutes',
    'daily_respiratory_rate_daily_respiratory_rate',
    'minute_spo2_value_mean',
    'activity_level_sedentary_count', 'sleep_stage_transitions',
    'hrv_details_rmssd_min', 'daily_hrv_summary_entropy',
    # Exam context
    'is_exam_period', 'is_semana_santa',
    'days_to_next_exam', 'days_since_last_exam',
    'weeks_to_next_exam', 'weeks_since_last_exam',
    # Engineered
    'hr_deviation', 'activity_ratio', 'hrv_deviation', 'sleep_deviation',
    'exam_proximity', 'post_exam_proximity',
    'sleep_quality', 'activity_intensity', 'autonomic_balance',
    'recovery_score', 'cardio_stress',
    # Rolling (new in 4b)
    'hr_roll3_mean', 'hr_roll7_mean', 'hr_roll3_std',
    'hrv_roll3_mean', 'hrv_roll7_mean', 'hrv_roll3_std',
    'steps_roll3_mean', 'steps_roll7_mean', 'steps_roll3_std',
    'sleep_roll3_mean', 'sleep_roll7_mean', 'sleep_roll3_std',
]
FEATURES = [f for f in _CANDIDATE_FEATURES if f in df.columns]
print(f"Feature count: {len(FEATURES)}")


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def create_sequences(data, feature_cols, target_col, seq_len, median_threshold):
    """Sliding-window sequence creation. Labels via global median threshold."""
    data = data[data[target_col].notna()].copy()
    data['_label'] = (data[target_col] >= median_threshold).astype(int)
    X, y, users = [], [], []
    for uid, udf in data.groupby('userid'):
        if len(udf) < seq_len:
            continue
        feats  = udf[feature_cols].values
        labels = udf['_label'].values
        for i in range(len(udf) - seq_len + 1):
            X.append(feats[i:i + seq_len])
            y.append(labels[i + seq_len - 1])
            users.append(uid)
    return np.array(X), np.array(y), np.array(users)


def normalize_splits(X_tr, X_vl, X_te):
    """Fit imputer + scaler on train, apply to all splits."""
    n_ts, n_ft = X_tr.shape[1], X_tr.shape[2]
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xtr = imp.fit_transform(X_tr.reshape(-1, n_ft))
    Xvl = imp.transform(X_vl.reshape(-1, n_ft))
    Xte = imp.transform(X_te.reshape(-1, n_ft))
    Xtr = scl.fit_transform(Xtr)
    Xvl = scl.transform(Xvl)
    Xte = scl.transform(Xte)
    return (
        Xtr.reshape(-1, n_ts, n_ft),
        Xvl.reshape(-1, n_ts, n_ft),
        Xte.reshape(-1, n_ts, n_ft),
        scl, imp,
    )


def get_metrics(y_true, y_pred, y_proba):
    result = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        result['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        result['roc_auc'] = float('nan')
    return result


# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────

def build_model(n_timesteps, n_features):
    """
    BiLSTM(64) → BiLSTM(32, return_seq=True) → MultiHeadAttention(2 heads)
    → GlobalAveragePooling1D → Dense(32) → Dense(1)

    Improvements vs approach4:
      - Attention focuses on informative timesteps
      - Larger capacity (64/32 vs 26/13)
      - Lower dropout (0.30 vs 0.48)
      - Residual + LayerNorm around attention
    """
    reg = l2(0.005)
    inp = Input(shape=(n_timesteps, n_features))

    # Block 1
    x = Bidirectional(LSTM(64, return_sequences=True,
                            recurrent_dropout=0.10,
                            kernel_regularizer=reg))(inp)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    # Block 2 (return_sequences=True to feed attention)
    x = Bidirectional(LSTM(32, return_sequences=True,
                            recurrent_dropout=0.10,
                            kernel_regularizer=reg))(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    # Multi-head self-attention  (key_dim = 64 // num_heads = 32)
    attn = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.10)(x, x)
    x    = LayerNormalization(epsilon=1e-6)(x + attn)      # residual

    # Pooling + classification head
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.20)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


def _make_callbacks(checkpoint_path):
    return [
        EarlyStopping(monitor='val_loss', patience=25,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=12, min_lr=1e-6, verbose=0),
        ModelCheckpoint(checkpoint_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]


# ─── LOSO CROSS-VALIDATION ────────────────────────────────────────────────────

def loso_cv(df, target_col, target_name, global_median):
    """
    Leave-One-Subject-Out cross-validation.
    For each user: train on all other users → evaluate on this user.
    No sequence from the test user appears in training.
    """
    print(f"\n{'#'*80}")
    print(f"LOSO CV  |  {target_name.upper()}")
    print(f"{'#'*80}")
    print(f"Global median threshold: {global_median:.2f}")

    users = df['userid'].unique()
    fold_results = []

    for test_user in users:
        train_df = df[df['userid'] != test_user]
        test_df  = df[df['userid'] == test_user]

        # Skip if too few test observations
        if test_df[target_col].notna().sum() < SEQ_LEN:
            continue

        X_tr, y_tr, _ = create_sequences(train_df, FEATURES, target_col, SEQ_LEN, global_median)
        X_te, y_te, _ = create_sequences(test_df,  FEATURES, target_col, SEQ_LEN, global_median)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        if len(np.unique(y_tr)) < 2:
            print(f"  Skipping user {test_user}: only one class in training fold")
            continue

        # Last 15% of training as validation (time-ordered)
        n_val = max(1, int(0.15 * len(X_tr)))
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_tr2, y_tr2 = X_tr[:-n_val],  y_tr[:-n_val]

        if len(X_tr2) < 4:
            continue

        X_tr2, X_val, X_te_n, _, _ = normalize_splits(X_tr2, X_val, X_te)

        cw_arr  = compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)
        cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

        model = build_model(SEQ_LEN, len(FEATURES))
        ckpt  = os.path.join(OUTPUT_DIR, f'_tmp_loso_{target_name}_{test_user}.keras')
        model.fit(
            X_tr2, y_tr2,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=32,
            class_weight=cw_dict,
            callbacks=_make_callbacks(ckpt),
            verbose=0,
        )
        if os.path.exists(ckpt):
            os.remove(ckpt)

        y_proba = model.predict(X_te_n, verbose=0).flatten()
        y_pred  = (y_proba >= 0.5).astype(int)

        m = {'user': test_user, 'n_test': int(len(y_te))}
        m.update(get_metrics(y_te, y_pred, y_proba))
        fold_results.append(m)

        print(f"  user {str(test_user):>6}  n={m['n_test']:>4}  "
              f"Acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  "
              f"AUC={m['roc_auc']:.3f}")

    return pd.DataFrame(fold_results)


# ─── FINAL MODEL (TRAINED ON ALL DATA) ────────────────────────────────────────

def train_final_model(df, target_col, target_name, global_median, tgt_dir):
    """
    Train one final model on all users with a time-based 80/20 split.
    Used for reporting concrete test metrics and saving the model.
    """
    print(f"\n{'='*80}")
    print(f"FINAL MODEL (ALL DATA)  |  {target_name.upper()}")
    print(f"{'='*80}")

    X, y, _ = create_sequences(df, FEATURES, target_col, SEQ_LEN, global_median)
    if len(X) == 0:
        print("  No sequences created.")
        return None

    n     = len(X)
    n_tr  = int(0.80 * n)
    n_val = max(1, int(0.15 * n_tr))

    X_te, y_te   = X[n_tr:], y[n_tr:]
    X_val, y_val = X[n_tr - n_val:n_tr], y[n_tr - n_val:n_tr]
    X_tr2, y_tr2 = X[:n_tr - n_val], y[:n_tr - n_val]

    X_tr2, X_val, X_te, scaler, imputer = normalize_splits(X_tr2, X_val, X_te)

    cw_arr  = compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)
    cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

    model = build_model(SEQ_LEN, len(FEATURES))
    print("\nModel summary:")
    model.summary()
    print(f"\nTrain={len(X_tr2)}  Val={len(X_val)}  Test={len(X_te)}")
    print(f"Class weights: {cw_dict}")

    os.makedirs(tgt_dir, exist_ok=True)
    ckpt = os.path.join(tgt_dir, f'best_{target_name.lower()}_model.keras')
    history = model.fit(
        X_tr2, y_tr2,
        validation_data=(X_val, y_val),
        epochs=200, batch_size=32,
        class_weight=cw_dict,
        callbacks=_make_callbacks(ckpt),
        verbose=2,
    )

    y_proba = model.predict(X_te, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)
    m       = get_metrics(y_te, y_pred, y_proba)
    cm      = confusion_matrix(y_te, y_pred)

    print(f"\nTest metrics:")
    for k, v in m.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Low','High'], digits=4)}")

    # Save artefacts
    model.save(os.path.join(tgt_dir, f'{target_name.lower()}_final_model.keras'))
    joblib.dump(scaler,  os.path.join(tgt_dir, f'{target_name.lower()}_scaler.pkl'))
    joblib.dump(imputer, os.path.join(tgt_dir, f'{target_name.lower()}_imputer.pkl'))

    _save_plots(history, y_te, y_pred, y_proba, cm, target_name,
                os.path.join(tgt_dir, 'plots'))

    return {**m}


def _save_plots(history, y_te, y_pred, y_proba, cm, target_name, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, label) in zip(axes, [('loss','Loss'), ('accuracy','Accuracy'), ('auc','AUC')]):
        ax.plot(history.history[key],         label='Train', color='#2E86AB', lw=2)
        ax.plot(history.history[f'val_{key}'], label='Val',   color='#A23B72', lw=2)
        ax.set_title(f'{target_name} – {label}', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{target_name.lower()}_history.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','High'], yticklabels=['Low','High'], ax=ax)
    ax.set_title(f'{target_name} – Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{target_name.lower()}_confusion.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # ROC curve
    if len(np.unique(y_te)) > 1:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        roc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(fpr, tpr, lw=2.5, color='#2E86AB',
                label=f'BiLSTM+Attn (AUC={roc_val:.3f})')
        ax.plot([0,1],[0,1], 'k--', lw=2, label='Random (AUC=0.5)')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'{target_name} – ROC Curve', fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{target_name.lower()}_roc.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


def plot_loso_per_user(loso_df, target_name):
    clean = loso_df.dropna(subset=['roc_auc'])
    if len(clean) == 0:
        return
    metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'recall']
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 5))
    for ax, metric in zip(axes, metrics_to_plot):
        if metric not in clean.columns:
            continue
        vals = clean[metric].values
        ax.bar(range(len(vals)), vals, color='#2E86AB', alpha=0.75, edgecolor='white')
        ax.axhline(vals.mean(), color='#A23B72', lw=2, ls='--',
                   label=f'Mean={vals.mean():.3f}')
        ax.set_title(f'{target_name} LOSO – {metric.upper()}', fontweight='bold')
        ax.set_xlabel('Subject fold'); ax.set_ylim([0, 1.0])
        ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots',
                             f'{target_name.lower()}_loso_per_user.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────

TARGETS = [('Stress', 'stress_level'), ('Anxiety', 'anxiety_level')]
summary  = []

for target_name, target_col in TARGETS:
    if target_col not in df.columns:
        print(f"  Column '{target_col}' not found – skipping.")
        continue

    global_median = df[df[target_col].notna()][target_col].median()

    # ── LOSO evaluation ──────────────────────────────────────────────────────
    loso_df = loso_cv(df, target_col, target_name, global_median)

    print(f"\n{'='*80}")
    print(f"LOSO SUMMARY  |  {target_name.upper()}")
    print(f"{'='*80}")
    for m in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
        if m in loso_df.columns:
            vals = loso_df[m].dropna()
            print(f"  {m:12s}: {vals.mean():.4f} ± {vals.std():.4f}"
                  f"  [min={vals.min():.3f}, max={vals.max():.3f}]")

    loso_df.to_csv(
        os.path.join(OUTPUT_DIR, f'{target_name.lower()}_loso_results.csv'),
        index=False,
    )
    plot_loso_per_user(loso_df, target_name)

    # ── Final model ───────────────────────────────────────────────────────────
    tgt_dir = os.path.join(OUTPUT_DIR, target_name.lower())
    os.makedirs(os.path.join(tgt_dir, 'plots'), exist_ok=True)
    final = train_final_model(df, target_col, target_name, global_median, tgt_dir)

    if final:
        summary.append({
            'target':          target_name,
            'global_median':   global_median,
            'loso_n_folds':    len(loso_df),
            'loso_acc_mean':   loso_df['accuracy'].mean(),
            'loso_acc_std':    loso_df['accuracy'].std(),
            'loso_f1_mean':    loso_df['f1'].mean(),
            'loso_f1_std':     loso_df['f1'].std(),
            'loso_auc_mean':   loso_df['roc_auc'].dropna().mean(),
            'loso_auc_std':    loso_df['roc_auc'].dropna().std(),
            'final_acc':       final['accuracy'],
            'final_f1':        final['f1'],
            'final_auc':       final['roc_auc'],
        })

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("FINAL SUMMARY  |  APPROACH 4B  –  WHOLE DATASET")
print(f"{'='*80}")
for s in summary:
    print(f"\n{s['target'].upper()}:")
    print(f"  Median threshold : {s['global_median']:.2f}")
    print(f"  LOSO ({s['loso_n_folds']} folds):")
    print(f"    Accuracy : {s['loso_acc_mean']:.4f} ± {s['loso_acc_std']:.4f}")
    print(f"    F1       : {s['loso_f1_mean']:.4f} ± {s['loso_f1_std']:.4f}")
    print(f"    AUC      : {s['loso_auc_mean']:.4f} ± {s['loso_auc_std']:.4f}")
    print(f"  Final model (time-based 80/20):")
    print(f"    Accuracy : {s['final_acc']:.4f}")
    print(f"    F1       : {s['final_f1']:.4f}")
    print(f"    AUC      : {s['final_auc']:.4f}")

pd.DataFrame(summary).to_csv(
    os.path.join(OUTPUT_DIR, 'approach4b_summary.csv'), index=False
)
print(f"\nAll results saved to: {OUTPUT_DIR}")
