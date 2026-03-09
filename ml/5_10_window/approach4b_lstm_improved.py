"""
approach4b_lstm_improved.py  -  Enhanced BiLSTM + Multi-Head Attention
Datasets : 5-min & 10-min window CSVs  (ml_ready_5min_window_enriched.csv,
                                         ml_ready_10min_window_enriched.csv)

Key improvements over approach4:
  1. Sequence length   : 3  →  5
  2. Architecture      : BiLSTM(64) + BiLSTM(32,seq) + MultiHeadAttention(2h) + GAP
  3. Evaluation        : LOSO cross-validation (Leave One Subject Out)
  4. Rolling features  : 3-measurement rolling mean/std per user
  5. Regularisation    : dropout 0.30 (was 0.40), l2=0.005 (was 0.008)
  6. Model size        : 64/32 units (was 32/16)
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
ML_DIR     = '/Users/YusMolina/Downloads/smieae/data/ml_ready'
OUTPUT_DIR = '/Users/YusMolina/Downloads/smieae/results/5_10_dataset/loso/model4b'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)

SEQ_LEN = 5   # was 3 in approach4

WINDOW_FILES = [
    ('5min',  'w5',  'enriched/ml_ready_5min_window_enriched.csv'),
    ('10min', 'w10', 'enriched/ml_ready_10min_window_enriched.csv'),
]

print("=" * 80)
print("APPROACH 4B  –  BiLSTM + MULTI-HEAD ATTENTION  |  5min & 10min WINDOWS")
print(f"Sequence length: {SEQ_LEN}  |  Eval: LOSO  |  Features: raw + rolling stats")
print("=" * 80)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def build_feature_list(df, window_prefix):
    """Build feature list available in this dataframe."""
    base = [
        'is_exam_period', 'days_until_exam', 'is_pre_exam_week', 'is_easter_break',
        'daily_total_steps',
    ]
    hr_feats = [
        f'{window_prefix}_heart_rate_activity_beats per minute_mean',
        f'{window_prefix}_heart_rate_activity_beats per minute_std',
        f'{window_prefix}_heart_rate_activity_beats per minute_min',
        f'{window_prefix}_heart_rate_activity_beats per minute_max',
        f'{window_prefix}_heart_rate_activity_beats per minute_median',
    ]
    derived = ['hr_deviation', 'steps_ratio',
               'hr_roll3_mean', 'hr_roll3_std', 'steps_roll3_mean']
    candidates = base + hr_feats + derived
    return [c for c in candidates if c in df.columns]


def add_user_features(df, window_prefix):
    """Per-user deviation + rolling features."""
    hr_mean_col = f'{window_prefix}_heart_rate_activity_beats per minute_mean'
    if hr_mean_col in df.columns:
        bl = df.groupby('userid')[hr_mean_col].transform('mean')
        df['hr_deviation'] = df[hr_mean_col] - bl
        # Rolling (3 measurements per user, chronological)
        df['hr_roll3_mean'] = df.groupby('userid')[hr_mean_col].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df['hr_roll3_std']  = df.groupby('userid')[hr_mean_col].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    else:
        df['hr_deviation'] = 0
        df['hr_roll3_mean'] = 0
        df['hr_roll3_std']  = 0

    if 'daily_total_steps' in df.columns:
        bl_steps = df.groupby('userid')['daily_total_steps'].transform('mean')
        df['steps_ratio']     = df['daily_total_steps'] / (bl_steps.replace(0, 1) + 1)
        df['steps_roll3_mean'] = df.groupby('userid')['daily_total_steps'].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
    else:
        df['steps_ratio']     = 0
        df['steps_roll3_mean'] = 0
    return df


def detect_targets(df):
    """Return (stress_col, anxiety_col) or None for each."""
    stress  = ('q_i_stress_sliderNeutralPos'  if 'q_i_stress_sliderNeutralPos'  in df.columns
               else next((c for c in df.columns if 'stress'  in c.lower() and c.startswith('q_')), None))
    anxiety = ('q_i_anxiety_sliderNeutralPos' if 'q_i_anxiety_sliderNeutralPos' in df.columns
               else next((c for c in df.columns if 'anxiety' in c.lower() and c.startswith('q_')), None))
    return stress, anxiety


def create_sequences(data, feature_cols, target_col, seq_len, median_threshold):
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
        'roc_auc':   roc_auc_score(y_true, y_proba)
                     if len(np.unique(y_true)) > 1 else float('nan'),
    }
    return result


# ─── MODEL ────────────────────────────────────────────────────────────────────

def build_model(n_timesteps, n_features):
    """BiLSTM(64) + BiLSTM(32,seq) + MultiHeadAttention(2h) + GAP → Dense(1)"""
    reg = l2(0.005)
    inp = Input(shape=(n_timesteps, n_features))

    x = Bidirectional(LSTM(64, return_sequences=True,
                            recurrent_dropout=0.10,
                            kernel_regularizer=reg))(inp)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    x = Bidirectional(LSTM(32, return_sequences=True,
                            recurrent_dropout=0.10,
                            kernel_regularizer=reg))(x)
    x = Dropout(0.30)(x)
    x = BatchNormalization()(x)

    attn = MultiHeadAttention(num_heads=2, key_dim=32, dropout=0.10)(x, x)
    x    = LayerNormalization(epsilon=1e-6)(x + attn)

    x   = GlobalAveragePooling1D()(x)
    x   = Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x   = Dropout(0.20)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )
    return model


def _callbacks(ckpt_path):
    return [
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=10, min_lr=1e-6, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]


# ─── LOSO EVALUATION ─────────────────────────────────────────────────────────

def loso_cv(df, feature_cols, target_col, target_name, global_median, run_dir):
    print(f"\n{'#'*80}")
    print(f"LOSO CV  |  {target_name.upper()}")
    print(f"{'#'*80}")
    print(f"Global median threshold: {global_median:.2f}")

    users = df['userid'].unique()
    fold_results = []

    for test_user in users:
        train_df = df[df['userid'] != test_user]
        test_df  = df[df['userid'] == test_user]

        if test_df[target_col].notna().sum() < SEQ_LEN:
            continue

        X_tr, y_tr, _ = create_sequences(train_df, feature_cols, target_col, SEQ_LEN, global_median)
        X_te, y_te, _ = create_sequences(test_df,  feature_cols, target_col, SEQ_LEN, global_median)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        if len(np.unique(y_tr)) < 2:
            print(f"  Skipping user {test_user}: one class in training fold")
            continue

        n_val = max(1, int(0.15 * len(X_tr)))
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_tr2, y_tr2 = X_tr[:-n_val],  y_tr[:-n_val]

        if len(X_tr2) < 4:
            continue

        X_tr2, X_val, X_te_n, _, _ = normalize_splits(X_tr2, X_val, X_te)

        cw_arr  = compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)
        cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

        model = build_model(SEQ_LEN, len(feature_cols))
        ckpt  = os.path.join(run_dir, f'_tmp_loso_{test_user}.keras')
        model.fit(
            X_tr2, y_tr2,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=16,
            class_weight=cw_dict,
            callbacks=_callbacks(ckpt),
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


# ─── FINAL MODEL ─────────────────────────────────────────────────────────────

def train_final_model(df, feature_cols, target_col, target_name,
                      global_median, tgt_dir):
    print(f"\n{'='*80}")
    print(f"FINAL MODEL  |  {target_name.upper()}")
    print(f"{'='*80}")

    X, y, _ = create_sequences(df, feature_cols, target_col, SEQ_LEN, global_median)
    if len(X) == 0:
        return None

    n     = len(X)
    n_tr  = int(0.80 * n)
    n_val = max(1, int(0.15 * n_tr))

    X_te, y_te   = X[n_tr:], y[n_tr:]
    X_val, y_val = X[n_tr - n_val:n_tr], y[n_tr - n_val:n_tr]
    X_tr2, y_tr2 = X[:n_tr - n_val], y[:n_tr - n_val]

    if len(X_tr2) == 0:
        return None

    X_tr2, X_val, X_te, scaler, imputer = normalize_splits(X_tr2, X_val, X_te)

    cw_arr  = compute_class_weight('balanced', classes=np.unique(y_tr2), y=y_tr2)
    cw_dict = {i: float(w) for i, w in enumerate(cw_arr)}

    model = build_model(SEQ_LEN, len(feature_cols))
    print(f"Train={len(X_tr2)}  Val={len(X_val)}  Test={len(X_te)}")

    os.makedirs(tgt_dir, exist_ok=True)
    ckpt = os.path.join(tgt_dir, f'best_{target_name.lower()}_model.keras')
    history = model.fit(
        X_tr2, y_tr2,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=16,
        class_weight=cw_dict,
        callbacks=_callbacks(ckpt),
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

    model.save(os.path.join(tgt_dir, f'{target_name.lower()}_final_model.keras'))
    joblib.dump(scaler,  os.path.join(tgt_dir, f'{target_name.lower()}_scaler.pkl'))
    joblib.dump(imputer, os.path.join(tgt_dir, f'{target_name.lower()}_imputer.pkl'))

    _save_plots(history, y_te, y_pred, y_proba, cm, target_name,
                os.path.join(tgt_dir, 'plots'))
    return {**m}


def _save_plots(history, y_te, y_pred, y_proba, cm, label, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, [('loss','Loss'), ('accuracy','Accuracy'), ('auc','AUC')]):
        ax.plot(history.history[key],          label='Train', color='#2E86AB', lw=2)
        ax.plot(history.history[f'val_{key}'], label='Val',   color='#A23B72', lw=2)
        ax.set_title(f'{label} – {title}', fontweight='bold')
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{label.lower()}_history.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low','High'], yticklabels=['Low','High'], ax=ax)
    ax.set_title(f'{label} – Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{label.lower()}_confusion.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    if len(np.unique(y_te)) > 1:
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        roc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(fpr, tpr, lw=2.5, color='#2E86AB',
                label=f'BiLSTM+Attn (AUC={roc_val:.3f})')
        ax.plot([0,1],[0,1],'k--', lw=2, label='Random (AUC=0.5)')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f'{label} – ROC Curve', fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{label.lower()}_roc.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


# ─── MAIN LOOP (over window sizes) ───────────────────────────────────────────

all_results = {}

for window_name, window_prefix, rel_path in WINDOW_FILES:
    data_file = os.path.join(ML_DIR, rel_path)
    if not os.path.exists(data_file):
        print(f"\n  File not found: {data_file}  – skipping {window_name}")
        continue

    print(f"\n{'='*80}")
    print(f"PROCESSING  |  {window_name.upper()} WINDOW")
    print(f"{'='*80}")

    df = pd.read_csv(data_file)

    # Ensure userid column
    if 'userid' not in df.columns and 'user_id' in df.columns:
        df = df.rename(columns={'user_id': 'userid'})

    # Sort chronologically
    if 'response_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['response_timestamp'])
    elif 'timestamp' not in df.columns:
        df['timestamp'] = df.groupby('userid').cumcount()
    df = df.sort_values(['userid', 'timestamp']).reset_index(drop=True)

    print(f"Loaded {len(df):,} rows  |  {df['userid'].nunique()} users")

    df = add_user_features(df, window_prefix)
    feature_cols = build_feature_list(df, window_prefix)
    print(f"Feature count: {len(feature_cols)}")

    stress_col, anxiety_col = detect_targets(df)
    targets_to_run = []
    if stress_col:
        targets_to_run.append(('Stress',  stress_col))
        print(f"Stress target  : {stress_col}")
    if anxiety_col:
        targets_to_run.append(('Anxiety', anxiety_col))
        print(f"Anxiety target : {anxiety_col}")

    if not targets_to_run:
        print("  No valid targets found – skipping.")
        continue

    window_output = os.path.join(OUTPUT_DIR, f'{window_name}_window')
    os.makedirs(window_output, exist_ok=True)
    all_results[window_name] = {}

    for target_name, target_col in targets_to_run:
        global_median = df[df[target_col].notna()][target_col].median()

        tgt_dir  = os.path.join(window_output, target_name.lower())
        loso_dir = os.path.join(tgt_dir, 'loso')
        os.makedirs(loso_dir, exist_ok=True)
        os.makedirs(os.path.join(tgt_dir, 'plots'), exist_ok=True)

        # LOSO
        loso_df = loso_cv(df, feature_cols, target_col, target_name,
                           global_median, loso_dir)

        print(f"\n{'='*80}")
        print(f"LOSO SUMMARY  |  {target_name.upper()}  ({window_name})")
        print(f"{'='*80}")
        for m in ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']:
            if m in loso_df.columns:
                vals = loso_df[m].dropna()
                print(f"  {m:12s}: {vals.mean():.4f} ± {vals.std():.4f}")

        loso_df.to_csv(
            os.path.join(tgt_dir, f'{target_name.lower()}_loso_results.csv'),
            index=False,
        )

        # Final model
        final = train_final_model(df, feature_cols, target_col, target_name,
                                  global_median, tgt_dir)

        if final:
            all_results[window_name][target_name] = {
                'loso_acc_mean': loso_df['accuracy'].mean(),
                'loso_f1_mean':  loso_df['f1'].mean(),
                'loso_auc_mean': loso_df['roc_auc'].dropna().mean(),
                'final_acc':  final['accuracy'],
                'final_f1':   final['f1'],
                'final_auc':  final['roc_auc'],
            }

# ─── OVERALL SUMMARY ─────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("OVERALL SUMMARY  |  APPROACH 4B  –  5min & 10min WINDOWS")
print(f"{'='*80}")

summary_rows = []
for wname in ['5min', '10min']:
    if wname not in all_results:
        continue
    print(f"\n{wname.upper()} window:")
    for tname in ['Stress', 'Anxiety']:
        if tname not in all_results[wname]:
            continue
        r = all_results[wname][tname]
        print(f"  {tname}:")
        print(f"    LOSO  : Acc={r['loso_acc_mean']:.4f}  F1={r['loso_f1_mean']:.4f}  "
              f"AUC={r['loso_auc_mean']:.4f}")
        print(f"    Final : Acc={r['final_acc']:.4f}  F1={r['final_f1']:.4f}  "
              f"AUC={r['final_auc']:.4f}")
        summary_rows.append({'Window': wname, 'Target': tname, **r})

if summary_rows:
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'approach4b_summary.csv'), index=False
    )
    print(f"\nSaved summary to: {OUTPUT_DIR}")
