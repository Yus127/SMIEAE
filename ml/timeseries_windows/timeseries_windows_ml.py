"""
Stress & Anxiety Prediction from Time Windows (5, 10, 30, 60 min)
Branch: timeseries-windows-prediction

Strategy:
- Join window HR features with daily Fitbit data (sleep, HRV, SpO2, respiration)
- Add lag features (previous stress/anxiety as predictors)
- Four evaluation strategies: random, time-series, per-user, LOPO-CV
- Models: Random Forest, XGBoost, LightGBM, SVM
- Compare all 4 windows + a multi-window fusion model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, roc_auc_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE = '/Users/YusMolina/Downloads/smieae'
WINDOW_DATA = {
    5:  f'{BASE}/data/ml_ready/enriched/ml_ready_5min_window_enriched.csv',
    10: f'{BASE}/data/ml_ready/enriched/ml_ready_10min_window_enriched.csv',
    30: f'{BASE}/data/ml_ready/enriched/ml_ready_30min_window_enriched.csv',
    60: f'{BASE}/data/ml_ready/enriched/ml_ready_60min_window_enriched.csv',
}
DAILY_DATA = f'{BASE}/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR = f'{BASE}/results/timeseries_windows/'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DAILY FITBIT FEATURES (sleep, HRV, SpO2 - strong stress predictors)
# ─────────────────────────────────────────────────────────────────────────────
DAILY_FEATURES = [
    'sleep_global_duration', 'sleep_global_efficiency',
    'deep_sleep_minutes', 'rem_sleep_minutes', 'light_sleep_minutes',
    'wake_count', 'sleep_stage_transitions',
    'micro_awakening_per_hour', 'minutes_to_first_deep_sleep',
    'daily_hrv_summary_rmssd', 'daily_hrv_summary_entropy',
    'hrv_details_rmssd_mean', 'hrv_details_rmssd_min',
    'lf_hf_ratio_hrv_mean',
    'daily_respiratory_rate_daily_respiratory_rate',
    'daily_spo2_average_value', 'minute_spo2_value_mean',
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_std',
    'activity_level_sedentary_count',
    'daily_total_steps',
]

WINDOW_FEATURES = [
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_std',
    'heart_rate_activity_beats per minute_min',
    'heart_rate_activity_beats per minute_max',
    'heart_rate_activity_beats per minute_median',
    'record_count',
]

CONTEXT_FEATURES = [
    'is_exam_period', 'is_easter_break', 'days_until_exam',
    'is_pre_exam_week', 'q_hour',
]

WEEKDAY_MAP = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_enrich(window_min, daily_df):
    """
    Load window data, join with daily Fitbit, add lag features.
    Returns enriched dataframe sorted by (userid, response_timestamp).
    """
    df = pd.read_csv(WINDOW_DATA[window_min])
    df['response_timestamp'] = pd.to_datetime(df['response_timestamp'])
    df['q_date_only'] = pd.to_datetime(df['q_date_only'])
    df['weekday_num'] = df['q_weekday'].map(WEEKDAY_MAP).fillna(0)

    # ---- Rename window HR features to avoid collision with daily HR ----
    hr_cols = {c: f'win_{c}' for c in WINDOW_FEATURES if c in df.columns}
    df = df.rename(columns=hr_cols)
    renamed_window_feats = [hr_cols.get(c, c) for c in WINDOW_FEATURES]

    # ---- Join with daily Fitbit data (sleep, HRV, SpO2 from SAME day) ----
    daily_clean = daily_df[['userid', 'unified_date'] + DAILY_FEATURES].copy()
    daily_clean['unified_date'] = pd.to_datetime(daily_clean['unified_date'])
    daily_clean = daily_clean.drop_duplicates(subset=['userid', 'unified_date'])

    df = df.merge(
        daily_clean,
        left_on=['userid', 'q_date_only'],
        right_on=['userid', 'unified_date'],
        how='left'
    )

    # ---- Raw targets ----
    df['stress_raw'] = df['q_i_stress_sliderNeutralPos']
    df['anxiety_raw'] = df['q_i_anxiety_sliderNeutralPos']

    # ---- Lag features (previous response per user) ----
    df = df.sort_values(['userid', 'response_timestamp']).reset_index(drop=True)
    df['stress_lag1'] = df.groupby('userid')['stress_raw'].shift(1)
    df['anxiety_lag1'] = df.groupby('userid')['anxiety_raw'].shift(1)
    df['stress_lag2'] = df.groupby('userid')['stress_raw'].shift(2)
    df['anxiety_lag2'] = df.groupby('userid')['anxiety_raw'].shift(2)

    # ---- Rolling mean of stress/anxiety (past 3 responses) ----
    df['stress_roll3'] = df.groupby('userid')['stress_raw'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    df['anxiety_roll3'] = df.groupby('userid')['anxiety_raw'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # ---- Days since last response ----
    df['days_since_last'] = df.groupby('userid')['response_timestamp'].diff().dt.total_seconds() / 86400

    df['window_min'] = window_min
    return df, renamed_window_feats


def make_targets(series, bins=None):
    """Convert continuous 0-100 scale to 3-class (Low/Med/High)."""
    if bins is None:
        p33 = series.quantile(0.33)
        p67 = series.quantile(0.67)
        bins = [-np.inf, p33, p67, np.inf]
    labels = pd.cut(series, bins=bins, labels=[0, 1, 2])
    return labels.astype(float).fillna(-1).astype(int), bins


def get_feature_cols(renamed_window_feats):
    """All features used for prediction."""
    lag_feats = ['stress_lag1', 'anxiety_lag1', 'stress_lag2', 'anxiety_lag2',
                 'stress_roll3', 'anxiety_roll3', 'days_since_last']
    return renamed_window_feats + DAILY_FEATURES + CONTEXT_FEATURES + ['weekday_num'] + lag_feats


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

def get_models():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='mlogloss', random_state=RANDOM_STATE,
            verbosity=0
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, class_weight='balanced',
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1
        ),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            probability=True, class_weight='balanced',
            random_state=RANDOM_STATE
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba):
    valid = y_true != -1
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_proba = y_proba[valid]

    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    try:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        if y_bin.shape[1] == 3:
            auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    return dict(accuracy=acc, f1_weighted=f1_w, f1_macro=f1_m,
                precision=prec, recall=rec, roc_auc=auc,
                n_test=len(y_true))


def train_and_eval(model, X_train, y_train, X_test, y_test,
                   imputer=None, scaler=None):
    """Fit imputer+scaler on train, transform both, train model, eval on test."""
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
    if scaler is None:
        scaler = StandardScaler()

    X_tr = imputer.fit_transform(X_train)
    X_tr = scaler.fit_transform(X_tr)
    X_te = imputer.transform(X_test)
    X_te = scaler.transform(X_te)

    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)
    return compute_metrics(y_test, preds, proba)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def random_split(df, feature_cols, target_col):
    """Standard 70/30 random split (baseline)."""
    mask = df[target_col] != -1
    df_v = df[mask].copy()
    X = df_v[feature_cols]
    y = df_v[target_col]
    return train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)


def timeseries_split(df, feature_cols, target_col, test_frac=0.30):
    """
    Chronological split: earliest records → train, latest → test.
    Respects temporal order across all users.
    """
    df_sorted = df.sort_values('response_timestamp').reset_index(drop=True)
    mask = df_sorted[target_col] != -1
    df_v = df_sorted[mask]
    n = len(df_v)
    n_train = int(n * (1 - test_frac))
    X_train = df_v[feature_cols].iloc[:n_train]
    y_train = df_v[target_col].iloc[:n_train]
    X_test = df_v[feature_cols].iloc[n_train:]
    y_test = df_v[target_col].iloc[n_train:]
    return X_train, X_test, y_train, y_test


def per_user_timeseries(df, feature_cols, target_col, test_frac=0.30):
    """
    Per-user chronological split: for each user, train on early, test on late.
    Returns aggregated metrics across all users.
    """
    all_results = {}
    user_results = []

    for uid in sorted(df['userid'].unique()):
        udf = df[df['userid'] == uid].sort_values('response_timestamp')
        mask = udf[target_col] != -1
        udf = udf[mask]
        if len(udf) < 6:
            continue
        n_train = max(1, int(len(udf) * (1 - test_frac)))
        X_train = udf[feature_cols].iloc[:n_train]
        y_train = udf[target_col].iloc[:n_train]
        X_test = udf[feature_cols].iloc[n_train:]
        y_test = udf[target_col].iloc[n_train:]
        if len(y_test) == 0 or len(y_train) == 0:
            continue
        user_results.append((uid, X_train, y_train, X_test, y_test))

    return user_results


def lopo_cv(df, feature_cols, target_col):
    """Leave-One-Person-Out: train on all users except one, test on that one."""
    users = sorted(df['userid'].unique())
    return [(uid, df[df['userid'] != uid], df[df['userid'] == uid])
            for uid in users if (df['userid'] == uid).sum() >= 5]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("STRESS & ANXIETY PREDICTION: TIME WINDOWS (5, 10, 30, 60 min)")
print("Branch: timeseries-windows-prediction")
print("=" * 80)

daily_df = pd.read_csv(DAILY_DATA)
print(f"\nDaily Fitbit data: {len(daily_df)} rows, {len(daily_df.columns)} cols")

all_results = []   # list of dicts for final CSV

# ─────────────────────────────────────────────────────────────────────────────
# LOOP OVER EACH WINDOW SIZE
# ─────────────────────────────────────────────────────────────────────────────
datasets = {}  # store for later multi-window fusion

for window_min in [5, 10, 30, 60]:
    print(f"\n{'='*80}")
    print(f"WINDOW: {window_min} MINUTES")
    print('='*80)

    df, renamed_window_feats = load_and_enrich(window_min, daily_df)
    feature_cols = get_feature_cols(renamed_window_feats)
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Create 3-class targets
    valid_stress = df['stress_raw'].notna()
    valid_anxiety = df['anxiety_raw'].notna()

    stress_vals = df.loc[valid_stress, 'stress_raw']
    anxiety_vals = df.loc[valid_anxiety, 'anxiety_raw']

    stress_targets, stress_bins = make_targets(stress_vals)
    anxiety_targets, anxiety_bins = make_targets(anxiety_vals)

    df['stress_cls'] = -1
    df['anxiety_cls'] = -1
    df.loc[valid_stress, 'stress_cls'] = stress_targets
    df.loc[valid_anxiety, 'anxiety_cls'] = anxiety_targets

    print(f"  Records: {len(df)} | Stress valid: {valid_stress.sum()} | Anxiety valid: {valid_anxiety.sum()}")
    print(f"  Stress bins: {[round(b,1) for b in stress_bins[1:3]]}")
    print(f"  Anxiety bins: {[round(b,1) for b in anxiety_bins[1:3]]}")
    s_dist = pd.Series(stress_targets).value_counts().sort_index()
    a_dist = pd.Series(anxiety_targets).value_counts().sort_index()
    print(f"  Stress class dist (0/1/2): {s_dist.tolist()}")
    print(f"  Anxiety class dist (0/1/2): {a_dist.tolist()}")

    datasets[window_min] = (df, feature_cols, renamed_window_feats)
    models = get_models()

    # ── For each target ──
    for target_name, target_col in [('stress', 'stress_cls'), ('anxiety', 'anxiety_cls')]:
        print(f"\n  ─── Target: {target_name.upper()} ───")

        # ── 1. RANDOM SPLIT (baseline) ──
        print(f"  [1] Random split...")
        X_tr, X_te, y_tr, y_te = random_split(df, feature_cols, target_col)
        for mname, model in models.items():
            m = clone(model)
            metrics = train_and_eval(m, X_tr, y_tr, X_te, y_te)
            all_results.append({
                'window_min': window_min, 'target': target_name,
                'split': 'random', 'model': mname, **metrics
            })
            print(f"     {mname:15s} acc={metrics['accuracy']:.3f} f1={metrics['f1_weighted']:.3f}")

        # ── 2. TIMESERIES SPLIT (chronological) ──
        print(f"  [2] Time-series split (chronological)...")
        X_tr, X_te, y_tr, y_te = timeseries_split(df, feature_cols, target_col)
        for mname, model in models.items():
            m = clone(model)
            metrics = train_and_eval(m, X_tr, y_tr, X_te, y_te)
            all_results.append({
                'window_min': window_min, 'target': target_name,
                'split': 'timeseries', 'model': mname, **metrics
            })
            print(f"     {mname:15s} acc={metrics['accuracy']:.3f} f1={metrics['f1_weighted']:.3f}")

        # ── 3. PER-USER TIMESERIES ──
        print(f"  [3] Per-user time-series split...")
        user_splits = per_user_timeseries(df, feature_cols, target_col)
        per_user_acc = {mname: [] for mname in models}
        per_user_f1 = {mname: [] for mname in models}

        for uid, X_tr, y_tr, X_te, y_te in user_splits:
            if len(y_tr.unique()) < 2:
                continue
            for mname, model in models.items():
                m = clone(model)
                try:
                    met = train_and_eval(m, X_tr, y_tr, X_te, y_te)
                    per_user_acc[mname].append(met['accuracy'])
                    per_user_f1[mname].append(met['f1_weighted'])
                except Exception:
                    pass

        for mname in models:
            if per_user_acc[mname]:
                acc_mean = np.mean(per_user_acc[mname])
                f1_mean = np.mean(per_user_f1[mname])
                all_results.append({
                    'window_min': window_min, 'target': target_name,
                    'split': 'per_user_ts', 'model': mname,
                    'accuracy': acc_mean, 'f1_weighted': f1_mean,
                    'f1_macro': np.nan, 'precision': np.nan,
                    'recall': np.nan, 'roc_auc': np.nan,
                    'n_test': len(per_user_acc[mname])
                })
                print(f"     {mname:15s} mean_acc={acc_mean:.3f} mean_f1={f1_mean:.3f} (over {len(per_user_acc[mname])} users)")

        # ── 4. LOPO-CV ──
        print(f"  [4] Leave-One-Person-Out CV...")
        lopo_splits = lopo_cv(df, feature_cols, target_col)
        lopo_preds_all = {mname: ([], []) for mname in models}

        for uid, df_train, df_test in lopo_splits:
            mask_tr = df_train[target_col] != -1
            mask_te = df_test[target_col] != -1
            if mask_tr.sum() < 10 or mask_te.sum() < 3:
                continue
            X_tr = df_train.loc[mask_tr, feature_cols]
            y_tr = df_train.loc[mask_tr, target_col]
            X_te = df_test.loc[mask_te, feature_cols]
            y_te = df_test.loc[mask_te, target_col]

            if len(y_tr.unique()) < 2:
                continue

            imp = SimpleImputer(strategy='median')
            scl = StandardScaler()
            X_tr_s = scl.fit_transform(imp.fit_transform(X_tr))
            X_te_s = scl.transform(imp.transform(X_te))

            for mname, model in models.items():
                m = clone(model)
                try:
                    m.fit(X_tr_s, y_tr)
                    preds = m.predict(X_te_s)
                    lopo_preds_all[mname][0].extend(y_te.tolist())
                    lopo_preds_all[mname][1].extend(preds.tolist())
                except Exception:
                    pass

        for mname in models:
            y_true_all, y_pred_all = lopo_preds_all[mname]
            if y_true_all:
                acc = accuracy_score(y_true_all, y_pred_all)
                f1_w = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
                f1_m = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
                all_results.append({
                    'window_min': window_min, 'target': target_name,
                    'split': 'lopo_cv', 'model': mname,
                    'accuracy': acc, 'f1_weighted': f1_w, 'f1_macro': f1_m,
                    'precision': np.nan, 'recall': np.nan, 'roc_auc': np.nan,
                    'n_test': len(y_true_all)
                })
                print(f"     {mname:15s} acc={acc:.3f} f1={f1_w:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-WINDOW FUSION: combine all 4 windows into one feature set
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("MULTI-WINDOW FUSION (5+10+30+60 min combined)")
print('='*80)

# Join all 4 on (userid, response_timestamp)
fusion_dfs = []
for wmin in [5, 10, 30, 60]:
    df_w, feat_cols, win_feats = datasets[wmin]
    # Keep only window-specific HR features + identifiers
    keep = ['userid', 'response_timestamp', 'stress_raw', 'anxiety_raw',
            'stress_cls', 'anxiety_cls'] + win_feats + ['q_date_only', 'weekday_num',
            'q_hour', 'is_exam_period', 'is_easter_break', 'days_until_exam', 'is_pre_exam_week',
            'stress_lag1', 'anxiety_lag1', 'stress_lag2', 'anxiety_lag2',
            'stress_roll3', 'anxiety_roll3', 'days_since_last']
    keep = [c for c in keep if c in df_w.columns]
    sub = df_w[keep].rename(columns={c: f'w{wmin}_{c}' for c in win_feats})
    fusion_dfs.append(sub)

# Merge all 4 windows on (userid, response_timestamp)
df_fusion = fusion_dfs[0]
for sub in fusion_dfs[1:]:
    suffix_cols = [c for c in sub.columns if c not in ['userid', 'response_timestamp',
                   'stress_raw', 'anxiety_raw', 'stress_cls', 'anxiety_cls',
                   'q_date_only', 'weekday_num', 'q_hour', 'is_exam_period',
                   'is_easter_break', 'days_until_exam', 'is_pre_exam_week',
                   'stress_lag1', 'anxiety_lag1', 'stress_lag2', 'anxiety_lag2',
                   'stress_roll3', 'anxiety_roll3', 'days_since_last']]
    df_fusion = df_fusion.merge(
        sub[['userid', 'response_timestamp'] + suffix_cols],
        on=['userid', 'response_timestamp'], how='inner'
    )

# Add daily features back
daily_clean = daily_df[['userid', 'unified_date'] + DAILY_FEATURES].copy()
daily_clean['unified_date'] = pd.to_datetime(daily_clean['unified_date'])
daily_clean = daily_clean.drop_duplicates(subset=['userid', 'unified_date'])
df_fusion['q_date_only'] = pd.to_datetime(df_fusion['q_date_only'])
df_fusion = df_fusion.merge(
    daily_clean,
    left_on=['userid', 'q_date_only'],
    right_on=['userid', 'unified_date'],
    how='left'
)

# Fusion feature cols
fusion_win_feats = [f'w{w}_{c}' for w in [5, 10, 30, 60]
                    for c in [f'win_{col}' for col in WINDOW_FEATURES]
                    if f'w{w}_{c}' in df_fusion.columns or f'w{w}_win_{col}' in df_fusion.columns]
# Rebuild correctly
fusion_win_cols = []
for w in [5, 10, 30, 60]:
    for col in WINDOW_FEATURES:
        win_col = f'w{w}_win_{col}'
        if win_col in df_fusion.columns:
            fusion_win_cols.append(win_col)

fusion_feature_cols = (fusion_win_cols + DAILY_FEATURES + CONTEXT_FEATURES +
                       ['weekday_num', 'stress_lag1', 'anxiety_lag1',
                        'stress_lag2', 'anxiety_lag2', 'stress_roll3', 'anxiety_roll3',
                        'days_since_last'])
fusion_feature_cols = [c for c in fusion_feature_cols if c in df_fusion.columns]

print(f"Fusion dataset: {len(df_fusion)} rows, {len(fusion_feature_cols)} features")

for target_name, target_col in [('stress', 'stress_cls'), ('anxiety', 'anxiety_cls')]:
    print(f"\n  ─── Fusion Target: {target_name.upper()} ───")
    for split_name in ['random', 'timeseries']:
        if split_name == 'random':
            X_tr, X_te, y_tr, y_te = random_split(df_fusion, fusion_feature_cols, target_col)
        else:
            X_tr, X_te, y_tr, y_te = timeseries_split(df_fusion, fusion_feature_cols, target_col)

        for mname, model in get_models().items():
            m = clone(model)
            try:
                metrics = train_and_eval(m, X_tr, y_tr, X_te, y_te)
                all_results.append({
                    'window_min': 'fusion', 'target': target_name,
                    'split': split_name, 'model': mname, **metrics
                })
                print(f"   [{split_name:12s}] {mname:15s} acc={metrics['accuracy']:.3f} f1={metrics['f1_weighted']:.3f}")
            except Exception as e:
                print(f"   [{split_name:12s}] {mname:15s} ERROR: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (Random Forest on best setup per target)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("FEATURE IMPORTANCE ANALYSIS (60-min window, Random Forest, Random Split)")
print('='*80)

df60, feat60, win60 = datasets[60]
feature_cols_60 = get_feature_cols(win60)
feature_cols_60 = [c for c in feature_cols_60 if c in df60.columns]

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for ax, (target_name, target_col) in zip(axes, [('Stress', 'stress_cls'), ('Anxiety', 'anxiety_cls')]):
    mask = df60[target_col] != -1
    df_v = df60[mask]
    X = df_v[feature_cols_60]
    y = df_v[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=RANDOM_STATE, stratify=y)
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    X_tr_s = scl.fit_transform(imp.fit_transform(X_tr))

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                 class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr_s, y_tr)

    feat_imp = pd.DataFrame({
        'feature': feature_cols_60,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)

    ax.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue', alpha=0.8)
    ax.set_title(f'Feature Importance: {target_name} (60-min window)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'feature_importance_60min.png', dpi=200, bbox_inches='tight')
print("Saved: feature_importance_60min.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR + 'all_results.csv', index=False)
print(f"\nSaved: all_results.csv ({len(results_df)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 1: Window × Model × Split accuracy heatmap
# ─────────────────────────────────────────────────────────────────────────────

for target_name in ['stress', 'anxiety']:
    tdf = results_df[results_df['target'] == target_name].copy()

    # Pivot: rows = model, cols = (window, split), values = accuracy
    pivot_data = []
    for w in [5, 10, 30, 60, 'fusion']:
        for sp in ['random', 'timeseries', 'lopo_cv', 'per_user_ts']:
            sub = tdf[(tdf['window_min'] == w) & (tdf['split'] == sp)]
            for _, row in sub.iterrows():
                pivot_data.append({
                    'model': row['model'],
                    'config': f'{w}min_{sp[:2].upper()}',
                    'accuracy': row['accuracy']
                })

    if not pivot_data:
        continue

    pvdf = pd.DataFrame(pivot_data)
    try:
        hm = pvdf.pivot_table(index='model', columns='config', values='accuracy')

        fig, ax = plt.subplots(figsize=(max(12, len(hm.columns)), 6))
        sns.heatmap(hm, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.3, vmax=0.8, ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'Accuracy'})
        ax.set_title(f'{target_name.capitalize()} Prediction: Accuracy by Window × Split × Model',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Window × Split Strategy', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f'heatmap_{target_name}.png', dpi=200, bbox_inches='tight')
        print(f"Saved: heatmap_{target_name}.png")
    except Exception as e:
        print(f"Heatmap error for {target_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 2: Window comparison – accuracy bar chart (time-series split)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Window Size Comparison: Accuracy (Time-Series Split)', fontsize=14, fontweight='bold')

windows_to_plot = [5, 10, 30, 60]
model_names = list(get_models().keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
x = np.arange(len(windows_to_plot))
w = 0.2

for ax, target_name in zip(axes, ['stress', 'anxiety']):
    tdf = results_df[(results_df['target'] == target_name) &
                     (results_df['split'] == 'timeseries') &
                     (results_df['window_min'].isin(windows_to_plot))]

    for i, (mname, color) in enumerate(zip(model_names, colors)):
        accs = []
        for wmin in windows_to_plot:
            sub = tdf[(tdf['model'] == mname) & (tdf['window_min'] == wmin)]
            accs.append(sub['accuracy'].values[0] if len(sub) > 0 else 0)
        ax.bar(x + i * w - 1.5 * w, accs, w, label=mname, color=color, alpha=0.85)

    # Baseline
    ax.axhline(1/3, color='black', linestyle='--', linewidth=1, label='Random (33%)', alpha=0.7)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{target_name.capitalize()} Prediction', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w} min' for w in windows_to_plot], fontsize=11)
    ax.set_ylim([0, 0.85])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'window_comparison_ts_split.png', dpi=200, bbox_inches='tight')
print("Saved: window_comparison_ts_split.png")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION 3: Split strategy impact (for best window, all models)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Effect of Split Strategy on Accuracy (60-min window)', fontsize=14, fontweight='bold')

splits_plot = ['random', 'timeseries', 'per_user_ts', 'lopo_cv']
split_labels = ['Random', 'Chronological', 'Per-User TS', 'LOPO-CV']

for ax, target_name in zip(axes, ['stress', 'anxiety']):
    tdf = results_df[(results_df['target'] == target_name) &
                     (results_df['window_min'] == 60)]

    x = np.arange(len(splits_plot))
    for i, (mname, color) in enumerate(zip(model_names, colors)):
        accs = []
        for sp in splits_plot:
            sub = tdf[(tdf['model'] == mname) & (tdf['split'] == sp)]
            accs.append(sub['accuracy'].values[0] if len(sub) > 0 else np.nan)
        ax.plot(x, accs, marker='o', label=mname, color=color, linewidth=2, markersize=8)

    ax.axhline(1/3, color='black', linestyle='--', linewidth=1, label='Chance (33%)', alpha=0.7)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{target_name.capitalize()} Prediction', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=10)
    ax.set_ylim([0.2, 0.85])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'split_strategy_impact.png', dpi=200, bbox_inches='tight')
print("Saved: split_strategy_impact.png")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("FINAL SUMMARY: BEST RESULTS PER WINDOW × TARGET × SPLIT")
print('='*80)

summary_rows = []
for target_name in ['stress', 'anxiety']:
    for split in ['random', 'timeseries', 'lopo_cv', 'per_user_ts']:
        for window_min in [5, 10, 30, 60, 'fusion']:
            sub = results_df[
                (results_df['target'] == target_name) &
                (results_df['split'] == split) &
                (results_df['window_min'] == window_min)
            ]
            if len(sub) == 0:
                continue
            best = sub.loc[sub['accuracy'].idxmax()]
            summary_rows.append({
                'target': target_name,
                'window': window_min,
                'split': split,
                'best_model': best['model'],
                'accuracy': round(best['accuracy'], 4),
                'f1_weighted': round(best['f1_weighted'], 4),
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR + 'summary_best_results.csv', index=False)

print(f"\n{'Window':>8} {'Target':>8} {'Split':>15} {'Best Model':>18} {'Accuracy':>10} {'F1-W':>8}")
print('-' * 75)
for _, row in summary_df.iterrows():
    print(f"{str(row['window']):>8} {row['target']:>8} {row['split']:>15} "
          f"{row['best_model']:>18} {row['accuracy']:>10.4f} {row['f1_weighted']:>8.4f}")

print(f"\n{'='*80}")
print("KEY OBSERVATIONS")
print('='*80)

best_stress = summary_df[summary_df['target'] == 'stress']['accuracy'].max()
best_anxiety = summary_df[summary_df['target'] == 'anxiety']['accuracy'].max()
best_stress_row = summary_df[summary_df['accuracy'] == best_stress].iloc[0]
best_anxiety_row = summary_df[summary_df['accuracy'] == best_anxiety].iloc[0]

print(f"\n[STRESS]  Best: {best_stress:.1%} acc ({best_stress_row['best_model']}, "
      f"{best_stress_row['window']}min, {best_stress_row['split']} split)")
print(f"[ANXIETY] Best: {best_anxiety:.1%} acc ({best_anxiety_row['best_model']}, "
      f"{best_anxiety_row['window']}min, {best_anxiety_row['split']} split)")
print(f"\n[NOTE] Chance level for 3-class: 33.3%")

# Compare random vs time-series split (degradation)
for target_name in ['stress', 'anxiety']:
    for window_min in [5, 10, 30, 60]:
        rand_sub = results_df[
            (results_df['target'] == target_name) &
            (results_df['split'] == 'random') &
            (results_df['window_min'] == window_min)
        ]
        ts_sub = results_df[
            (results_df['target'] == target_name) &
            (results_df['split'] == 'timeseries') &
            (results_df['window_min'] == window_min)
        ]
        if len(rand_sub) > 0 and len(ts_sub) > 0:
            rand_best = rand_sub['accuracy'].max()
            ts_best = ts_sub['accuracy'].max()
            delta = rand_best - ts_best
            print(f"  [{target_name:7s} {window_min:2d}min] Random={rand_best:.3f} vs TS={ts_best:.3f}  "
                  f"Δ={delta:+.3f} {'⚠️ leakage?' if delta > 0.10 else '✓'}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
print("Done!")
