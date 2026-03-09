"""
Stress & Anxiety Prediction — SEMESTER-AWARE SPLITS (pure physiological)
Two clean evaluations:
  1. Within-S1 TS   : train on S1 early (70%), test on S1 late (30%)
  2. Cross-semester : train on ALL of S1,   test on ALL of S2

Semester definitions (from data inspection):
  S1  Feb–Jul 2025  users 1-35   (~31 users with Fitbit data)
  S2  Oct–Dec 2025  users 37,41,44,46,50  (5 users)

Branch: timeseries-windows-prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE       = '/Users/YusMolina/Downloads/smieae'
DAILY_DATA = f'{BASE}/data/data_clean/csv_joined/data_with_exam_features.csv'
WINDOW_DATA = {
    5:  f'{BASE}/data/ml_ready/enriched/ml_ready_5min_window_enriched.csv',
    10: f'{BASE}/data/ml_ready/enriched/ml_ready_10min_window_enriched.csv',
    30: f'{BASE}/data/ml_ready/enriched/ml_ready_30min_window_enriched.csv',
    60: f'{BASE}/data/ml_ready/enriched/ml_ready_60min_window_enriched.csv',
}
OUTPUT_DIR = f'{BASE}/results/timeseries_windows/semester_splits/'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Semester membership ────────────────────────────────────────────────────
S1_USERS = list(range(1, 36))          # users 1-35, Feb-Jul 2025
S2_USERS = [37, 41, 44, 46, 50]       # users in Oct-Dec 2025
S1_END   = pd.Timestamp('2025-07-31')  # conservative S1 boundary
S2_START = pd.Timestamp('2025-10-01') # S2 starts October

# ── Features (pure physiological, same as previous run) ───────────────────
WINDOW_FEATURES = [
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_std',
    'heart_rate_activity_beats per minute_min',
    'heart_rate_activity_beats per minute_max',
    'heart_rate_activity_beats per minute_median',
    'record_count',
]
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
CONTEXT_FEATURES = [
    'is_exam_period', 'is_easter_break', 'days_until_exam',
    'is_pre_exam_week', 'q_hour',
]
WEEKDAY_MAP = {
    'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
    'Friday':4,'Saturday':5,'Sunday':6
}

# ── Helpers ────────────────────────────────────────────────────────────────

def load_window(window_min, daily_df):
    df = pd.read_csv(WINDOW_DATA[window_min])
    df['response_timestamp'] = pd.to_datetime(df['response_timestamp'])
    df['q_date_only']        = pd.to_datetime(df['q_date_only'])
    df['weekday_num']        = df['q_weekday'].map(WEEKDAY_MAP).fillna(0)

    hr_rename = {c: f'win_{c}' for c in WINDOW_FEATURES if c in df.columns}
    df = df.rename(columns=hr_rename)

    daily_clean = daily_df[['userid','unified_date'] + DAILY_FEATURES].copy()
    daily_clean['unified_date'] = pd.to_datetime(daily_clean['unified_date'])
    daily_clean = daily_clean.drop_duplicates(subset=['userid','unified_date'])
    df = df.merge(daily_clean,
                  left_on=['userid','q_date_only'],
                  right_on=['userid','unified_date'], how='left')

    df['stress_raw']  = df['q_i_stress_sliderNeutralPos']
    df['anxiety_raw'] = df['q_i_anxiety_sliderNeutralPos']
    df = df.sort_values(['userid','response_timestamp']).reset_index(drop=True)
    df['window_min'] = window_min

    win_cols     = [c for c in df.columns if c.startswith('win_')]
    feature_cols = win_cols + DAILY_FEATURES + CONTEXT_FEATURES + ['weekday_num']
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def make_targets_from_train(train_series, all_series):
    """Compute percentile bins on TRAIN only, apply to all."""
    p33 = train_series.quantile(0.33)
    p67 = train_series.quantile(0.67)
    bins = [-np.inf, p33, p67, np.inf]
    labels = pd.cut(all_series, bins=bins, labels=[0,1,2])
    return labels.astype(float).fillna(-1).astype(int), p33, p67


def run_model(model, X_tr, y_tr, X_te, y_te):
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xtr = scl.fit_transform(imp.fit_transform(X_tr))
    Xte = scl.transform(imp.transform(X_te))
    model.fit(Xtr, y_tr)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)
    valid = y_te != -1
    yt, yp, ypr = y_te[valid], preds[valid], proba[valid]
    if len(yt) == 0:
        return None
    acc = accuracy_score(yt, yp)
    f1w = f1_score(yt, yp, average='weighted', zero_division=0)
    f1m = f1_score(yt, yp, average='macro',    zero_division=0)
    try:
        yb  = label_binarize(yt, classes=[0,1,2])
        auc = roc_auc_score(yb, ypr, average='macro', multi_class='ovr') \
              if yb.shape[1] == 3 else np.nan
    except Exception:
        auc = np.nan
    return dict(accuracy=acc, f1_weighted=f1w, f1_macro=f1m,
                roc_auc=auc, n_test=int(valid.sum()),
                report=classification_report(yt, yp,
                       target_names=['Low','Med','High'], zero_division=0))


def get_models():
    return {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric='mlogloss',
            random_state=RANDOM_STATE, verbosity=0),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, class_weight='balanced',
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True,
            class_weight='balanced', random_state=RANDOM_STATE),
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

print('=' * 80)
print('SEMESTER-AWARE SPLITS  —  Pure Physiological')
print('  Split A: Within-S1 chronological (70% early → 30% late, same users)')
print('  Split B: Cross-semester          (all S1 → all S2, new users)')
print('=' * 80)

daily_df = pd.read_csv(DAILY_DATA)
all_results = []

for window_min in [5, 10, 30, 60]:
    print(f"\n{'='*80}")
    print(f"WINDOW: {window_min} min")
    print('='*80)

    df, feature_cols = load_window(window_min, daily_df)

    # Tag semester
    df['semester'] = 'other'
    df.loc[df['userid'].isin(S1_USERS) & (df['response_timestamp'] <= S1_END), 'semester'] = 'S1'
    df.loc[df['userid'].isin(S2_USERS) & (df['response_timestamp'] >= S2_START), 'semester'] = 'S2'

    df_s1 = df[df['semester'] == 'S1'].sort_values('response_timestamp').reset_index(drop=True)
    df_s2 = df[df['semester'] == 'S2'].sort_values('response_timestamp').reset_index(drop=True)

    print(f"  S1 records: {len(df_s1)} | users: {sorted(df_s1['userid'].unique().tolist())}")
    print(f"  S2 records: {len(df_s2)} | users: {sorted(df_s2['userid'].unique().tolist())}")

    for target_name, raw_col in [('stress','stress_raw'), ('anxiety','anxiety_raw')]:

        # ── targets: bins from S1 train portion only ──────────────────────
        s1_valid   = df_s1[raw_col].notna()
        n_s1_valid = s1_valid.sum()
        n_train    = int(n_s1_valid * 0.70)

        s1_vals_sorted = df_s1.loc[s1_valid, raw_col].reset_index(drop=True)
        train_vals = s1_vals_sorted.iloc[:n_train]

        # Build targets for S1 (using train bins)
        s1_tgt, p33, p67 = make_targets_from_train(train_vals, df_s1[raw_col].where(s1_valid))
        df_s1['target_cls'] = s1_tgt

        # Apply same bins to S2
        s2_valid = df_s2[raw_col].notna()
        bins     = [-np.inf, p33, p67, np.inf]
        s2_labels = pd.cut(df_s2.loc[s2_valid, raw_col], bins=bins, labels=[0,1,2])
        df_s2['target_cls'] = -1
        df_s2.loc[s2_valid, 'target_cls'] = s2_labels.astype(float).fillna(-1).astype(int)

        s1_dist = pd.Series(df_s1.loc[df_s1['target_cls']!=-1,'target_cls']).value_counts().sort_index().tolist()
        s2_dist = pd.Series(df_s2.loc[df_s2['target_cls']!=-1,'target_cls']).value_counts().sort_index().tolist()

        print(f"\n  [{target_name.upper()}]  bins=({p33:.0f},{p67:.0f})")
        print(f"    S1 class dist (L/M/H): {s1_dist}")
        print(f"    S2 class dist (L/M/H): {s2_dist}")

        # ── SPLIT A: Within-S1 chronological 70/30 ────────────────────────
        mask_s1  = df_s1['target_cls'] != -1
        df_s1_v  = df_s1[mask_s1]
        n_tr     = int(len(df_s1_v) * 0.70)
        X_tr     = df_s1_v[feature_cols].iloc[:n_tr]
        y_tr     = df_s1_v['target_cls'].iloc[:n_tr]
        X_te     = df_s1_v[feature_cols].iloc[n_tr:]
        y_te     = df_s1_v['target_cls'].iloc[n_tr:]

        # Date boundary info
        cut_date = df_s1_v['response_timestamp'].iloc[n_tr]
        print(f"\n    [Split A – Within-S1 TS]  cut date: {cut_date.date()}  "
              f"train={n_tr}  test={len(y_te)}")

        for mname, model in get_models().items():
            m   = clone(model)
            met = run_model(m, X_tr, y_tr, X_te, y_te)
            if met:
                print(f"      {mname:15s}  acc={met['accuracy']:.3f}  "
                      f"f1={met['f1_weighted']:.3f}  auc={met['roc_auc']:.3f}")
                all_results.append({
                    'window': window_min, 'target': target_name,
                    'split': 'within_S1_TS', 'model': mname, **{k:v for k,v in met.items() if k!='report'}
                })

        # ── SPLIT B: Cross-semester  S1 → S2 ─────────────────────────────
        mask_s2 = df_s2['target_cls'] != -1
        X_tr_full = df_s1_v[feature_cols]
        y_tr_full = df_s1_v['target_cls']
        X_te_full = df_s2.loc[mask_s2, feature_cols]
        y_te_full = df_s2.loc[mask_s2, 'target_cls']

        print(f"\n    [Split B – Cross-semester S1→S2]  "
              f"train={len(y_tr_full)} (S1)  test={len(y_te_full)} (S2, {len(df_s2['userid'].unique())} users)")

        if len(y_te_full) < 5:
            print("      ⚠ Not enough S2 data for this window — skipping")
        else:
            for mname, model in get_models().items():
                m   = clone(model)
                met = run_model(m, X_tr_full, y_tr_full, X_te_full, y_te_full)
                if met:
                    print(f"      {mname:15s}  acc={met['accuracy']:.3f}  "
                          f"f1={met['f1_weighted']:.3f}  auc={met['roc_auc']:.3f}")
                    all_results.append({
                        'window': window_min, 'target': target_name,
                        'split': 'cross_semester', 'model': mname,
                        **{k:v for k,v in met.items() if k!='report'}
                    })

            # Print detailed classification report for best model (RF, 60-min)
            if window_min == 60:
                print(f"\n    [Full Classification Report — RF, {target_name}, cross-semester]")
                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=8, min_samples_leaf=3,
                    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
                met_detail = run_model(rf, X_tr_full, y_tr_full, X_te_full, y_te_full)
                if met_detail:
                    print(met_detail['report'])

# ── Save results ──────────────────────────────────────────────────────────
rdf = pd.DataFrame(all_results)
rdf.to_csv(OUTPUT_DIR + 'results.csv', index=False)

# ── Visualizations ────────────────────────────────────────────────────────

# 1. Side-by-side: Split A vs Split B per window (best model per bar)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Semester-Aware Splits — Pure Physiological\n'
             'Split A: Within-S1 TS (same users, future)  |  '
             'Split B: Cross-semester S1→S2 (new users)',
             fontsize=13, fontweight='bold')

model_names = list(get_models().keys())
colors_a    = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
colors_b    = ['#aec7e8','#ffbb78','#98df8a','#ff9896']
x = np.arange(len(model_names))
bw = 0.35

for idx, (target_name, window_min) in enumerate(
        [('stress',60),('anxiety',60),('stress',10),('anxiety',10)]):
    ax = axes[idx//2][idx%2]
    sub_a = rdf[(rdf['target']==target_name) & (rdf['window']==window_min) &
                (rdf['split']=='within_S1_TS')]
    sub_b = rdf[(rdf['target']==target_name) & (rdf['window']==window_min) &
                (rdf['split']=='cross_semester')]

    accs_a = [sub_a[sub_a['model']==m]['accuracy'].values[0]
              if len(sub_a[sub_a['model']==m]) > 0 else 0 for m in model_names]
    accs_b = [sub_b[sub_b['model']==m]['accuracy'].values[0]
              if len(sub_b[sub_b['model']==m]) > 0 else 0 for m in model_names]

    bars_a = ax.bar(x - bw/2, accs_a, bw, label='Split A (within-S1)', color='steelblue', alpha=0.85)
    bars_b = ax.bar(x + bw/2, accs_b, bw, label='Split B (cross-sem)', color='tomato',    alpha=0.85)

    ax.axhline(1/3, color='black', linestyle='--', lw=1.2, label='Chance 33%', alpha=0.7)
    ax.set_title(f'{target_name.capitalize()} — {window_min}-min window',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0, 0.70])
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color='darkred')

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'split_A_vs_B.png', dpi=200, bbox_inches='tight')
print("\nSaved: split_A_vs_B.png")

# 2. Window comparison for cross-semester (the harder, more realistic split)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Cross-Semester Generalization (S1→S2) — Pure Physiological',
             fontsize=13, fontweight='bold')

windows = [5, 10, 30, 60]
x = np.arange(len(windows))
bw = 0.2

for ax, target_name in zip(axes, ['stress', 'anxiety']):
    for i, (mname, col) in enumerate(zip(model_names, colors_a)):
        sub = rdf[(rdf['target']==target_name) & (rdf['split']=='cross_semester') &
                  (rdf['model']==mname)]
        accs = [sub[sub['window']==w]['accuracy'].values[0]
                if len(sub[sub['window']==w]) > 0 else 0 for w in windows]
        ax.bar(x + i*bw - 1.5*bw, accs, bw, label=mname, color=col, alpha=0.85)

    ax.axhline(1/3, color='black', linestyle='--', lw=1.2, label='Chance 33%', alpha=0.7)
    ax.set_title(f'{target_name.capitalize()} — Cross-Semester', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{w} min' for w in windows])
    ax.set_ylim([0, 0.70])
    ax.set_ylabel('Accuracy')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'cross_semester_windows.png', dpi=200, bbox_inches='tight')
print("Saved: cross_semester_windows.png")

# ── Final summary ─────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")
print(f"\n{'Window':>8} {'Target':>8} {'Split':>18} {'Best Model':>18} "
      f"{'Accuracy':>10} {'F1-W':>8} {'ROC-AUC':>9}")
print('-' * 83)

for target_name in ['stress','anxiety']:
    for split in ['within_S1_TS','cross_semester']:
        for w in [5,10,30,60]:
            sub = rdf[(rdf['target']==target_name) & (rdf['split']==split) & (rdf['window']==w)]
            if sub.empty: continue
            best = sub.loc[sub['accuracy'].idxmax()]
            auc_str = f"{best['roc_auc']:.3f}" if not pd.isna(best['roc_auc']) else '  N/A'
            print(f"{w:>8} {target_name:>8} {split:>18} {best['model']:>18} "
                  f"{best['accuracy']:>10.3f} {best['f1_weighted']:>8.3f} {auc_str:>9}")

print(f"\n{'='*80}")
print("KEY COMPARISON")
print(f"{'='*80}")
for target_name in ['stress','anxiety']:
    print(f"\n  {target_name.upper()}:")
    for split, label in [('within_S1_TS','Within-S1 TS  (same users, future)'),
                         ('cross_semester','Cross-semester (new users, S2)   ')]:
        sub = rdf[(rdf['target']==target_name) & (rdf['split']==split)]
        if sub.empty: continue
        best_acc = sub['accuracy'].max()
        best_row = sub.loc[sub['accuracy'].idxmax()]
        print(f"    {label}  →  best {best_acc:.1%}  ({best_row['model']}, {best_row['window']}min)")

print(f"\n  Previous (mixed/naive) TS split result: stress≈41%  anxiety≈40%")
print(f"  Chance baseline (3-class): 33.3%")
print(f"\nAll outputs → {OUTPUT_DIR}")
