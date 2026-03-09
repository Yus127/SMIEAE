"""
Stress & Anxiety Prediction — PURE PHYSIOLOGICAL only
No lag features, no previous questionnaire answers.
Only Fitbit wearable data + context (exam period, time of day).

Branch: timeseries-windows-prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score)
from sklearn.preprocessing import label_binarize
from sklearn.base import clone
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = '/Users/YusMolina/Downloads/smieae'
WINDOW_DATA = {
    5:  f'{BASE}/data/ml_ready/enriched/ml_ready_5min_window_enriched.csv',
    10: f'{BASE}/data/ml_ready/enriched/ml_ready_10min_window_enriched.csv',
    30: f'{BASE}/data/ml_ready/enriched/ml_ready_30min_window_enriched.csv',
    60: f'{BASE}/data/ml_ready/enriched/ml_ready_60min_window_enriched.csv',
}
DAILY_DATA  = f'{BASE}/data/data_clean/csv_joined/data_with_exam_features.csv'
OUTPUT_DIR  = f'{BASE}/results/timeseries_windows/pure_physiological/'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SETS  (NO lag / questionnaire history)
# ─────────────────────────────────────────────────────────────────────────────

# Real-time HR in the N minutes before the questionnaire
WINDOW_FEATURES = [
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_std',
    'heart_rate_activity_beats per minute_min',
    'heart_rate_activity_beats per minute_max',
    'heart_rate_activity_beats per minute_median',
    'record_count',
]

# Daily Fitbit summary features (same day)
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

# Contextual (no psychological history)
CONTEXT_FEATURES = [
    'is_exam_period', 'is_easter_break', 'days_until_exam',
    'is_pre_exam_week', 'q_hour',
]

WEEKDAY_MAP = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
    'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_window(window_min, daily_df):
    df = pd.read_csv(WINDOW_DATA[window_min])
    df['response_timestamp'] = pd.to_datetime(df['response_timestamp'])
    df['q_date_only'] = pd.to_datetime(df['q_date_only'])
    df['weekday_num'] = df['q_weekday'].map(WEEKDAY_MAP).fillna(0)

    # Rename window HR cols to avoid collision with daily HR
    hr_rename = {c: f'win_{c}' for c in WINDOW_FEATURES if c in df.columns}
    df = df.rename(columns=hr_rename)
    win_cols = [f'win_{c}' for c in WINDOW_FEATURES if f'win_{c}' in df.rename(columns=hr_rename).columns]

    # Join daily Fitbit
    daily_clean = daily_df[['userid', 'unified_date'] + DAILY_FEATURES].copy()
    daily_clean['unified_date'] = pd.to_datetime(daily_clean['unified_date'])
    daily_clean = daily_clean.drop_duplicates(subset=['userid', 'unified_date'])
    df = df.merge(daily_clean,
                  left_on=['userid', 'q_date_only'],
                  right_on=['userid', 'unified_date'],
                  how='left')

    df['stress_raw']  = df['q_i_stress_sliderNeutralPos']
    df['anxiety_raw'] = df['q_i_anxiety_sliderNeutralPos']
    df = df.sort_values(['userid', 'response_timestamp']).reset_index(drop=True)
    df['window_min'] = window_min

    # Rename to get the actual win_cols that exist
    win_cols = [c for c in df.columns if c.startswith('win_')]
    feature_cols = win_cols + DAILY_FEATURES + CONTEXT_FEATURES + ['weekday_num']
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols, win_cols


def make_targets(series):
    p33, p67 = series.quantile(0.33), series.quantile(0.67)
    bins = [-np.inf, p33, p67, np.inf]
    labels = pd.cut(series, bins=bins, labels=[0, 1, 2])
    return labels.astype(float).fillna(-1).astype(int), (p33, p67)


def metrics(y_true, y_pred, y_proba):
    valid = y_true != -1
    yt, yp, ypr = y_true[valid], y_pred[valid], y_proba[valid]
    acc  = accuracy_score(yt, yp)
    f1w  = f1_score(yt, yp, average='weighted', zero_division=0)
    f1m  = f1_score(yt, yp, average='macro',    zero_division=0)
    try:
        yb  = label_binarize(yt, classes=[0,1,2])
        auc = roc_auc_score(yb, ypr, average='macro', multi_class='ovr') if yb.shape[1]==3 else np.nan
    except Exception:
        auc = np.nan
    return dict(accuracy=acc, f1_weighted=f1w, f1_macro=f1m, roc_auc=auc, n=valid.sum())


def run(model, X_tr, y_tr, X_te, y_te):
    imp = SimpleImputer(strategy='median')
    scl = StandardScaler()
    Xtr = scl.fit_transform(imp.fit_transform(X_tr))
    Xte = scl.transform(imp.transform(X_te))
    model.fit(Xtr, y_tr)
    return metrics(y_te, model.predict(Xte), model.predict_proba(Xte))


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

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("PURE PHYSIOLOGICAL: Stress & Anxiety Prediction (no prior questionnaire data)")
print("=" * 80)

daily_df = pd.read_csv(DAILY_DATA)
all_results = []

for window_min in [5, 10, 30, 60]:
    print(f"\n{'='*80}")
    print(f"WINDOW: {window_min} min")
    print('='*80)

    df, feature_cols, win_cols = load_window(window_min, daily_df)

    for target_name, raw_col in [('stress', 'stress_raw'), ('anxiety', 'anxiety_raw')]:
        valid = df[raw_col].notna()
        tgt, (p33, p67) = make_targets(df.loc[valid, raw_col])
        df['target_cls'] = -1
        df.loc[valid, 'target_cls'] = tgt

        dist = pd.Series(tgt).value_counts().sort_index().tolist()
        print(f"\n  [{target_name.upper()}]  bins=({p33:.0f},{p67:.0f})  dist={dist}")

        # --- 1. Random split ---
        mask = df['target_cls'] != -1
        X, y = df.loc[mask, feature_cols], df.loc[mask, 'target_cls']
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

        print(f"  [random split]")
        for mname, model in get_models().items():
            m = clone(model)
            met = run(m, X_tr, y_tr, X_te, y_te)
            all_results.append({'window': window_min, 'target': target_name,
                                 'split': 'random', 'model': mname, **met})
            print(f"    {mname:15s}  acc={met['accuracy']:.3f}  f1={met['f1_weighted']:.3f}  auc={met['roc_auc']:.3f}")

        # --- 2. Chronological (time-series) split ---
        df_s = df[mask].sort_values('response_timestamp')
        n_tr = int(len(df_s) * 0.70)
        X_tr = df_s[feature_cols].iloc[:n_tr]
        y_tr = df_s['target_cls'].iloc[:n_tr]
        X_te = df_s[feature_cols].iloc[n_tr:]
        y_te = df_s['target_cls'].iloc[n_tr:]

        print(f"  [timeseries split]")
        for mname, model in get_models().items():
            m = clone(model)
            met = run(m, X_tr, y_tr, X_te, y_te)
            all_results.append({'window': window_min, 'target': target_name,
                                 'split': 'timeseries', 'model': mname, **met})
            print(f"    {mname:15s}  acc={met['accuracy']:.3f}  f1={met['f1_weighted']:.3f}  auc={met['roc_auc']:.3f}")

        # --- 3. LOPO-CV ---
        lopo_true, lopo_pred = {m: ([], []) for m in get_models()}, {m: [] for m in get_models()}
        users = sorted(df['userid'].unique())
        for uid in users:
            df_tr = df[(df['userid'] != uid) & mask]
            df_te = df[(df['userid'] == uid) & mask]
            if len(df_tr) < 20 or len(df_te) < 3:
                continue
            X_tr = df_tr[feature_cols]; y_tr = df_tr['target_cls']
            X_te = df_te[feature_cols]; y_te = df_te['target_cls']
            if len(y_tr.unique()) < 2:
                continue
            imp = SimpleImputer(strategy='median'); scl = StandardScaler()
            Xtr = scl.fit_transform(imp.fit_transform(X_tr))
            Xte = scl.transform(imp.transform(X_te))
            for mname, model in get_models().items():
                m = clone(model)
                try:
                    m.fit(Xtr, y_tr)
                    lopo_true[mname][0].extend(y_te.tolist())
                    lopo_true[mname][1].extend(m.predict(Xte).tolist())
                except Exception:
                    pass

        print(f"  [LOPO-CV]")
        for mname in get_models():
            yt, yp = lopo_true[mname]
            if yt:
                acc = accuracy_score(yt, yp)
                f1w = f1_score(yt, yp, average='weighted', zero_division=0)
                f1m = f1_score(yt, yp, average='macro',    zero_division=0)
                all_results.append({'window': window_min, 'target': target_name,
                                     'split': 'lopo_cv', 'model': mname,
                                     'accuracy': acc, 'f1_weighted': f1w,
                                     'f1_macro': f1m, 'roc_auc': np.nan,
                                     'n': len(yt)})
                print(f"    {mname:15s}  acc={acc:.3f}  f1={f1w:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE  (RF, 60-min, random split, both targets)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("FEATURE IMPORTANCE (RF, 60-min window, random split)")

df60, feat60, win60 = load_window(60, daily_df)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

for ax, (tname, rcol) in zip(axes, [('Stress','stress_raw'), ('Anxiety','anxiety_raw')]):
    valid = df60[rcol].notna()
    tgt, _ = make_targets(df60.loc[valid, rcol])
    df60['target_cls'] = -1
    df60.loc[valid, 'target_cls'] = tgt
    mask = df60['target_cls'] != -1
    X, y = df60.loc[mask, feat60], df60.loc[mask, 'target_cls']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               random_state=RANDOM_STATE, stratify=y)
    imp = SimpleImputer(strategy='median'); scl = StandardScaler()
    Xtr = scl.fit_transform(imp.fit_transform(X_tr))
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                 class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(Xtr, y_tr)
    fi = pd.DataFrame({'feature': feat60, 'importance': rf.feature_importances_})
    fi = fi.sort_values('importance', ascending=True).tail(15)
    ax.barh(fi['feature'], fi['importance'], color='steelblue', alpha=0.8)
    ax.set_title(f'{tname} — Top 15 Features (60-min, pure physiological)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Mean Decrease in Impurity')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'feature_importance.png', dpi=200, bbox_inches='tight')
print("Saved: feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
rdf = pd.DataFrame(all_results)
rdf.to_csv(OUTPUT_DIR + 'results.csv', index=False)

# Grouped bar: window × model accuracy (timeseries split)
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Pure Physiological — Window Comparison (Chronological Split)',
             fontsize=14, fontweight='bold')
windows = [5, 10, 30, 60]
model_names = list(get_models().keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
x = np.arange(len(windows))
bw = 0.2

for ax, tname in zip(axes, ['stress', 'anxiety']):
    tdf = rdf[(rdf['target'] == tname) & (rdf['split'] == 'timeseries')]
    for i, (mname, col) in enumerate(zip(model_names, colors)):
        accs = [tdf[(tdf['model']==mname) & (tdf['window']==w)]['accuracy'].values
                for w in windows]
        accs = [a[0] if len(a)>0 else 0 for a in accs]
        ax.bar(x + i*bw - 1.5*bw, accs, bw, label=mname, color=col, alpha=0.85)
    ax.axhline(1/3, color='black', linestyle='--', lw=1.2, label='Chance 33%', alpha=0.7)
    ax.set_ylabel('Accuracy'); ax.set_title(tname.capitalize(), fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'{w} min' for w in windows])
    ax.set_ylim([0, 0.75]); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'window_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: window_comparison.png")

# Heatmap: model × split strategy (60-min window)
for tname in ['stress', 'anxiety']:
    tdf = rdf[(rdf['target'] == tname) & (rdf['window'] == 60)]
    try:
        hm = tdf.pivot_table(index='model', columns='split', values='accuracy')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(hm, annot=True, fmt='.3f', cmap='RdYlGn',
                    vmin=0.30, vmax=0.65, ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'Accuracy'})
        ax.set_title(f'{tname.capitalize()} — Pure Physiological, 60-min window',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + f'heatmap_{tname}_60min.png', dpi=200, bbox_inches='tight')
        print(f"Saved: heatmap_{tname}_60min.png")
    except Exception as e:
        print(f"Heatmap error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY — BEST MODEL PER WINDOW × TARGET × SPLIT")
print(f"{'='*80}")
print(f"{'Window':>8} {'Target':>8} {'Split':>15} {'Best Model':>18} {'Accuracy':>10} {'F1-W':>8}")
print('-' * 73)

for tname in ['stress', 'anxiety']:
    for split in ['random', 'timeseries', 'lopo_cv']:
        for w in [5, 10, 30, 60]:
            sub = rdf[(rdf['target']==tname) & (rdf['split']==split) & (rdf['window']==w)]
            if sub.empty: continue
            best = sub.loc[sub['accuracy'].idxmax()]
            print(f"{w:>8} {tname:>8} {split:>15} {best['model']:>18} "
                  f"{best['accuracy']:>10.3f} {best['f1_weighted']:>8.3f}")

print(f"\n{'='*80}")
best_stress  = rdf[rdf['target']=='stress']['accuracy'].max()
best_anxiety = rdf[rdf['target']=='anxiety']['accuracy'].max()
br = rdf.loc[rdf[rdf['target']=='stress']['accuracy'].idxmax()]
ba = rdf.loc[rdf[rdf['target']=='anxiety']['accuracy'].idxmax()]
print(f"[STRESS]   peak={best_stress:.1%}  ({br['model']}, {br['window']}min, {br['split']})")
print(f"[ANXIETY]  peak={best_anxiety:.1%}  ({ba['model']}, {ba['window']}min, {ba['split']})")
print(f"[CHANCE]   33.3% (3-class baseline)")
print(f"\nWith lag features (prev run):  stress≈58%  anxiety≈61%")
print(f"Without lag features (this):   stress=???  anxiety=???")
print(f"\nAll outputs → {OUTPUT_DIR}")
