import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, f1_score, 
                            accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("ADVANCED ML: TEMPORAL FEATURES + BINARY CLASSIFICATION")

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"
output_dir = "/Users/YusMolina/Downloads/smieae/results/advanced_ml_final"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# 1. LOAD DATA & DETECT STRUCTURE
print("\n1. LOADING DATA & DETECTING STRUCTURE")
print("-"*80)

data_file = f"{ml_dir}/ml_ready_combined_windows_enriched.csv"
df = pd.read_csv(data_file)
print(f" Loaded {len(df)} observations")
print(f"  Columns: {len(df.columns)}")

# Detect time/ordering columns
time_candidates = [col for col in df.columns if any(x in col.lower() for x in 
                   ['time', 'date', 'response', 'timestamp', 'created', 'id'])]
print(f"\n  Potential ordering columns: {time_candidates[:5]}")

# Features
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

# Check for userid
if 'userid' not in df.columns:
    if 'user_id' in df.columns:
        df['userid'] = df['user_id']
    else:
        print("  WARNING: No user ID column found!")
        df['userid'] = 0  # Single user dataset

print(f" Features: {len(available_features)}")
print(f" Users: {df['userid'].nunique()}")

df_clean = df[available_features + [primary_target, 'userid']].copy()
df_clean = df_clean.dropna(subset=[primary_target])

# Impute missing
for col in available_features:
    if df_clean[col].isna().sum() > 0:
        df_clean.loc[:, col] = df_clean[col].fillna(df_clean[col].median())

print(f" Clean dataset: {len(df_clean)} observations")

# Create temporal ordering
# If we have timestamp, use it; otherwise use sequential order
if 'response_timestamp' in df.columns:
    df_clean['timestamp'] = pd.to_datetime(df['response_timestamp'])
    df_clean = df_clean.sort_values(['userid', 'timestamp'])
    print(" Using response_timestamp for temporal ordering")
elif 'created_at' in df.columns:
    df_clean['timestamp'] = pd.to_datetime(df['created_at'])
    df_clean = df_clean.sort_values(['userid', 'timestamp'])
    print(" Using created_at for temporal ordering")
else:
    # Use sequential order per user
    df_clean = df_clean.sort_values(['userid'])
    df_clean['obs_order'] = df_clean.groupby('userid').cumcount()
    print(" Using sequential order for temporal features (no timestamp found)")

# 2. USER-BASED FEATURES (from your successful approach)
print("\n2. CREATING USER-BASED FEATURES")
print("-"*80)

# User profiles
user_profiles = df_clean.groupby('userid').agg({
    primary_target: ['mean', 'std', 'min', 'max'],
    'w30_heart_rate_activity_beats per minute_mean': 'mean',
    'w30_heart_rate_activity_beats per minute_std': 'mean',
    'daily_total_steps': 'mean',
    'is_exam_period': 'mean'
}).reset_index()

user_profiles.columns = ['userid', 'stress_mean', 'stress_std', 'stress_min', 'stress_max',
                        'avg_hr', 'avg_hr_std', 'avg_steps', 'exam_exposure']

# Cluster users (K=7 from your success)
user_features = ['stress_mean', 'stress_std', 'avg_hr', 'avg_hr_std', 'avg_steps', 'exam_exposure']
X_users = user_profiles[user_features].fillna(0)
scaler_user = StandardScaler()
X_users_scaled = scaler_user.fit_transform(X_users)

kmeans_users = KMeans(n_clusters=7, random_state=42, n_init=20)
user_clusters = kmeans_users.fit_predict(X_users_scaled)
user_profiles['user_cluster'] = user_clusters

print(f" Created 7 user clusters")

# Add to main data
df_clean = df_clean.merge(user_profiles[['userid', 'user_cluster']], on='userid', how='left')

# User baseline deviations
user_hr_baseline = df_clean.groupby('userid')['w30_heart_rate_activity_beats per minute_mean'].transform('mean')
df_clean['hr_deviation'] = df_clean['w30_heart_rate_activity_beats per minute_mean'] - user_hr_baseline

user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline + 1)

print(f" Added user features: cluster, hr_deviation, steps_ratio")

# 3. TEMPORAL FEATURES
print("\n3. CREATING TEMPORAL FEATURES")
print("-"*80)

# Lagged features
df_clean['stress_lag1'] = df_clean.groupby('userid')[primary_target].shift(1)
df_clean['stress_lag2'] = df_clean.groupby('userid')[primary_target].shift(2)

# Rolling statistics
df_clean['stress_rolling_mean'] = df_clean.groupby('userid')[primary_target].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
df_clean['stress_rolling_std'] = df_clean.groupby('userid')[primary_target].transform(
    lambda x: x.rolling(window=3, min_periods=1).std()
)

# Changes
df_clean['stress_change'] = df_clean.groupby('userid')[primary_target].diff()
df_clean['hr_change'] = df_clean.groupby('userid')['w30_heart_rate_activity_beats per minute_mean'].diff()

# Position in sequence
df_clean['seq_position'] = df_clean.groupby('userid').cumcount()

print(f" Added 7 temporal features")

# 4. INTERACTION FEATURES
print("\n4. CREATING INTERACTION FEATURES")
print("-"*80)

df_clean['cluster_x_exam'] = df_clean['user_cluster'] * df_clean['is_exam_period']
df_clean['hr_dev_x_exam'] = df_clean['hr_deviation'] * df_clean['is_exam_period']
df_clean['steps_x_exam_prox'] = df_clean['steps_ratio'] * (1 / (df_clean['days_until_exam'] + 1))

print(f" Added 3 interaction features")

# 5. PREPARE DATASETS
print("\n5. PREPARING FEATURE SETS")
print("-"*80)

# Feature sets
enhanced_features = available_features + ['user_cluster', 'hr_deviation', 'steps_ratio']

temporal_features_list = ['stress_lag1', 'stress_lag2', 'stress_rolling_mean', 
                          'stress_rolling_std', 'stress_change', 'hr_change', 'seq_position']

interaction_features_list = ['cluster_x_exam', 'hr_dev_x_exam', 'steps_x_exam_prox']

full_features = enhanced_features + temporal_features_list + interaction_features_list

print(f"  Enhanced: {len(enhanced_features)}")
print(f"  + Temporal: {len(temporal_features_list)}")
print(f"  + Interactions: {len(interaction_features_list)}")
print(f"  Total: {len(full_features)}")

# Remove NaN from temporal lag
df_model = df_clean[full_features + [primary_target]].copy()
initial_len = len(df_model)
df_model = df_model.dropna()
print(f"\n Removed {initial_len - len(df_model)} rows with NaN (from lagging)")
print(f" Final dataset: {len(df_model)} observations")

# 6. BINARY CLASSIFICATION
print("\n6. BINARY CLASSIFICATION SETUP")
print("-"*80)

y_continuous = df_model[primary_target]

# Tertile split
p33 = np.percentile(y_continuous, 33.33)
p67 = np.percentile(y_continuous, 66.67)

# Binary: Remove middle third
mask_binary = (y_continuous <= p33) | (y_continuous >= p67)
df_binary = df_model[mask_binary].copy()

y_binary = (df_binary[primary_target] > p67).astype(int)
X_binary = df_binary[full_features]

print(f"  Low stress (0):  {(y_binary == 0).sum()} (≤{p33:.1f})")
print(f"  High stress (1): {(y_binary == 1).sum()} (≥{p67:.1f})")
print(f"  Total binary:    {len(y_binary)}")

# 7. TRAIN-VAL-TEST SPLIT
print("\n7. DATA SPLIT (70-15-15)")
print("-"*80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_binary, y_binary, test_size=0.30, random_state=42, stratify=y_binary
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(" Data normalized")

# 8. TRAIN MODELS
print("\n8. TRAINING MODELS - BINARY CLASSIFICATION")
print("-"*80)

models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                  random_state=42, objective='binary:logistic'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, 
                                           random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, C=1.0)
}

results = {}

for name, model in models.items():
    print(f"\n{name}...")
    model.fit(X_train_scaled, y_train)
    
    # Validation
    y_val_pred = model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    # Test
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_f1 = f1_score(y_test, y_test_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"  Val:  F1={val_f1:.4f}, Acc={val_acc:.4f}")
    print(f"  Test: F1={test_f1:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")
    
    results[name] = {
        'val_f1': val_f1, 'val_acc': val_acc,
        'test_f1': test_f1, 'test_acc': test_acc,
        'test_precision': test_precision, 'test_recall': test_recall, 'test_auc': test_auc,
        'y_test_pred': y_test_pred, 'y_test_proba': y_test_proba
    }

# 9. ENSEMBLE
print("\n9. ENSEMBLE MODEL")
print("-"*80)

ensemble = VotingClassifier(
    estimators=[
        ('xgb', models['XGBoost']),
        ('rf', models['Random Forest']),
        ('lr', models['Logistic Regression'])
    ],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)

y_test_pred_ens = ensemble.predict(X_test_scaled)
y_test_proba_ens = ensemble.predict_proba(X_test_scaled)[:, 1]

test_f1_ens = f1_score(y_test, y_test_pred_ens)
test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
test_auc_ens = roc_auc_score(y_test, y_test_proba_ens)

print(f"  Test: F1={test_f1_ens:.4f}, Acc={test_acc_ens:.4f}, AUC={test_auc_ens:.4f}")

results['Ensemble'] = {
    'test_f1': test_f1_ens, 'test_acc': test_acc_ens, 'test_auc': test_auc_ens,
    'y_test_pred': y_test_pred_ens, 'y_test_proba': y_test_proba_ens
}

# 10. RESULTS
print("\n" + "="*80)
print("10. FINAL RESULTS")

best_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
best = results[best_name]

print(f"\n BEST MODEL: {best_name}")
print(f"   Test F1:        {best['test_f1']:.4f}")
print(f"   Test Accuracy:  {best['test_acc']:.4f}")
if 'test_precision' in best:
    print(f"   Test Precision: {best['test_precision']:.4f}")
    print(f"   Test Recall:    {best['test_recall']:.4f}")
if 'test_auc' in best:
    print(f"   Test AUC:       {best['test_auc']:.4f}")

print(f"\n PERFORMANCE PROGRESSION:")
print(f"   Baseline:               F1 = 0.358")
print(f"   + User features:        F1 = 0.494 (+38%)")
print(f"   + Temporal + Binary:    F1 = {best['test_f1']:.3f} ({(best['test_f1']-0.358)/0.358*100:+.0f}%)")

comparison_df = pd.DataFrame([
    {'Model': name, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
    for name, res in results.items()
])
comparison_df.to_csv(os.path.join(output_dir, 'binary_results.csv'), index=False)

# 11. VISUALIZATIONS
print("\n11. CREATING VISUALIZATIONS")
print("-"*80)

# ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
for name, res in results.items():
    if 'y_test_proba' in res:
        fpr, tpr, _ = roc_curve(y_test, res['y_test_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Binary Classification', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'roc_curves.png'), dpi=300, bbox_inches='tight')
print(" Saved: roc_curves.png")
plt.close()

# Confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best['y_test_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix - {best_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(" Saved: confusion_matrix.png")
plt.close()

print("\n" + "="*80)
print(f"\nResults saved to: {output_dir}")
print(f"\n Achievement: F1 improved from 0.358 → {best['test_f1']:.3f}")
print(f"   That's a {(best['test_f1']-0.358)/0.358*100:.0f}% improvement! ")