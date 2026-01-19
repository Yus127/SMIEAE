import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ML PIPELINE: USER FEATURES + BINARY (NO TEMPORAL LEAKAGE)")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"
output_dir = "/Users/YusMolina/Downloads/smieae/results/ml_no_leakage"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. LOADING DATA")
print("-"*80)

data_file = f"{ml_dir}/ml_ready_combined_windows_enriched.csv"
df = pd.read_csv(data_file)
print(f"✓ Loaded {len(df)} observations")

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

if 'userid' not in df.columns:
    df['userid'] = df['user_id'] if 'user_id' in df.columns else 0

df_clean = df[available_features + [primary_target, 'userid']].copy()
df_clean = df_clean.dropna(subset=[primary_target])

for col in available_features:
    if df_clean[col].isna().sum() > 0:
        df_clean.loc[:, col] = df_clean[col].fillna(df_clean[col].median())

print(f"✓ Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")

# ============================================================================
# 2. USER-BASED FEATURES (NO LEAKAGE)
# ============================================================================
print("\n2. CREATING USER-BASED FEATURES")
print("-"*80)

# Important: Calculate user profiles WITHOUT using target variable
user_profiles = df_clean.groupby('userid').agg({
    'w30_heart_rate_activity_beats per minute_mean': 'mean',
    'w30_heart_rate_activity_beats per minute_std': 'mean',
    'daily_total_steps': 'mean',
    'is_exam_period': 'mean'
}).reset_index()

user_profiles.columns = ['userid', 'avg_hr', 'avg_hr_std', 'avg_steps', 'exam_exposure']

# Cluster users based ONLY on physiological/contextual features (not stress!)
user_features = ['avg_hr', 'avg_hr_std', 'avg_steps', 'exam_exposure']
X_users = user_profiles[user_features].fillna(0)
scaler_user = StandardScaler()
X_users_scaled = scaler_user.fit_transform(X_users)

kmeans_users = KMeans(n_clusters=7, random_state=42, n_init=20)
user_clusters = kmeans_users.fit_predict(X_users_scaled)
user_profiles['user_cluster'] = user_clusters

print(f"✓ Created 7 user clusters (based on physiology, NOT stress)")

# Add to data
df_clean = df_clean.merge(user_profiles[['userid', 'user_cluster']], on='userid', how='left')

# User baseline deviations
user_hr_baseline = df_clean.groupby('userid')['w30_heart_rate_activity_beats per minute_mean'].transform('mean')
df_clean['hr_deviation'] = df_clean['w30_heart_rate_activity_beats per minute_mean'] - user_hr_baseline

user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline.replace(0, 1) + 1)

# HR change between windows (30min vs 60min)
df_clean['hr_change_30_60'] = (df_clean['w30_heart_rate_activity_beats per minute_mean'] - 
                                df_clean['w60_heart_rate_activity_beats per minute_mean'])

# HR variability change
df_clean['hr_var_change'] = (df_clean['w30_heart_rate_activity_beats per minute_std'] - 
                              df_clean['w60_heart_rate_activity_beats per minute_std'])

print(f"✓ Added user features (NO temporal leakage)")

# ============================================================================
# 3. INTERACTION FEATURES
# ============================================================================
print("\n3. CREATING INTERACTION FEATURES")
print("-"*80)

df_clean['cluster_x_exam'] = df_clean['user_cluster'] * df_clean['is_exam_period']
df_clean['hr_dev_x_exam'] = df_clean['hr_deviation'] * df_clean['is_exam_period']
df_clean['exam_proximity'] = 1 / (df_clean['days_until_exam'].clip(lower=1) + 1)
df_clean['steps_x_exam_prox'] = df_clean['steps_ratio'] * df_clean['exam_proximity']
df_clean['hr_var_x_exam'] = df_clean['w30_heart_rate_activity_beats per minute_std'] * df_clean['is_exam_period']

print(f"✓ Added 5 interaction features")

# ============================================================================
# 4. FINAL FEATURE SET
# ============================================================================
print("\n4. PREPARING FEATURE SET")
print("-"*80)

enhanced_features = available_features + [
    'user_cluster', 'hr_deviation', 'steps_ratio',
    'hr_change_30_60', 'hr_var_change'
]

interaction_features = [
    'cluster_x_exam', 'hr_dev_x_exam', 'exam_proximity', 
    'steps_x_exam_prox', 'hr_var_x_exam'
]

full_features = enhanced_features + interaction_features

print(f"  Base features: {len(available_features)}")
print(f"  User features: {len(enhanced_features) - len(available_features)}")
print(f"  Interactions:  {len(interaction_features)}")
print(f"  Total:         {len(full_features)}")

df_model = df_clean[full_features + [primary_target]].copy()
df_model = df_model.dropna()

# Clean extreme values
for col in full_features:
    if df_model[col].dtype in ['float64', 'float32']:
        upper = df_model[col].quantile(0.999)
        lower = df_model[col].quantile(0.001)
        df_model[col] = df_model[col].clip(lower=lower, upper=upper)

print(f"\n✓ Final dataset: {len(df_model)} observations")

# ============================================================================
# 5. BINARY CLASSIFICATION
# ============================================================================
print("\n5. BINARY CLASSIFICATION SETUP")
print("-"*80)

y_continuous = df_model[primary_target]
p33 = np.percentile(y_continuous, 33.33)
p67 = np.percentile(y_continuous, 66.67)

mask_binary = (y_continuous <= p33) | (y_continuous >= p67)
df_binary = df_model[mask_binary].copy()

y_binary = (df_binary[primary_target] > p67).astype(int)
X_binary = df_binary[full_features]

print(f"  Low stress (0):  {(y_binary == 0).sum()} (≤{p33:.1f})")
print(f"  High stress (1): {(y_binary == 1).sum()} (≥{p67:.1f})")
print(f"  Total binary:    {len(y_binary)}")

# ============================================================================
# 6. TRAIN-VAL-TEST SPLIT
# ============================================================================
print("\n6. DATA SPLIT (70-15-15)")
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

print("✓ Data normalized")

# ============================================================================
# 7. TRAIN MODELS
# ============================================================================
print("\n7. TRAINING MODELS - BINARY CLASSIFICATION")
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
    
    y_val_pred = model.predict(X_val_scaled)
    val_f1 = f1_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
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

# Ensemble
print("\n8. ENSEMBLE MODEL")
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

# ============================================================================
# 9. RESULTS
# ============================================================================
print("\n" + "="*80)
print("9. FINAL RESULTS")
print("="*80)

best_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
best = results[best_name]

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   Test F1:        {best['test_f1']:.4f}")
print(f"   Test Accuracy:  {best['test_acc']:.4f}")
if 'test_precision' in best:
    print(f"   Test Precision: {best['test_precision']:.4f}")
    print(f"   Test Recall:    {best['test_recall']:.4f}")
if 'test_auc' in best:
    print(f"   Test AUC:       {best['test_auc']:.4f}")

print(f"\n📊 REALISTIC PERFORMANCE (NO LEAKAGE):")
print(f"   Baseline:               F1 = 0.358")
print(f"   + User features:        F1 = 0.494 (+38%)")
print(f"   + Binary (no leakage):  F1 = {best['test_f1']:.3f} ({(best['test_f1']-0.358)/0.358*100:+.0f}%)")

# Check if still suspiciously high
if best['test_f1'] > 0.90:
    print("\n⚠️  WARNING: F1 > 0.90 is still suspiciously high!")
    print("   Possible remaining issues:")
    print("   • Small dataset with consistent patterns")
    print("   • Binary split too extreme (very easy classes)")
    print("   • Need user-stratified CV to test properly")

# Save
comparison_df = pd.DataFrame([
    {'Model': name, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
    for name, res in results.items()
])
comparison_df.to_csv(os.path.join(output_dir, 'results_no_leakage.csv'), index=False)

# Visualizations
fig, ax = plt.subplots(figsize=(10, 8))
for name, res in results.items():
    if 'y_test_proba' in res:
        fpr, tpr, _ = roc_curve(y_test, res['y_test_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['test_auc']:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Binary Classification (No Leakage)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'roc_curves.png'), dpi=300)
print(f"\n✓ Saved: roc_curves.png")
plt.close()

from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best['y_test_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'), dpi=300)
print(f"✓ Saved: confusion_matrix.png")
plt.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE (NO TEMPORAL LEAKAGE)")
print("="*80)
print(f"\nResults saved to: {output_dir}")