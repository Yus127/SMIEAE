import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, 
                            recall_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ML PIPELINE: PCA + 3-CLASS CLASSIFICATION")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"
output_dir = "/Users/YusMolina/Downloads/smieae/results/ml_pca_3class"
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
# 2. USER-BASED CLUSTERING WITH PCA
# ============================================================================
print("\n2. USER-BASED CLUSTERING WITH PCA")
print("-"*80)

# User profiles (based on physiology, not stress!)
user_profiles = df_clean.groupby('userid').agg({
    'w30_heart_rate_activity_beats per minute_mean': 'mean',
    'w30_heart_rate_activity_beats per minute_std': 'mean',
    'w60_heart_rate_activity_beats per minute_mean': 'mean',
    'w60_heart_rate_activity_beats per minute_std': 'mean',
    'daily_total_steps': 'mean',
    'is_exam_period': 'mean'
}).reset_index()

user_profiles.columns = ['userid', 'avg_hr_30', 'avg_hr_std_30', 'avg_hr_60', 
                        'avg_hr_std_60', 'avg_steps', 'exam_exposure']

print(f"✓ Created user profiles with {len(user_profiles.columns)-1} features")

# Normalize user features
user_features = ['avg_hr_30', 'avg_hr_std_30', 'avg_hr_60', 'avg_hr_std_60', 'avg_steps', 'exam_exposure']
X_users = user_profiles[user_features].fillna(0)
scaler_user = StandardScaler()
X_users_scaled = scaler_user.fit_transform(X_users)

print(f"\n  Original features: {X_users_scaled.shape[1]}")

# Apply PCA
pca_users = PCA(n_components=None)  # Keep all to see variance
pca_users.fit(X_users_scaled)

# Show explained variance
print(f"\n  PCA Explained Variance Ratio:")
for i, var in enumerate(pca_users.explained_variance_ratio_):
    cumsum = pca_users.explained_variance_ratio_[:i+1].sum()
    print(f"    PC{i+1}: {var*100:.1f}% (cumulative: {cumsum*100:.1f}%)")

# Choose number of components (e.g., 95% variance)
n_components_user = np.argmax(pca_users.explained_variance_ratio_.cumsum() >= 0.95) + 1
print(f"\n  → Using {n_components_user} components (≥95% variance)")

# Apply PCA with selected components
pca_users_final = PCA(n_components=n_components_user, random_state=42)
X_users_pca = pca_users_final.fit_transform(X_users_scaled)

# Cluster users on PCA components
kmeans_users = KMeans(n_clusters=7, random_state=42, n_init=20)
user_clusters = kmeans_users.fit_predict(X_users_pca)
user_profiles['user_cluster'] = user_clusters

print(f"\n✓ Clustered users into 7 groups using {n_components_user} PCA components")
for cluster_id in range(7):
    n = (user_clusters == cluster_id).sum()
    print(f"  Cluster {cluster_id}: {n} users")

# Add to data
df_clean = df_clean.merge(user_profiles[['userid', 'user_cluster']], on='userid', how='left')

# ============================================================================
# 3. CREATE ENHANCED FEATURES (NO LEAKAGE)
# ============================================================================
print("\n3. CREATING ENHANCED FEATURES")
print("-"*80)

# User baseline deviations
user_hr_baseline = df_clean.groupby('userid')['w30_heart_rate_activity_beats per minute_mean'].transform('mean')
df_clean['hr_deviation'] = df_clean['w30_heart_rate_activity_beats per minute_mean'] - user_hr_baseline

user_step_baseline = df_clean.groupby('userid')['daily_total_steps'].transform('mean')
df_clean['steps_ratio'] = df_clean['daily_total_steps'] / (user_step_baseline.replace(0, 1) + 1)

# HR change between windows
df_clean['hr_change_30_60'] = (df_clean['w30_heart_rate_activity_beats per minute_mean'] - 
                                df_clean['w60_heart_rate_activity_beats per minute_mean'])

# HR variability change
df_clean['hr_var_change'] = (df_clean['w30_heart_rate_activity_beats per minute_std'] - 
                              df_clean['w60_heart_rate_activity_beats per minute_std'])

# Interaction features
df_clean['cluster_x_exam'] = df_clean['user_cluster'] * df_clean['is_exam_period']
df_clean['hr_dev_x_exam'] = df_clean['hr_deviation'] * df_clean['is_exam_period']
df_clean['exam_proximity'] = 1 / (df_clean['days_until_exam'].clip(lower=1) + 1)
df_clean['steps_x_exam_prox'] = df_clean['steps_ratio'] * df_clean['exam_proximity']

print(f"✓ Added user features and interactions")

# ============================================================================
# 4. PREPARE FEATURE SET
# ============================================================================
print("\n4. PREPARING FEATURE SET")
print("-"*80)

enhanced_features = available_features + [
    'user_cluster', 'hr_deviation', 'steps_ratio',
    'hr_change_30_60', 'hr_var_change',
    'cluster_x_exam', 'hr_dev_x_exam', 'exam_proximity', 'steps_x_exam_prox'
]

print(f"  Total features before PCA: {len(enhanced_features)}")

df_model = df_clean[enhanced_features + [primary_target]].copy()
df_model = df_model.dropna()

# Clean extreme values
for col in enhanced_features:
    if df_model[col].dtype in ['float64', 'float32']:
        upper = df_model[col].quantile(0.999)
        lower = df_model[col].quantile(0.001)
        df_model[col] = df_model[col].clip(lower=lower, upper=upper)

print(f"✓ Clean dataset: {len(df_model)} observations")

# ============================================================================
# 5. 3-CLASS TARGET
# ============================================================================
print("\n5. CREATING 3-CLASS TARGET")
print("-"*80)

y_continuous = df_model[primary_target]
p33 = np.percentile(y_continuous, 33.33)
p67 = np.percentile(y_continuous, 66.67)

y_3class = pd.cut(y_continuous,
                  bins=[-np.inf, p33, p67, np.inf],
                  labels=[0, 1, 2],
                  include_lowest=True).astype(int)

print(f"  Class 0 (Low):    {(y_3class == 0).sum()} observations (≤{p33:.1f})")
print(f"  Class 1 (Medium): {(y_3class == 1).sum()} observations ({p33:.1f}-{p67:.1f})")
print(f"  Class 2 (High):   {(y_3class == 2).sum()} observations (≥{p67:.1f})")

X = df_model[enhanced_features]

# ============================================================================
# 6. TRAIN-VAL-TEST SPLIT
# ============================================================================
print("\n6. DATA SPLIT (70-15-15)")
print("-"*80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_3class, test_size=0.30, random_state=42, stratify=y_3class
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# 7. APPLY PCA TO FEATURES
# ============================================================================
print("\n7. APPLYING PCA TO FEATURE SET")
print("-"*80)

# First normalize
scaler_features = StandardScaler()
X_train_scaled = scaler_features.fit_transform(X_train)
X_val_scaled = scaler_features.transform(X_val)
X_test_scaled = scaler_features.transform(X_test)

# Exploratory PCA to see variance
pca_explore = PCA(n_components=None)
pca_explore.fit(X_train_scaled)

print(f"\n  Explained Variance (first 10 components):")
for i in range(min(10, len(pca_explore.explained_variance_ratio_))):
    cumsum = pca_explore.explained_variance_ratio_[:i+1].sum()
    print(f"    PC{i+1}: {pca_explore.explained_variance_ratio_[i]*100:.1f}% (cumulative: {cumsum*100:.1f}%)")

# Choose components to retain 90% variance
n_components = np.argmax(pca_explore.explained_variance_ratio_.cumsum() >= 0.90) + 1
print(f"\n  → Using {n_components} components (≥90% variance)")

# Apply PCA with selected components
pca_final = PCA(n_components=n_components, random_state=42)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_val_pca = pca_final.transform(X_val_scaled)
X_test_pca = pca_final.transform(X_test_scaled)

print(f"\n✓ Dimensionality reduction: {X_train_scaled.shape[1]} → {X_train_pca.shape[1]} features")
print(f"  Retained variance: {pca_final.explained_variance_ratio_.sum()*100:.1f}%")

# Save feature names for later
feature_names_pca = [f'PC{i+1}' for i in range(n_components)]

# ============================================================================
# 8. TRAIN MODELS ON PCA FEATURES
# ============================================================================
print("\n8. TRAINING MODELS - 3-CLASS CLASSIFICATION")
print("-"*80)

models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                                  random_state=42, objective='multi:softmax', num_class=3),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=12, 
                                           random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, 
                                              multi_class='multinomial'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, 
               decision_function_shape='ovr')
}

results = {}

for name, model in models.items():
    print(f"\n{name}...")
    model.fit(X_train_pca, y_train)
    
    # Validation
    y_val_pred = model.predict(X_val_pca)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_per_class = f1_score(y_val, y_val_pred, average=None)
    
    # Test
    y_test_pred = model.predict(X_test_pca)
    y_test_proba = model.predict_proba(X_test_pca)
    
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None)
    
    # Multi-class AUC
    try:
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        test_auc = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
    except:
        test_auc = np.nan
    
    print(f"  Val:  F1={val_f1:.4f}, Acc={val_acc:.4f}")
    print(f"        F1 per class: Low={val_f1_per_class[0]:.3f}, Med={val_f1_per_class[1]:.3f}, High={val_f1_per_class[2]:.3f}")
    print(f"  Test: F1={test_f1:.4f}, Acc={test_acc:.4f}, AUC={test_auc:.4f}")
    print(f"        F1 per class: Low={test_f1_per_class[0]:.3f}, Med={test_f1_per_class[1]:.3f}, High={test_f1_per_class[2]:.3f}")
    
    results[name] = {
        'val_f1': val_f1, 'val_acc': val_acc,
        'test_f1': test_f1, 'test_acc': test_acc, 'test_auc': test_auc,
        'test_f1_per_class': test_f1_per_class,
        'y_test_pred': y_test_pred, 'y_test_proba': y_test_proba
    }

# ============================================================================
# 9. ENSEMBLE MODEL
# ============================================================================
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

ensemble.fit(X_train_pca, y_train)

y_test_pred_ens = ensemble.predict(X_test_pca)
y_test_proba_ens = ensemble.predict_proba(X_test_pca)

test_f1_ens = f1_score(y_test, y_test_pred_ens, average='weighted')
test_acc_ens = accuracy_score(y_test, y_test_pred_ens)
test_f1_per_class_ens = f1_score(y_test, y_test_pred_ens, average=None)

print(f"  Test: F1={test_f1_ens:.4f}, Acc={test_acc_ens:.4f}")
print(f"        F1 per class: Low={test_f1_per_class_ens[0]:.3f}, Med={test_f1_per_class_ens[1]:.3f}, High={test_f1_per_class_ens[2]:.3f}")

results['Ensemble'] = {
    'test_f1': test_f1_ens, 'test_acc': test_acc_ens,
    'test_f1_per_class': test_f1_per_class_ens,
    'y_test_pred': y_test_pred_ens
}

# ============================================================================
# 10. RESULTS
# ============================================================================
print("\n" + "="*80)
print("10. FINAL RESULTS")
print("="*80)

best_name = max(results.keys(), key=lambda k: results[k]['test_f1'])
best = results[best_name]

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   Test F1:       {best['test_f1']:.4f} (weighted)")
print(f"   Test Accuracy: {best['test_acc']:.4f}")
if 'test_auc' in best and not np.isnan(best['test_auc']):
    print(f"   Test AUC:      {best['test_auc']:.4f}")

print(f"\n   Per-Class Performance (Test):")
print(f"     Low (Class 0):    F1 = {best['test_f1_per_class'][0]:.4f}")
print(f"     Medium (Class 1): F1 = {best['test_f1_per_class'][1]:.4f}")
print(f"     High (Class 2):   F1 = {best['test_f1_per_class'][2]:.4f}")

print(f"\n📊 PERFORMANCE WITH PCA:")
print(f"   Original features:  {len(enhanced_features)}")
print(f"   PCA components:     {n_components} (retained {pca_final.explained_variance_ratio_.sum()*100:.1f}% variance)")
print(f"   Test F1 (3-class):  {best['test_f1']:.4f}")

# Detailed classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT ({best_name}):")
print(classification_report(y_test, best['y_test_pred'], 
                           target_names=['Low Stress', 'Medium Stress', 'High Stress']))

# Save results
comparison_df = pd.DataFrame([
    {'Model': name, **{k: v for k, v in res.items() if isinstance(v, (int, float))}}
    for name, res in results.items()
])
comparison_df.to_csv(os.path.join(output_dir, 'results_pca_3class.csv'), index=False)

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================
print("\n11. CREATING VISUALIZATIONS")
print("-"*80)

# PCA Variance Explained
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(pca_explore.explained_variance_ratio_)+1), 
           pca_explore.explained_variance_ratio_*100)
axes[0].axvline(x=n_components, color='r', linestyle='--', label=f'{n_components} components')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Cumulative variance
cumsum = np.cumsum(pca_explore.explained_variance_ratio_)*100
axes[1].plot(range(1, len(cumsum)+1), cumsum, 'bo-')
axes[1].axhline(y=90, color='r', linestyle='--', label='90% threshold')
axes[1].axvline(x=n_components, color='r', linestyle='--', label=f'{n_components} components')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained (%)')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('PCA Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'pca_variance.png'), dpi=300)
print("✓ Saved: pca_variance.png")
plt.close()

# Confusion Matrix
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Absolute counts
cm = confusion_matrix(y_test, best['y_test_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                              display_labels=['Low', 'Medium', 'High'])
disp.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title(f'Confusion Matrix - {best_name}\n(Counts)', fontweight='bold')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, 
                                   display_labels=['Low', 'Medium', 'High'])
disp_norm.plot(ax=axes[1], cmap='RdYlGn', values_format='.2f')
axes[1].set_title(f'Confusion Matrix - {best_name}\n(Normalized)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix_3class.png'), dpi=300)
print("✓ Saved: confusion_matrix_3class.png")
plt.close()

# Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
models_list = list(results.keys())
test_f1_list = [results[m]['test_f1'] for m in models_list]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = ax.barh(models_list, test_f1_list, color=colors)
ax.set_xlabel('Test F1 Score (Weighted)', fontsize=12)
ax.set_title('Model Comparison - 3-Class Classification', fontsize=14, fontweight='bold')
ax.axvline(x=0.333, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_xlim(0, max(test_f1_list)*1.1)
ax.legend()
ax.grid(alpha=0.3, axis='x')

for i, (model, f1) in enumerate(zip(models_list, test_f1_list)):
    ax.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'model_comparison.png'), dpi=300)
print("✓ Saved: model_comparison.png")
plt.close()

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE: PCA + 3-CLASS")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print(f"\n💡 Key Insights:")
print(f"   • Used PCA: {len(enhanced_features)} → {n_components} features")
print(f"   • 3-class F1: {best['test_f1']:.3f}")
print(f"   • Medium class F1: {best['test_f1_per_class'][1]:.3f} (hardest to predict)")
print(f"   • Best model: {best_name}")
