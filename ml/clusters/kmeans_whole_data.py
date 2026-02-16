import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Set style - FIXED: Updated to use modern matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("USER CLUSTERING ANALYSIS - 2 CLUSTERS")
print("PHYSIOLOGICAL & CONTEXTUAL FEATURES (NO TARGET LEAKAGE)")
print("ANALYSIS OF JOINED DATASET")

# Paths
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
output_dir = "/Users/YusMolina/Downloads/smieae/results/clustering_analysis/whole_dataset"
import os
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# 1. LOAD DATA
print(f"\n1. LOADING DATA")
print("-"*80)
print(f"Path: {INPUT_PATH}")

if not os.path.exists(INPUT_PATH):
    print(f" File not found: {INPUT_PATH}")
    exit(1)

df = pd.read_csv(INPUT_PATH)
print(f" Loaded {len(df)} observations")
print(f" Columns: {df.shape[1]}")

# Show available columns
print(f"\nAvailable columns ({len(df.columns)} total):")
for col in sorted(df.columns)[:20]:  # Show first 20
    print(f"  • {col}")
if len(df.columns) > 20:
    print(f"  ... and {len(df.columns) - 20} more columns")

# Ensure userid column exists
if 'userid' not in df.columns:
    if 'user_id' in df.columns:
        df['userid'] = df['user_id']
        print("\n Created 'userid' from 'user_id'")
    else:
        df['userid'] = 0
        print("\n No user ID column found, assigning default value")

# 2. IDENTIFY AVAILABLE FEATURES
print(f"\n2. IDENTIFYING FEATURES")
print("-"*80)

# Define base features to look for
base_features = [
    'is_exam_period',
    'days_until_exam',
    'is_pre_exam_week',
    'is_easter_break',
    'daily_total_steps',
]

# Look for heart rate features (any window)
hr_patterns = [
    'heart_rate_activity_beats per minute_mean',
    'heart_rate_activity_beats per minute_std',
    'heart_rate_activity_beats per minute_min',
    'heart_rate_activity_beats per minute_max',
    'heart_rate_activity_beats per minute_median',
]

available_features = []

# Add base features that exist
for feat in base_features:
    if feat in df.columns:
        available_features.append(feat)

# Add heart rate features (check for any window prefix)
for col in df.columns:
    for pattern in hr_patterns:
        if pattern in col and 'heart_rate' in col:
            if col not in available_features:
                available_features.append(col)

print(f"\n Found {len(available_features)} features:")
for feat in available_features:
    print(f"  • {feat}")

if len(available_features) == 0:
    print("\n No suitable features found for clustering!")
    print("Please check the dataset structure.")
    exit(1)

# Get stress and anxiety targets for analysis (but NOT for clustering!)
stress_target = None
anxiety_target = None

if 'q_i_stress_sliderNeutralPos' in df.columns:
    stress_target = 'q_i_stress_sliderNeutralPos'
else:
    stress_cols = [col for col in df.columns if 'stress' in col.lower() and col.startswith('q_i')]
    stress_target = stress_cols[0] if stress_cols else None

if 'q_i_anxiety_sliderNeutralPos' in df.columns:
    anxiety_target = 'q_i_anxiety_sliderNeutralPos'
else:
    anxiety_cols = [col for col in df.columns if 'anxiety' in col.lower() and col.startswith('q_i')]
    anxiety_target = anxiety_cols[0] if anxiety_cols else None

if stress_target:
    print(f"\n Stress target: {stress_target}")
if anxiety_target:
    print(f" Anxiety target: {anxiety_target}")

# 3. PREPARE DATA
print(f"\n3. DATA PREPARATION")
print("-"*80)

# Clean data
cols_to_keep = available_features + ['userid']
if stress_target:
    cols_to_keep.append(stress_target)
if anxiety_target:
    cols_to_keep.append(anxiety_target)

df_clean = df[cols_to_keep].copy()

# Remove rows with missing userid
df_clean = df_clean.dropna(subset=['userid'])

# Impute missing features with median
for col in available_features:
    if df_clean[col].isna().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"  Imputed {col}: median={median_val:.2f}")

print(f"\n Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")

# 4. CREATE USER PROFILES (NO TARGET LEAKAGE!)
print(f"\n4. CREATING USER PROFILES")
print("-"*80)
print("  IMPORTANT: Using ONLY physiological & contextual features")
print("             NOT using stress or anxiety labels for clustering!")

# Calculate user profiles ONLY from physiological/contextual features
user_agg_dict = {}
for feat in available_features:
    user_agg_dict[feat] = 'mean'

user_profiles = df_clean.groupby('userid').agg(user_agg_dict).reset_index()

# Rename columns for clarity
col_mapping = {'userid': 'userid'}
user_feature_cols = []

for feat in available_features:
    # Create cleaner names
    if 'daily_total_steps' in feat:
        new_name = 'avg_steps'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'is_exam_period' in feat:
        new_name = 'exam_exposure'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'heart_rate' in feat and 'mean' in feat:
        new_name = 'avg_hr'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'heart_rate' in feat and 'std' in feat:
        new_name = 'avg_hr_std'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'heart_rate' in feat and 'min' in feat:
        new_name = 'avg_hr_min'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'heart_rate' in feat and 'max' in feat:
        new_name = 'avg_hr_max'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    elif 'heart_rate' in feat and 'median' in feat:
        new_name = 'avg_hr_median'
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)
    else:
        # Keep original name with slight cleanup
        new_name = feat.replace(' ', '_').lower()
        col_mapping[feat] = new_name
        user_feature_cols.append(new_name)

user_profiles = user_profiles.rename(columns=col_mapping)

# Remove duplicate feature names
original_count = len(user_feature_cols)
user_feature_cols = list(dict.fromkeys(user_feature_cols))  # Remove duplicates while preserving order
if len(user_feature_cols) < original_count:
    print(f"\n Removed {original_count - len(user_feature_cols)} duplicate feature names")

print(f"\n Created user profiles with {len(user_feature_cols)} features:")
for feat in user_feature_cols:
    print(f"  • {feat}")

# Also calculate average stress/anxiety per user for visualization (NOT for clustering!)
if stress_target:
    user_stress = df_clean.groupby('userid')[stress_target].mean().reset_index()
    user_stress.columns = ['userid', 'avg_stress']
    user_profiles = user_profiles.merge(user_stress, on='userid', how='left')
    print(f"\n Calculated average stress per user (for visualization only)")

if anxiety_target:
    user_anxiety = df_clean.groupby('userid')[anxiety_target].mean().reset_index()
    user_anxiety.columns = ['userid', 'avg_anxiety']
    user_profiles = user_profiles.merge(user_anxiety, on='userid', how='left')
    print(f" Calculated average anxiety per user (for visualization only)")

# 5. NORMALIZE USER FEATURES
print(f"\n5. NORMALIZING USER FEATURES")
print("-"*80)

X_users = user_profiles[user_feature_cols].fillna(0)
scaler_user = StandardScaler()
X_users_scaled = scaler_user.fit_transform(X_users)

print(f" Normalized {X_users_scaled.shape[0]} users with {X_users_scaled.shape[1]} features")
print(f"\nFeature statistics after normalization:")
print(f"  Mean: {X_users_scaled.mean():.3f} (should be ~0)")
print(f"  Std:  {X_users_scaled.std():.3f} (should be ~1)")

# 6. ELBOW METHOD
print(f"\n6. ELBOW METHOD ANALYSIS")
print("-"*80)

inertias = []
silhouette_scores = []
k_range = range(2, 11)

print("Testing different numbers of clusters...")
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
    cluster_labels = kmeans_temp.fit_predict(X_users_scaled)
    inertias.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_users_scaled, cluster_labels)
    silhouette_scores.append(sil_score)
    print(f"  k={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={sil_score:.3f}")

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Inertia
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=2, color='r', linestyle='--', linewidth=2, label='k=2 (chosen)')
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes[0].set_title('Elbow Method - Inertia', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Silhouette score
axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=2, color='r', linestyle='--', linewidth=2, label='k=2 (chosen)')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score Analysis', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Optimal Number of Clusters - Joined Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'elbow_method.png'), dpi=300, bbox_inches='tight')
print(f"\n Saved: elbow_method.png")
plt.close()

# 7. PERFORM CLUSTERING WITH K=2
print(f"\n7. K-MEANS CLUSTERING (k=2)")
print("-"*80)

kmeans_users = KMeans(n_clusters=2, random_state=42, n_init=20)
user_clusters = kmeans_users.fit_predict(X_users_scaled)
user_profiles['user_cluster'] = user_clusters

# Calculate clustering metrics
silhouette = silhouette_score(X_users_scaled, user_clusters)
davies_bouldin = davies_bouldin_score(X_users_scaled, user_clusters)
calinski_harabasz = calinski_harabasz_score(X_users_scaled, user_clusters)

print(f"\n Clustering completed with 2 clusters")
print(f"\nClustering Quality Metrics:")
print(f"  Silhouette Score:        {silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"  Davies-Bouldin Index:    {davies_bouldin:.4f} (lower is better)")
print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")

print(f"\nCluster Sizes:")
cluster_sizes = []
for cluster_id in range(2):
    n = (user_clusters == cluster_id).sum()
    pct = n / len(user_clusters) * 100
    cluster_sizes.append(n)
    print(f"  Cluster {cluster_id}: {n:3d} users ({pct:5.1f}%)")

# 8. CLUSTER CHARACTERISTICS
print(f"\n8. CLUSTER CHARACTERISTICS")
print("-"*80)

# Calculate cluster centroids in original (unnormalized) space
cluster_profiles = []

for cluster_id in range(2):
    cluster_mask = user_profiles['user_cluster'] == cluster_id
    cluster_data = user_profiles[cluster_mask]
    
    profile = {'cluster': cluster_id, 'n_users': cluster_mask.sum()}
    
    for feat in user_feature_cols:
        # Use unique keys for mean and std
        mean_key = f'{feat}_mean'
        std_key = f'{feat}_std'
        
        # Make sure we don't create duplicate keys
        if mean_key not in profile:
            profile[mean_key] = cluster_data[feat].mean()
        if std_key not in profile:
            profile[std_key] = cluster_data[feat].std()
    
    # Add stress/anxiety if available (for characterization only)
    if 'avg_stress' in user_profiles.columns:
        profile['avg_stress_mean'] = cluster_data['avg_stress'].mean()
        profile['avg_stress_std'] = cluster_data['avg_stress'].std()
    
    if 'avg_anxiety' in user_profiles.columns:
        profile['avg_anxiety_mean'] = cluster_data['avg_anxiety'].mean()
        profile['avg_anxiety_std'] = cluster_data['avg_anxiety'].std()
    
    cluster_profiles.append(profile)

cluster_summary_df = pd.DataFrame(cluster_profiles)

# Check for duplicate columns and print warning
if cluster_summary_df.columns.duplicated().any():
    print("\n WARNING: Duplicate columns detected in cluster summary!")
    duplicate_cols = cluster_summary_df.columns[cluster_summary_df.columns.duplicated()].tolist()
    print(f"  Duplicate columns: {duplicate_cols}")
    # Remove duplicate columns, keeping first occurrence
    cluster_summary_df = cluster_summary_df.loc[:, ~cluster_summary_df.columns.duplicated()]
    print(f"  Removed duplicates, kept first occurrence")

# Print summary
print("\nCluster Summaries (mean ± std):")
print("-" * 80)
for i in range(len(cluster_summary_df)):
    cluster_id = int(cluster_summary_df.iloc[i]['cluster'])
    n_users = int(cluster_summary_df.iloc[i]['n_users'])
    print(f"\nCluster {cluster_id} ({n_users} users):")
    
    for feat in user_feature_cols:
        mean_col = f'{feat}_mean'
        std_col = f'{feat}_std'
        
        if mean_col in cluster_summary_df.columns:
            # Get column index and use iloc for guaranteed scalar access
            mean_col_idx = cluster_summary_df.columns.get_loc(mean_col)
            # FIXED: Use .item() to convert to Python scalar
            mean_val = cluster_summary_df.iloc[i, mean_col_idx]
            if isinstance(mean_val, pd.Series):
                mean_val = mean_val.iloc[0] if len(mean_val) > 0 else 0.0
            
            if std_col in cluster_summary_df.columns:
                std_col_idx = cluster_summary_df.columns.get_loc(std_col)
                # FIXED: Use .item() to convert to Python scalar
                std_val = cluster_summary_df.iloc[i, std_col_idx]
                if isinstance(std_val, pd.Series):
                    std_val = std_val.iloc[0] if len(std_val) > 0 else 0.0
            else:
                std_val = 0.0
            
            # Handle NaN values - now with scalar values
            if pd.isna(mean_val):
                mean_val = 0.0
            if pd.isna(std_val):
                std_val = 0.0
            
            print(f"  {feat}: {mean_val:.2f} ± {std_val:.2f}")
    
    if 'avg_stress_mean' in cluster_summary_df.columns:
        stress_mean_idx = cluster_summary_df.columns.get_loc('avg_stress_mean')
        stress_mean = cluster_summary_df.iloc[i, stress_mean_idx]
        # FIXED: Handle Series case
        if isinstance(stress_mean, pd.Series):
            stress_mean = stress_mean.iloc[0] if len(stress_mean) > 0 else np.nan
        
        if not pd.isna(stress_mean):
            if 'avg_stress_std' in cluster_summary_df.columns:
                stress_std_idx = cluster_summary_df.columns.get_loc('avg_stress_std')
                stress_std = cluster_summary_df.iloc[i, stress_std_idx]
                # FIXED: Handle Series case
                if isinstance(stress_std, pd.Series):
                    stress_std = stress_std.iloc[0] if len(stress_std) > 0 else 0.0
                if pd.isna(stress_std):
                    stress_std = 0.0
            else:
                stress_std = 0.0
            print(f"  Avg Stress:  {stress_mean:.1f} ± {stress_std:.1f} (NOT used for clustering)")
    
    if 'avg_anxiety_mean' in cluster_summary_df.columns:
        anxiety_mean_idx = cluster_summary_df.columns.get_loc('avg_anxiety_mean')
        anxiety_mean = cluster_summary_df.iloc[i, anxiety_mean_idx]
        # FIXED: Handle Series case
        if isinstance(anxiety_mean, pd.Series):
            anxiety_mean = anxiety_mean.iloc[0] if len(anxiety_mean) > 0 else np.nan
        
        if not pd.isna(anxiety_mean):
            if 'avg_anxiety_std' in cluster_summary_df.columns:
                anxiety_std_idx = cluster_summary_df.columns.get_loc('avg_anxiety_std')
                anxiety_std = cluster_summary_df.iloc[i, anxiety_std_idx]
                # FIXED: Handle Series case
                if isinstance(anxiety_std, pd.Series):
                    anxiety_std = anxiety_std.iloc[0] if len(anxiety_std) > 0 else 0.0
                if pd.isna(anxiety_std):
                    anxiety_std = 0.0
            else:
                anxiety_std = 0.0
            print(f"  Avg Anxiety: {anxiety_mean:.1f} ± {anxiety_std:.1f} (NOT used for clustering)")

# Save cluster summary
cluster_summary_df.to_csv(os.path.join(output_dir, 'cluster_summary.csv'), index=False)
print(f"\n Saved: cluster_summary.csv")

# 9. PCA FOR VISUALIZATION
print(f"\n9. PCA FOR VISUALIZATION")
print("-"*80)

pca_viz = PCA(n_components=2)
X_pca_2d = pca_viz.fit_transform(X_users_scaled)

variance_explained = pca_viz.explained_variance_ratio_
print(f"\nPCA Explained Variance:")
print(f"  PC1: {variance_explained[0]*100:.1f}%")
print(f"  PC2: {variance_explained[1]*100:.1f}%")
print(f"  Total: {variance_explained.sum()*100:.1f}%")

user_profiles['pca_1'] = X_pca_2d[:, 0]
user_profiles['pca_2'] = X_pca_2d[:, 1]

# 10. VISUALIZATIONS
print(f"\n10. CREATING VISUALIZATIONS")
print("-"*80)

# Define cluster colors (using first 2 colors from original palette)
cluster_colors = ['#FF6B6B', '#4ECDC4']

# 10.1. PCA Scatter Plot (2D)
fig, ax = plt.subplots(figsize=(12, 8))

for cluster_id in range(2):
    cluster_mask = user_profiles['user_cluster'] == cluster_id
    cluster_data = user_profiles[cluster_mask]
    
    ax.scatter(cluster_data['pca_1'], cluster_data['pca_2'],
              c=cluster_colors[cluster_id], label=f'Cluster {cluster_id} (n={cluster_mask.sum()})',
              s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

# Plot centroids
centroids_pca = pca_viz.transform(kmeans_users.cluster_centers_)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
          c='black', s=300, alpha=0.8, edgecolors='white', linewidth=2,
          marker='X', label='Centroids')

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=13)
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=13)
ax.set_title(f'User Clusters - PCA Visualization\nJoined Dataset (2 clusters, {len(user_profiles)} users)',
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'clusters_pca_2d.png'), dpi=300, bbox_inches='tight')
print(" Saved: clusters_pca_2d.png")
plt.close()

# 10.2. Feature Distribution by Cluster
n_features = len(user_feature_cols)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feat in enumerate(user_feature_cols):
    ax = axes[idx]
    
    cluster_data_list = []
    cluster_labels = []
    for cluster_id in range(2):
        cluster_mask = user_profiles['user_cluster'] == cluster_id
        # FIXED: Flatten to ensure 1D array
        data = user_profiles[cluster_mask][feat].values
        if len(data.shape) > 1:
            data = data.flatten()
        cluster_data_list.append(data)
        cluster_labels.append(f'C{cluster_id}')
    
    bp = ax.boxplot(cluster_data_list, labels=cluster_labels, patch_artist=True,
                   showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel(feat.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'{feat.replace("_", " ").title()} by Cluster', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# Hide unused subplots
for idx in range(n_features, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Feature Distributions by Cluster - Joined Dataset', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'feature_distributions_by_cluster.png'), 
           dpi=300, bbox_inches='tight')
print(" Saved: feature_distributions_by_cluster.png")
plt.close()

# 10.3. Cluster Heatmap (Normalized Feature Means)
# Create heatmap data
heatmap_data = []
for cluster_id in range(2):
    cluster_mask = user_profiles['user_cluster'] == cluster_id
    cluster_means = []
    for feat in user_feature_cols:
        # FIXED: Ensure we get scalar values, not arrays
        mean_val = user_profiles[cluster_mask][feat].mean()
        # Convert to scalar if it's a Series or array
        if isinstance(mean_val, (pd.Series, np.ndarray)):
            mean_val = float(mean_val.item()) if mean_val.size == 1 else float(mean_val[0])
        cluster_means.append(float(mean_val))
    heatmap_data.append(cluster_means)

heatmap_df = pd.DataFrame(heatmap_data, 
                         columns=[f.replace('_', ' ').title() for f in user_feature_cols],
                         index=[f'Cluster {i}' for i in range(2)])

# FIXED: Normalize with safety check for zero std
heatmap_std = heatmap_df.std()
# Replace zero std with 1 to avoid division by zero
heatmap_std = heatmap_std.replace(0, 1)
heatmap_df_norm = (heatmap_df - heatmap_df.mean()) / heatmap_std

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_df_norm, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Normalized Value (z-score)'}, ax=ax, linewidths=0.5)
ax.set_title(f'Cluster Characteristics Heatmap - Joined Dataset\n(Normalized Feature Means)',
            fontsize=14, fontweight='bold')
ax.set_ylabel('Cluster', fontsize=12)
ax.set_xlabel('Feature', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'cluster_heatmap.png'), dpi=300, bbox_inches='tight')
print(" Saved: cluster_heatmap.png")
plt.close()

# 10.4. Stress/Anxiety by Cluster (if available)
if 'avg_stress' in user_profiles.columns or 'avg_anxiety' in user_profiles.columns:
    targets_available = []
    if 'avg_stress' in user_profiles.columns:
        targets_available.append(('avg_stress', 'Stress'))
    if 'avg_anxiety' in user_profiles.columns:
        targets_available.append(('avg_anxiety', 'Anxiety'))
    
    fig, axes = plt.subplots(1, len(targets_available), figsize=(8*len(targets_available), 6))
    if len(targets_available) == 1:
        axes = [axes]
    
    for idx, (target_col, target_name) in enumerate(targets_available):
        ax = axes[idx]
        
        cluster_data_list = []
        cluster_labels = []
        for cluster_id in range(2):
            cluster_mask = user_profiles['user_cluster'] == cluster_id
            # FIXED: Flatten to ensure 1D array
            data = user_profiles[cluster_mask][target_col].dropna().values
            if len(data.shape) > 1:
                data = data.flatten()
            cluster_data_list.append(data)
            cluster_labels.append(f'C{cluster_id}')
        
        bp = ax.boxplot(cluster_data_list, labels=cluster_labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], cluster_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_xlabel('Cluster', fontsize=13)
        ax.set_ylabel(f'Average {target_name}', fontsize=13)
        ax.set_title(f'{target_name} Levels by Cluster\n(NOT used for clustering - visualization only)',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Stress/Anxiety by Cluster - Joined Dataset', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'stress_anxiety_by_cluster.png'), 
               dpi=300, bbox_inches='tight')
    print(" Saved: stress_anxiety_by_cluster.png")
    plt.close()

# 10.5. Cluster Size Pie Chart
fig, ax = plt.subplots(figsize=(10, 8))

cluster_labels_pie = [f'Cluster {i}\n({cluster_sizes[i]} users)' for i in range(2)]

wedges, texts, autotexts = ax.pie(cluster_sizes, labels=cluster_labels_pie, colors=cluster_colors,
                                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax.set_title(f'Cluster Size Distribution - Joined Dataset\n{len(user_profiles)} total users',
            fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plots', 'cluster_sizes_pie.png'), dpi=300, bbox_inches='tight')
print(" Saved: cluster_sizes_pie.png")
plt.close()

# 10.6. 3D PCA Plot (if we have 3+ features)
if len(user_feature_cols) >= 3:
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_users_scaled)
    
    variance_3d = pca_3d.explained_variance_ratio_
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for cluster_id in range(2):
        cluster_mask = user_clusters == cluster_id
        ax.scatter(X_pca_3d[cluster_mask, 0],
                  X_pca_3d[cluster_mask, 1],
                  X_pca_3d[cluster_mask, 2],
                  c=cluster_colors[cluster_id],
                  label=f'Cluster {cluster_id} (n={np.sum(cluster_mask)})',
                  s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot centroids
    centroids_3d = pca_3d.transform(kmeans_users.cluster_centers_)
    ax.scatter(centroids_3d[:, 0], centroids_3d[:, 1], centroids_3d[:, 2],
              c='black', s=300, alpha=0.8, edgecolors='white', linewidth=2,
              marker='X', label='Centroids')
    
    ax.set_xlabel(f'PC1 ({variance_3d[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({variance_3d[1]*100:.1f}%)', fontsize=12)
    ax.set_zlabel(f'PC3 ({variance_3d[2]*100:.1f}%)', fontsize=12)
    ax.set_title(f'User Clusters - 3D PCA Visualization\nJoined Dataset (Total variance: {variance_3d.sum()*100:.1f}%)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plots', 'clusters_pca_3d.png'), dpi=300, bbox_inches='tight')
    print(" Saved: clusters_pca_3d.png")
    plt.close()

# 11. SAVE USER-CLUSTER MAPPING
print(f"\n11. SAVING RESULTS")
print("-"*80)

# Save user-cluster assignments
user_cluster_mapping = user_profiles[['userid', 'user_cluster']].copy()
user_cluster_mapping.to_csv(os.path.join(output_dir, 'user_cluster_assignments.csv'), index=False)
print(" Saved: user_cluster_assignments.csv")

# Save full user profiles with clusters
user_profiles.to_csv(os.path.join(output_dir, 'user_profiles_with_clusters.csv'), index=False)
print(" Saved: user_profiles_with_clusters.csv")

# 12. SUMMARY
print(f"\n12. SUMMARY")

print(f"\n CLUSTERING COMPLETE")
print(f"   Dataset: data_with_exam_features.csv (joined dataset)")
print(f"   Users:                   {len(user_profiles)}")
print(f"   Clusters:                2")
print(f"   Features Used:           {len(user_feature_cols)}")
print(f"   Silhouette Score:        {silhouette:.4f}")
print(f"   Davies-Bouldin Index:    {davies_bouldin:.4f}")
print(f"   Calinski-Harabasz Score: {calinski_harabasz:.2f}")

print(f"\n Results saved to: {output_dir}")
