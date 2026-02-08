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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("USER CLUSTERING ANALYSIS - 5 CLUSTERS")
print("PHYSIOLOGICAL & CONTEXTUAL FEATURES (NO TARGET LEAKAGE)")
print("SEPARATE ANALYSIS FOR 5MIN AND 10MIN WINDOWS")
print("="*80)

# Paths
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/results/clustering_analysis"
import os
os.makedirs(output_dir, exist_ok=True)

# Define files to process
files = [
    "enriched/ml_ready_5min_window_enriched.csv",
    "enriched/ml_ready_10min_window_enriched.csv"
]

# Store results
all_clustering_results = {}

# Process each window
for file_idx, file in enumerate(files):
    window_type = "5min" if "5min" in file else "10min"
    window_prefix = "w5" if "5min" in file else "w10"
    
    print("\n" + "="*80)
    print(f"PROCESSING: {window_type.upper()} WINDOW")
    print("="*80)
    
    # Create window-specific output directory
    window_output_dir = os.path.join(output_dir, f"{window_type}_window")
    os.makedirs(window_output_dir, exist_ok=True)
    os.makedirs(os.path.join(window_output_dir, "plots"), exist_ok=True)
    
    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print(f"\n1. LOADING DATA ({window_type})")
    print("-"*80)
    
    data_file = os.path.join(ml_dir, file)
    if not os.path.exists(data_file):
        print(f"✗ File not found: {data_file}")
        continue
    
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} observations")
    
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
    print(f"✓ Using {len(available_features)} features")
    
    # Ensure userid column exists
    if 'userid' not in df.columns:
        df['userid'] = df['user_id'] if 'user_id' in df.columns else 0
    
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
    
    # ========================================================================
    # 2. PREPARE DATA
    # ========================================================================
    print(f"\n2. DATA PREPARATION ({window_type})")
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
    
    print(f"\n✓ Clean dataset: {len(df_clean)} observations from {df_clean['userid'].nunique()} users")
    
    # ========================================================================
    # 3. CREATE USER PROFILES (NO TARGET LEAKAGE!)
    # ========================================================================
    print(f"\n3. CREATING USER PROFILES ({window_type})")
    print("-"*80)
    print("⚠️  IMPORTANT: Using ONLY physiological & contextual features")
    print("             NOT using stress or anxiety labels for clustering!")
    
    # Calculate user profiles ONLY from physiological/contextual features
    user_agg_dict = {
        'daily_total_steps': 'mean',
        'is_exam_period': 'mean'
    }
    
    # Add HR features that exist
    for feat in hr_features:
        if feat in df_clean.columns:
            user_agg_dict[feat] = 'mean'
    
    user_profiles = df_clean.groupby('userid').agg(user_agg_dict).reset_index()
    
    # Rename columns for clarity
    col_mapping = {'userid': 'userid'}
    user_feature_cols = []
    
    if 'daily_total_steps' in user_agg_dict:
        col_mapping['daily_total_steps'] = 'avg_steps'
        user_feature_cols.append('avg_steps')
    
    if 'is_exam_period' in user_agg_dict:
        col_mapping['is_exam_period'] = 'exam_exposure'
        user_feature_cols.append('exam_exposure')
    
    # Add HR features
    hr_mean_feat = None
    hr_std_feat = None
    for feat in hr_features:
        if feat in user_profiles.columns:
            if 'mean' in feat:
                col_mapping[feat] = 'avg_hr'
                user_feature_cols.append('avg_hr')
                hr_mean_feat = 'avg_hr'
            elif 'std' in feat:
                col_mapping[feat] = 'avg_hr_std'
                user_feature_cols.append('avg_hr_std')
                hr_std_feat = 'avg_hr_std'
            elif 'min' in feat:
                col_mapping[feat] = 'avg_hr_min'
                user_feature_cols.append('avg_hr_min')
            elif 'max' in feat:
                col_mapping[feat] = 'avg_hr_max'
                user_feature_cols.append('avg_hr_max')
            elif 'median' in feat:
                col_mapping[feat] = 'avg_hr_median'
                user_feature_cols.append('avg_hr_median')
    
    user_profiles = user_profiles.rename(columns=col_mapping)
    
    print(f"\n✓ Created user profiles with {len(user_feature_cols)} features:")
    for feat in user_feature_cols:
        print(f"  • {feat}")
    
    # Also calculate average stress/anxiety per user for visualization (NOT for clustering!)
    if stress_target:
        user_stress = df_clean.groupby('userid')[stress_target].mean().reset_index()
        user_stress.columns = ['userid', 'avg_stress']
        user_profiles = user_profiles.merge(user_stress, on='userid', how='left')
        print(f"\n✓ Calculated average stress per user (for visualization only)")
    
    if anxiety_target:
        user_anxiety = df_clean.groupby('userid')[anxiety_target].mean().reset_index()
        user_anxiety.columns = ['userid', 'avg_anxiety']
        user_profiles = user_profiles.merge(user_anxiety, on='userid', how='left')
        print(f"✓ Calculated average anxiety per user (for visualization only)")
    
    # ========================================================================
    # 4. NORMALIZE USER FEATURES
    # ========================================================================
    print(f"\n4. NORMALIZING USER FEATURES ({window_type})")
    print("-"*80)
    
    X_users = user_profiles[user_feature_cols].fillna(0)
    scaler_user = StandardScaler()
    X_users_scaled = scaler_user.fit_transform(X_users)
    
    print(f"✓ Normalized {X_users_scaled.shape[0]} users with {X_users_scaled.shape[1]} features")
    print(f"\nFeature statistics after normalization:")
    print(f"  Mean: {X_users_scaled.mean():.3f} (should be ~0)")
    print(f"  Std:  {X_users_scaled.std():.3f} (should be ~1)")
    
    # ========================================================================
    # 5. ELBOW METHOD (Optional - show why 3 clusters)
    # ========================================================================
    print(f"\n5. ELBOW METHOD ANALYSIS ({window_type})")
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
    
    # Inertia (within-cluster sum of squares)
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=5, color='r', linestyle='--', linewidth=2, label='k=5 (chosen)')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    axes[0].set_title('Elbow Method - Inertia', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette score
    axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=5, color='r', linestyle='--', linewidth=2, label='k=5 (chosen)')
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score Analysis', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Optimal Number of Clusters - {window_type} window', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'plots', 'elbow_method.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: elbow_method.png")
    plt.close()
    
    # ========================================================================
    # 6. PERFORM CLUSTERING WITH K=5
    # ========================================================================
    print(f"\n6. K-MEANS CLUSTERING (k=5) ({window_type})")
    print("-"*80)
    
    kmeans_users = KMeans(n_clusters=5, random_state=42, n_init=20)
    user_clusters = kmeans_users.fit_predict(X_users_scaled)
    user_profiles['user_cluster'] = user_clusters
    
    # Calculate clustering metrics
    silhouette = silhouette_score(X_users_scaled, user_clusters)
    davies_bouldin = davies_bouldin_score(X_users_scaled, user_clusters)
    calinski_harabasz = calinski_harabasz_score(X_users_scaled, user_clusters)
    
    print(f"\n✓ Clustering completed with 5 clusters")
    print(f"\nClustering Quality Metrics:")
    print(f"  Silhouette Score:        {silhouette:.4f} (higher is better, range: -1 to 1)")
    print(f"  Davies-Bouldin Index:    {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")
    
    print(f"\nCluster Sizes:")
    for cluster_id in range(5):
        n = (user_clusters == cluster_id).sum()
        pct = n / len(user_clusters) * 100
        print(f"  Cluster {cluster_id}: {n:3d} users ({pct:5.1f}%)")
    
    # ========================================================================
    # 7. CLUSTER CHARACTERISTICS
    # ========================================================================
    print(f"\n7. CLUSTER CHARACTERISTICS ({window_type})")
    print("-"*80)
    
    # Calculate cluster centroids in original (unnormalized) space
    cluster_profiles = []
    
    for cluster_id in range(5):
        cluster_mask = user_profiles['user_cluster'] == cluster_id
        cluster_data = user_profiles[cluster_mask]
        
        profile = {'cluster': cluster_id, 'n_users': cluster_mask.sum()}
        
        for feat in user_feature_cols:
            profile[f'{feat}_mean'] = cluster_data[feat].mean()
            profile[f'{feat}_std'] = cluster_data[feat].std()
        
        # Add stress/anxiety if available (for characterization only)
        if 'avg_stress' in user_profiles.columns:
            profile['avg_stress_mean'] = cluster_data['avg_stress'].mean()
            profile['avg_stress_std'] = cluster_data['avg_stress'].std()
        
        if 'avg_anxiety' in user_profiles.columns:
            profile['avg_anxiety_mean'] = cluster_data['avg_anxiety'].mean()
            profile['avg_anxiety_std'] = cluster_data['avg_anxiety'].std()
        
        cluster_profiles.append(profile)
    
    cluster_summary_df = pd.DataFrame(cluster_profiles)
    
    # Print summary
    print("\nCluster Summaries (mean ± std):")
    print("-" * 80)
    for idx, row in cluster_summary_df.iterrows():
        print(f"\nCluster {int(row['cluster'])} ({int(row['n_users'])} users):")
        if 'avg_hr_mean' in row:
            print(f"  Heart Rate:    {row['avg_hr_mean']:.1f} ± {row.get('avg_hr_std', 0):.1f} bpm")
        if 'avg_steps_mean' in row:
            print(f"  Steps:         {row['avg_steps_mean']:.0f} ± {row.get('avg_steps_std', 0):.0f}")
        if 'exam_exposure_mean' in row:
            print(f"  Exam Exposure: {row['exam_exposure_mean']:.3f} ± {row.get('exam_exposure_std', 0):.3f}")
        if 'avg_stress_mean' in row:
            print(f"  Avg Stress:    {row['avg_stress_mean']:.1f} ± {row.get('avg_stress_std', 0):.1f} (NOT used for clustering)")
        if 'avg_anxiety_mean' in row:
            print(f"  Avg Anxiety:   {row['avg_anxiety_mean']:.1f} ± {row.get('avg_anxiety_std', 0):.1f} (NOT used for clustering)")
    
    # Save cluster summary
    cluster_summary_df.to_csv(os.path.join(window_output_dir, 'cluster_summary.csv'), index=False)
    print(f"\n✓ Saved: cluster_summary.csv")
    
    # ========================================================================
    # 8. PCA FOR VISUALIZATION
    # ========================================================================
    print(f"\n8. PCA FOR VISUALIZATION ({window_type})")
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
    
    # ========================================================================
    # 9. VISUALIZATIONS
    # ========================================================================
    print(f"\n9. CREATING VISUALIZATIONS ({window_type})")
    print("-"*80)
    
    # Define cluster colors
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#95E1D3', '#F38181']
    
    # -------------------------------------------------------------------------
    # 9.1. PCA Scatter Plot (2D)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cluster_id in range(5):
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
    ax.set_title(f'User Clusters - PCA Visualization\n{window_type} window (5 clusters, {len(user_profiles)} users)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'plots', 'clusters_pca_2d.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: clusters_pca_2d.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 9.2. Feature Distribution by Cluster
    # -------------------------------------------------------------------------
    n_features = len(user_feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feat in enumerate(user_feature_cols):
        ax = axes[idx]
        
        cluster_data_list = []
        cluster_labels = []
        for cluster_id in range(5):
            cluster_mask = user_profiles['user_cluster'] == cluster_id
            cluster_data_list.append(user_profiles[cluster_mask][feat].values)
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
    
    plt.suptitle(f'Feature Distributions by Cluster - {window_type} window', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'plots', 'feature_distributions_by_cluster.png'), 
               dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_distributions_by_cluster.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 9.3. Cluster Heatmap (Normalized Feature Means)
    # -------------------------------------------------------------------------
    # Create heatmap data
    heatmap_data = []
    for cluster_id in range(5):
        cluster_mask = user_profiles['user_cluster'] == cluster_id
        cluster_means = []
        for feat in user_feature_cols:
            cluster_means.append(user_profiles[cluster_mask][feat].mean())
        heatmap_data.append(cluster_means)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             columns=[f.replace('_', ' ').title() for f in user_feature_cols],
                             index=[f'Cluster {i}' for i in range(5)])
    
    # Normalize for better visualization
    heatmap_df_norm = (heatmap_df - heatmap_df.mean()) / heatmap_df.std()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_df_norm, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Normalized Value (z-score)'}, ax=ax, linewidths=0.5)
    ax.set_title(f'Cluster Characteristics Heatmap - {window_type} window\n(Normalized Feature Means)',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'plots', 'cluster_heatmap.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: cluster_heatmap.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 9.4. Stress/Anxiety by Cluster (if available)
    # -------------------------------------------------------------------------
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
            for cluster_id in range(5):
                cluster_mask = user_profiles['user_cluster'] == cluster_id
                cluster_data_list.append(user_profiles[cluster_mask][target_col].dropna().values)
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
        
        plt.suptitle(f'Stress/Anxiety by Cluster - {window_type} window', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(window_output_dir, 'plots', 'stress_anxiety_by_cluster.png'), 
                   dpi=300, bbox_inches='tight')
        print("✓ Saved: stress_anxiety_by_cluster.png")
        plt.close()
    
    # -------------------------------------------------------------------------
    # 9.5. Cluster Size Pie Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cluster_sizes = [np.sum(user_clusters == i) for i in range(5)]
    cluster_labels = [f'Cluster {i}\n({cluster_sizes[i]} users)' for i in range(5)]
    
    wedges, texts, autotexts = ax.pie(cluster_sizes, labels=cluster_labels, colors=cluster_colors,
                                       autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title(f'Cluster Size Distribution - {window_type} window\n{len(user_profiles)} total users',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'plots', 'cluster_sizes_pie.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: cluster_sizes_pie.png")
    plt.close()
    
    # -------------------------------------------------------------------------
    # 9.6. 3D PCA Plot (if we have 3+ features)
    # -------------------------------------------------------------------------
    if len(user_feature_cols) >= 3:
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_users_scaled)
        
        variance_3d = pca_3d.explained_variance_ratio_
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster_id in range(5):
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
        ax.set_title(f'User Clusters - 3D PCA Visualization\n{window_type} window (Total variance: {variance_3d.sum()*100:.1f}%)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(window_output_dir, 'plots', 'clusters_pca_3d.png'), dpi=300, bbox_inches='tight')
        print("✓ Saved: clusters_pca_3d.png")
        plt.close()
    
    # ========================================================================
    # 10. SAVE USER-CLUSTER MAPPING
    # ========================================================================
    print(f"\n10. SAVING RESULTS ({window_type})")
    print("-"*80)
    
    # Save user-cluster assignments
    user_cluster_mapping = user_profiles[['userid', 'user_cluster']].copy()
    user_cluster_mapping.to_csv(os.path.join(window_output_dir, 'user_cluster_assignments.csv'), index=False)
    print("✓ Saved: user_cluster_assignments.csv")
    
    # Save full user profiles with clusters
    user_profiles.to_csv(os.path.join(window_output_dir, 'user_profiles_with_clusters.csv'), index=False)
    print("✓ Saved: user_profiles_with_clusters.csv")
    
    # Store results
    all_clustering_results[window_type] = {
        'n_users': len(user_profiles),
        'n_clusters': 5,
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'cluster_sizes': cluster_sizes,
        'features_used': user_feature_cols
    }
    
    print("\n" + "#"*80)
    print(f"# ✅ COMPLETE: {window_type} WINDOW")
    print("#"*80)

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("OVERALL CLUSTERING SUMMARY")
print("="*80)

if all_clustering_results:
    print("\n" + "-"*80)
    print("CLUSTERING QUALITY COMPARISON:")
    print("-"*80)
    
    for window_type in ['5min', '10min']:
        if window_type in all_clustering_results:
            res = all_clustering_results[window_type]
            print(f"\n{window_type.upper()} Window:")
            print(f"  Users:                   {res['n_users']}")
            print(f"  Clusters:                {res['n_clusters']}")
            print(f"  Silhouette Score:        {res['silhouette_score']:.4f}")
            print(f"  Davies-Bouldin Index:    {res['davies_bouldin_index']:.4f}")
            print(f"  Calinski-Harabasz Score: {res['calinski_harabasz_score']:.2f}")
            print(f"  Features Used:           {', '.join(res['features_used'])}")

print("✅ CLUSTERING ANALYSIS COMPLETE!")
