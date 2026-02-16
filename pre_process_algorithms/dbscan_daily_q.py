import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv')
# Select the relevant columns for clustering
features = ['i_stress_sliderNeutralPos', 'i_anxiety_sliderNeutralPos']
X = df[features].values

# Standardize the features (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
# eps: maximum distance between two samples to be considered in the same neighborhood
# min_samples: minimum number of samples in a neighborhood to form a cluster
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['cluster'] = clusters

# Print clustering results
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print(f"Number of clusters found: {n_clusters}")
print(f"Number of noise points: {n_noise}")
print(f"\nCluster distribution:")
print(df['cluster'].value_counts().sort_index())

# Per-user analysis
print("\n" + "="*60)
print("PER-USER CLUSTER ANALYSIS")

for user_id in df['userid'].unique():
    user_data = df[df['userid'] == user_id]
    print(f"\nUser {user_id}:")
    print(f"  Total entries: {len(user_data)}")
    print(f"  Cluster distribution:")
    for cluster_id in sorted(user_data['cluster'].unique()):
        count = len(user_data[user_data['cluster'] == cluster_id])
        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        print(f"    {cluster_name}: {count} entries ({count/len(user_data)*100:.1f}%)")

# Statistics by cluster
print("\n" + "="*60)
print("CLUSTER STATISTICS")

for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise Points"
    
    print(f"\n{cluster_name} (n={len(cluster_data)}):")
    print(f"  Stress - Mean: {cluster_data['i_stress_sliderNeutralPos'].mean():.2f}, "
          f"Std: {cluster_data['i_stress_sliderNeutralPos'].std():.2f}")
    print(f"  Anxiety - Mean: {cluster_data['i_anxiety_sliderNeutralPos'].mean():.2f}, "
          f"Std: {cluster_data['i_anxiety_sliderNeutralPos'].std():.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))


df.to_csv('daily_questions_with_clusters.csv', index=False)
print("\nResults saved to 'daily_questions_with_clusters.csv'")

# Create and save summary report to text file
summary_report = []
summary_report.append("="*70)
summary_report.append("DBSCAN CLUSTERING ANALYSIS SUMMARY")
summary_report.append("Stress and Anxiety Patterns")
summary_report.append("="*70)
summary_report.append("")

# Overall clustering results
summary_report.append("OVERALL CLUSTERING RESULTS")
summary_report.append("-"*70)
summary_report.append(f"Total data points analyzed: {len(df)}")
summary_report.append(f"Number of clusters identified: {n_clusters}")
summary_report.append(f"Number of noise/outlier points: {n_noise}")
summary_report.append(f"Percentage of data classified as noise: {(n_noise/len(df)*100):.2f}%")
summary_report.append("")

# Cluster distribution
summary_report.append("Cluster Distribution:")
for cluster_id in sorted(df['cluster'].unique()):
    count = len(df[df['cluster'] == cluster_id])
    cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
    summary_report.append(f"  {cluster_name}: {count} points ({count/len(df)*100:.2f}%)")
summary_report.append("")

# Per-user summary
summary_report.append("="*70)
summary_report.append("PER-USER ANALYSIS")
summary_report.append("="*70)
summary_report.append("")

for user_id in sorted(df['userid'].unique()):
    user_data = df[df['userid'] == user_id]
    summary_report.append(f"USER {user_id}")
    summary_report.append("-"*70)
    summary_report.append(f"Total entries: {len(user_data)}")
    summary_report.append(f"Date range: {user_data['date_only'].min()} to {user_data['date_only'].max()}")
    summary_report.append("")
    summary_report.append("Cluster Distribution:")
    
    for cluster_id in sorted(user_data['cluster'].unique()):
        count = len(user_data[user_data['cluster'] == cluster_id])
        cluster_name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
        summary_report.append(f"  {cluster_name}: {count} entries ({count/len(user_data)*100:.1f}%)")
    
    summary_report.append("")
    summary_report.append("Average Stress and Anxiety Levels:")
    summary_report.append(f"  Overall Stress: {user_data['i_stress_sliderNeutralPos'].mean():.2f} (±{user_data['i_stress_sliderNeutralPos'].std():.2f})")
    summary_report.append(f"  Overall Anxiety: {user_data['i_anxiety_sliderNeutralPos'].mean():.2f} (±{user_data['i_anxiety_sliderNeutralPos'].std():.2f})")
    summary_report.append("")

# Cluster characteristics
summary_report.append("="*70)
summary_report.append("CLUSTER CHARACTERISTICS")
summary_report.append("="*70)
summary_report.append("")

for cluster_id in sorted(df['cluster'].unique()):
    cluster_data = df[df['cluster'] == cluster_id]
    cluster_name = f"CLUSTER {cluster_id}" if cluster_id != -1 else "NOISE POINTS"
    
    summary_report.append(cluster_name)
    summary_report.append("-"*70)
    summary_report.append(f"Number of points: {len(cluster_data)}")
    summary_report.append(f"Percentage of total data: {(len(cluster_data)/len(df)*100):.2f}%")
    summary_report.append("")
    summary_report.append("Stress Levels:")
    summary_report.append(f"  Mean: {cluster_data['i_stress_sliderNeutralPos'].mean():.2f}")
    summary_report.append(f"  Std Dev: {cluster_data['i_stress_sliderNeutralPos'].std():.2f}")
    summary_report.append(f"  Range: {cluster_data['i_stress_sliderNeutralPos'].min():.0f} - {cluster_data['i_stress_sliderNeutralPos'].max():.0f}")
    summary_report.append("")
    summary_report.append("Anxiety Levels:")
    summary_report.append(f"  Mean: {cluster_data['i_anxiety_sliderNeutralPos'].mean():.2f}")
    summary_report.append(f"  Std Dev: {cluster_data['i_anxiety_sliderNeutralPos'].std():.2f}")
    summary_report.append(f"  Range: {cluster_data['i_anxiety_sliderNeutralPos'].min():.0f} - {cluster_data['i_anxiety_sliderNeutralPos'].max():.0f}")
    summary_report.append("")
    
    # User distribution within cluster
    summary_report.append("User distribution in this cluster:")
    for user_id in sorted(cluster_data['userid'].unique()):
        user_count = len(cluster_data[cluster_data['userid'] == user_id])
        summary_report.append(f"  User {user_id}: {user_count} points ({user_count/len(cluster_data)*100:.1f}%)")
    summary_report.append("")

# Key insights
summary_report.append("="*70)
summary_report.append("KEY INSIGHTS")
summary_report.append("="*70)
summary_report.append("")

# Find cluster with highest stress
highest_stress_cluster = df.groupby('cluster')['i_stress_sliderNeutralPos'].mean().idxmax()
if highest_stress_cluster != -1:
    summary_report.append(f"Highest stress cluster: Cluster {highest_stress_cluster}")
    summary_report.append(f"  Average stress level: {df[df['cluster']==highest_stress_cluster]['i_stress_sliderNeutralPos'].mean():.2f}")

# Find cluster with highest anxiety
highest_anxiety_cluster = df.groupby('cluster')['i_anxiety_sliderNeutralPos'].mean().idxmax()
if highest_anxiety_cluster != -1:
    summary_report.append(f"Highest anxiety cluster: Cluster {highest_anxiety_cluster}")
    summary_report.append(f"  Average anxiety level: {df[df['cluster']==highest_anxiety_cluster]['i_anxiety_sliderNeutralPos'].mean():.2f}")

summary_report.append("")
summary_report.append("="*70)
summary_report.append("END OF REPORT")
summary_report.append("="*70)

# Write to file
with open('/Users/YusMolina/Downloads/smieae/data/data_clean/dbscan_clustering_summary.txt', 'w') as f:
    f.write('\n'.join(summary_report))

print("\nSummary report saved to 'dbscan_clustering_summary.txt'")

# Time-based analysis (optional)
print("\n" + "="*60)
print("TEMPORAL CLUSTER ANALYSIS")

df['date_only'] = pd.to_datetime(df['date_only'])
df = df.sort_values('date_only')

for user_id in df['userid'].unique():
    user_data = df[df['userid'] == user_id]
    print(f"\nUser {user_id} cluster progression over time:")
    print(user_data[['date_only', 'i_stress_sliderNeutralPos', 
                     'i_anxiety_sliderNeutralPos', 'cluster']].to_string(index=False))