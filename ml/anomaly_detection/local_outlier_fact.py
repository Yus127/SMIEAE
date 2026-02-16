"""
Local Outlier Factor (LOF) Anomaly Detection for Stress and Anxiety
Multi-Window Analysis: 5min, 10min, 30min, 60min, and Combined Enriched Dataset

LOF is particularly effective for:
- Detecting local density-based outliers
- Finding anomalies in clusters of varying densities
- Capturing context-dependent anomalies
- Neighborhood-based detection
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class StressAnxietyLOF:
    """
    Local Outlier Factor model for detecting anomalies in stress and anxiety patterns
    """
    
    def __init__(self, contamination=0.1, n_neighbors=20, novelty=False, random_state=42):
        """
        Initialize the LOF detector
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers (default: 0.1 or 10%)
        n_neighbors : int
            Number of neighbors to use for LOF computation
            Higher values = more global detection
            Lower values = more local detection
        novelty : bool
            If True, can predict on new data (but slower)
            If False, only fit_predict on training data (faster)
        random_state : int
            Random state for reproducibility
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.novelty = novelty
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.pca = None
        
    def prepare_features(self, df):
        """
        Prepare features for anomaly detection
        """
        # Identify numeric columns (excluding IDs and timestamps)
        exclude_cols = ['userid', 'q_userid', 'response_timestamp', 'window_start', 'window_end',
                       'q_timestamp_fmt', 'q_timestamp_for_day', 'q_date_only', 'q_weekday',
                       'course', 'window_start_30min', 'window_end_30min', 
                       'window_start_1hour', 'window_end_1hour']
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Store target variables separately
        self.stress_col = 'q_i_stress_sliderNeutralPos' if 'q_i_stress_sliderNeutralPos' in df.columns else None
        self.anxiety_col = 'q_i_anxiety_sliderNeutralPos' if 'q_i_anxiety_sliderNeutralPos' in df.columns else None
        
        # Remove stress and anxiety from features (we're predicting these)
        if self.stress_col in feature_cols:
            feature_cols.remove(self.stress_col)
        if self.anxiety_col in feature_cols:
            feature_cols.remove(self.anxiety_col)
        
        self.feature_names = feature_cols
        
        return df[feature_cols].copy()
    
    def fit(self, df):
        """
        Fit the LOF model
        """
        X = self.prepare_features(df)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of samples: {len(X)}")
        
        # Scale features (important for distance-based algorithms)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit LOF
        
        self.model = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors,
            novelty=self.novelty,
            n_jobs=-1
        )
        
        # LOF doesn't have separate fit() - it's fit_predict()
        # We'll store the scaled data for later use
        self.X_scaled = X_scaled
        
        print("Model training complete!")
        
        return self
    
    def predict(self, df):
        """
        Predict anomalies
        
        Returns:
        --------
        predictions : array
            1 for normal, -1 for anomaly
        lof_scores : array
            Negative LOF scores (lower = more anomalous)
        """
        X = self.prepare_features(df)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        # Fit and predict in one go (LOF behavior)
        predictions = self.model.fit_predict(X_scaled)
        
        # Get negative outlier factor (lower = more anomalous)
        lof_scores = self.model.negative_outlier_factor_
        
        return predictions, lof_scores
    
    def analyze_results(self, df, predictions, lof_scores):
        """
        Analyze and visualize anomaly detection results
        """
        results_df = df.copy()
        results_df['anomaly'] = predictions
        results_df['lof_score'] = lof_scores
        results_df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Statistics
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        print("\n" + "="*80)
        print("LOCAL OUTLIER FACTOR ANOMALY DETECTION RESULTS")
        print(f"Total samples: {len(predictions)}")
        print(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")
        print(f"Normal samples: {(predictions == 1).sum()} ({100-anomaly_rate:.2f}%)")
        print(f"Number of neighbors used: {self.n_neighbors}")
        print(f"Mean LOF score (normal): {lof_scores[predictions == 1].mean():.4f}")
        print(f"Mean LOF score (anomaly): {lof_scores[predictions == -1].mean():.4f}")
        
        # Debug: Check for stress/anxiety columns
        stress_anxiety_cols = [col for col in results_df.columns if 'stress' in col.lower() or 'anxiety' in col.lower()]
        if stress_anxiety_cols and not self.stress_col:
            print(f"\n{'-'*80}")
            print(f"WARNING: Found stress/anxiety columns but they weren't identified during feature prep:")
            print(f"  Available: {stress_anxiety_cols}")
            # Try to set them now
            if 'q_i_stress_sliderNeutralPos' in results_df.columns:
                self.stress_col = 'q_i_stress_sliderNeutralPos'
            if 'q_i_anxiety_sliderNeutralPos' in results_df.columns:
                self.anxiety_col = 'q_i_anxiety_sliderNeutralPos'
        
        # Analyze stress and anxiety in anomalies
        if self.stress_col and self.stress_col in results_df.columns:
            stress_normal = results_df[results_df['anomaly']==1][self.stress_col].mean()
            stress_anomaly = results_df[results_df['anomaly']==-1][self.stress_col].mean()
            if not np.isnan(stress_normal) and not np.isnan(stress_anomaly):
                print(f"\n{'-'*80}")
                print("STRESS LEVELS:")
                print(f"Normal samples - Mean stress: {stress_normal:.2f}")
                print(f"Anomalies - Mean stress: {stress_anomaly:.2f}")
            
        if self.anxiety_col and self.anxiety_col in results_df.columns:
            anxiety_normal = results_df[results_df['anomaly']==1][self.anxiety_col].mean()
            anxiety_anomaly = results_df[results_df['anomaly']==-1][self.anxiety_col].mean()
            if not np.isnan(anxiety_normal) and not np.isnan(anxiety_anomaly):
                print(f"\n{'-'*80}")
                print("ANXIETY LEVELS:")
                print(f"Normal samples - Mean anxiety: {anxiety_normal:.2f}")
                print(f"Anomalies - Mean anxiety: {anxiety_anomaly:.2f}")
        
        return results_df
    
    def get_feature_importance(self, df_original, lof_scores, top_n=20):
        """
        Analyze feature importance using LOF scores
        Uses correlation between features and LOF scores
        """
        X = self.prepare_features(df_original)
        X = X.fillna(X.median())
        
        # Calculate correlation between features and LOF scores
        feature_importance = {}
        for col in self.feature_names:
            corr = np.abs(np.corrcoef(X[col], lof_scores)[0, 1])
            feature_importance[col] = corr
        
        # Sort by importance
        importance_df = pd.DataFrame(
            list(feature_importance.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def visualize_results(self, df_original, results_df, title_suffix="", output_dir=None):
        """
        Create comprehensive visualizations
        """
        if output_dir is None:
            output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_lof'
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. LOF Score Distribution
        plt.subplot(2, 3, 1)
        plt.hist(results_df[results_df['anomaly']==1]['lof_score'], 
                bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(results_df[results_df['anomaly']==-1]['lof_score'], 
                bins=50, alpha=0.7, label='Anomaly', color='red')
        plt.axvline(x=-1, color='black', linestyle='--', linewidth=2, label='LOF Threshold')
        plt.xlabel('Negative Outlier Factor (lower = more anomalous)')
        plt.ylabel('Frequency')
        plt.title('Distribution of LOF Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Stress levels comparison
        if self.stress_col and self.stress_col in results_df.columns:
            plt.subplot(2, 3, 2)
            results_df.boxplot(column=self.stress_col, by='anomaly', ax=plt.gca())
            plt.xlabel('Sample Type (-1: Anomaly, 1: Normal)')
            plt.ylabel('Stress Level')
            plt.title('Stress Levels: Normal vs Anomalies')
            plt.suptitle('')
        
        # 3. Anxiety levels comparison
        if self.anxiety_col and self.anxiety_col in results_df.columns:
            plt.subplot(2, 3, 3)
            results_df.boxplot(column=self.anxiety_col, by='anomaly', ax=plt.gca())
            plt.xlabel('Sample Type (-1: Anomaly, 1: Normal)')
            plt.ylabel('Anxiety Level')
            plt.title('Anxiety Levels: Normal vs Anomalies')
            plt.suptitle('')
        
        # 4. Anomaly rate by user
        plt.subplot(2, 3, 4)
        if 'userid' in results_df.columns:
            user_anomalies = results_df.groupby('userid')['is_anomaly'].agg(['sum', 'count'])
            user_anomalies['rate'] = (user_anomalies['sum'] / user_anomalies['count']) * 100
            user_anomalies['rate'].plot(kind='bar', color='coral')
            plt.xlabel('User ID')
            plt.ylabel('Anomaly Rate (%)')
            plt.title('Anomaly Rate by User (LOF)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 5. Temporal pattern
        plt.subplot(2, 3, 5)
        if 'q_hour' in results_df.columns:
            hour_anomalies = results_df.groupby('q_hour')['is_anomaly'].agg(['sum', 'count'])
            hour_anomalies['rate'] = (hour_anomalies['sum'] / hour_anomalies['count']) * 100
            hour_anomalies['rate'].plot(kind='line', marker='o', color='purple')
            plt.xlabel('Hour of Day')
            plt.ylabel('Anomaly Rate (%)')
            plt.title('Anomaly Rate by Hour')
            plt.grid(True, alpha=0.3)
        
        # 6. PCA Visualization with LOF scores as color intensity
        plt.subplot(2, 3, 6)
        X = self.prepare_features(df_original)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        normal_mask = results_df['anomaly'] == 1
        anomaly_mask = results_df['anomaly'] == -1
        
        # Plot normal samples
        scatter1 = plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c=results_df[normal_mask]['lof_score'], cmap='Greens',
                   alpha=0.5, s=20, label='Normal', vmin=-2, vmax=-0.5)
        
        # Plot anomalies with red X
        plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
        
        plt.colorbar(scatter1, label='LOF Score (higher = more normal)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization with LOF Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'lof_analysis_{title_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
        plt.close()
        
        return fig


def analyze_dataset(file_path, window_name, contamination=0.1, n_neighbors=20, output_dir=None):
    """
    Analyze a single dataset with Local Outlier Factor
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    window_name : str
        Name identifier for the window
    contamination : float
        Expected proportion of outliers
    n_neighbors : int
        Number of neighbors for LOF computation
    output_dir : str
        Directory to save outputs
    """
    if output_dir is None:
        output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_lof'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print(f"ANALYZING: {window_name}")
    
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape}")
    
    # Initialize detector
    detector = StressAnxietyLOF(contamination=contamination, n_neighbors=n_neighbors)
    
    # Fit model
    detector.fit(df)
    
    # Predict anomalies
    predictions, lof_scores = detector.predict(df)
    
    # Analyze results
    results_df = detector.analyze_results(df, predictions, lof_scores)
    
    # Feature importance
    print(f"\n{'-'*80}")
    print("TOP 20 MOST IMPORTANT FEATURES:")
    print(f"{'-'*80}")
    importance_df = detector.get_feature_importance(df, lof_scores, top_n=20)
    print(importance_df.to_string(index=False))
    
    # Visualize
    detector.visualize_results(df, results_df, title_suffix=window_name, output_dir=output_dir)
    
    output_file = os.path.join(output_dir, f'lof_results_{window_name}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    
    # Save feature importance
    importance_file = os.path.join(output_dir, f'lof_feature_importance_{window_name}.csv')
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved: {importance_file}")
    
    return detector, results_df, importance_df


def compare_n_neighbors(file_path, window_name, contamination=0.1, output_dir=None):
    """
    Compare different n_neighbors values on the same dataset
    """
    if output_dir is None:
        output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_lof'
    
    n_neighbors_list = [5, 10, 20, 30, 50]
    results = {}
    
    print("\n" + "="*100)
    print(f"N_NEIGHBORS COMPARISON FOR: {window_name}")
    
    df = pd.read_csv(file_path)
    
    for n_neighbors in n_neighbors_list:
        print(f"\n{'='*80}")
        print(f"Testing n_neighbors: {n_neighbors}")
        print(f"{'='*80}")
        
        try:
            detector = StressAnxietyLOF(contamination=contamination, n_neighbors=n_neighbors)
            detector.fit(df)
            predictions, lof_scores = detector.predict(df)
            
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions) * 100
            mean_lof_normal = lof_scores[predictions == 1].mean()
            mean_lof_anomaly = lof_scores[predictions == -1].mean()
            
            results[n_neighbors] = {
                'anomalies': n_anomalies,
                'rate': anomaly_rate,
                'mean_lof_normal': mean_lof_normal,
                'mean_lof_anomaly': mean_lof_anomaly
            }
            
            print(f"Anomalies: {n_anomalies} ({anomaly_rate:.2f}%)")
            print(f"Mean LOF (normal): {mean_lof_normal:.4f}")
            print(f"Mean LOF (anomaly): {mean_lof_anomaly:.4f}")
            
        except Exception as e:
            print(f"Error with n_neighbors={n_neighbors}: {str(e)}")
            results[n_neighbors] = None
    
    # Save comparison
    comparison_df = pd.DataFrame(results).T
    comparison_file = os.path.join(output_dir, f'n_neighbors_comparison_{window_name}.csv')
    comparison_df.to_csv(comparison_file)
    print(f"\nN_neighbors comparison saved: {comparison_file}")
    
    return results


def main():
    """
    Main execution function
    """
    print("LOCAL OUTLIER FACTOR (LOF) ANOMALY DETECTION FOR STRESS AND ANXIETY")
    
    ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
    
    datasets = {
        '5min': f'{ml_dir}/enriched/ml_ready_5min_window_enriched.csv',
        '10min': f'{ml_dir}/enriched/ml_ready_10min_window_enriched.csv',
        '30min': f'{ml_dir}/enriched/ml_ready_30min_window_enriched.csv',
        '60min': f'{ml_dir}/enriched/ml_ready_60min_window_enriched.csv',
        'combined_enriched': f'{ml_dir}/enriched/data_with_exam_features.csv'
    }
    
    # Diagnostic: Check files exist
    print("\n" + "="*100)
    print("FILE DIAGNOSTICS")
    for window_name, file_path in datasets.items():
        if os.path.exists(file_path):
            df_check = pd.read_csv(file_path, nrows=1)
            has_stress = 'q_i_stress_sliderNeutralPos' in df_check.columns
            has_anxiety = 'q_i_anxiety_sliderNeutralPos' in df_check.columns
            print(f" {window_name:20s}: {file_path}")
            print(f"  Columns: {len(df_check.columns)}, Has Stress: {has_stress}, Has Anxiety: {has_anxiety}")
        else:
            print(f" {window_name:20s}: FILE NOT FOUND - {file_path}")
    
    # Store results
    all_results = {}
    all_importance = {}
    
    # Set output directory
    output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_lof'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Analyze each dataset with LOF
    contamination_rate = 0.1  # 10% expected anomalies
    n_neighbors = 20  # Standard choice - can also try: 5, 10, 30, 50
    
    for window_name, file_path in datasets.items():
        if not os.path.exists(file_path):
            print(f"\nSkipping {window_name}: file not found")
            continue
            
        try:
            detector, results_df, importance_df = analyze_dataset(
                file_path, window_name, 
                contamination=contamination_rate, 
                n_neighbors=n_neighbors, 
                output_dir=output_dir
            )
            all_results[window_name] = results_df
            all_importance[window_name] = importance_df
        except Exception as e:
            print(f"\nError processing {window_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary comparison
    print("\n" + "="*100)
    print("SUMMARY COMPARISON ACROSS ALL WINDOWS (LOCAL OUTLIER FACTOR)")
    
    summary_data = []
    for window_name, results_df in all_results.items():
        n_total = len(results_df)
        n_anomalies = (results_df['anomaly'] == -1).sum()
        anomaly_rate = (n_anomalies / n_total) * 100
        
        # Check if stress column exists and has valid data
        if 'q_i_stress_sliderNeutralPos' in results_df.columns:
            normal_stress_vals = results_df[results_df['anomaly']==1]['q_i_stress_sliderNeutralPos'].dropna()
            anomaly_stress_vals = results_df[results_df['anomaly']==-1]['q_i_stress_sliderNeutralPos'].dropna()
            
            if len(normal_stress_vals) > 0 and len(anomaly_stress_vals) > 0:
                normal_stress = normal_stress_vals.mean()
                anomaly_stress = anomaly_stress_vals.mean()
            else:
                normal_stress = anomaly_stress = np.nan
        else:
            normal_stress = anomaly_stress = np.nan
            
        # Check if anxiety column exists and has valid data
        if 'q_i_anxiety_sliderNeutralPos' in results_df.columns:
            normal_anxiety_vals = results_df[results_df['anomaly']==1]['q_i_anxiety_sliderNeutralPos'].dropna()
            anomaly_anxiety_vals = results_df[results_df['anomaly']==-1]['q_i_anxiety_sliderNeutralPos'].dropna()
            
            if len(normal_anxiety_vals) > 0 and len(anomaly_anxiety_vals) > 0:
                normal_anxiety = normal_anxiety_vals.mean()
                anomaly_anxiety = anomaly_anxiety_vals.mean()
            else:
                normal_anxiety = anomaly_anxiety = np.nan
        else:
            normal_anxiety = anomaly_anxiety = np.nan
        
        summary_data.append({
            'Window': window_name,
            'Total Samples': n_total,
            'Anomalies': n_anomalies,
            'Anomaly Rate (%)': f"{anomaly_rate:.2f}",
            'Normal Stress': f"{normal_stress:.2f}" if not np.isnan(normal_stress) else 'N/A',
            'Anomaly Stress': f"{anomaly_stress:.2f}" if not np.isnan(anomaly_stress) else 'N/A',
            'Normal Anxiety': f"{normal_anxiety:.2f}" if not np.isnan(normal_anxiety) else 'N/A',
            'Anomaly Anxiety': f"{anomaly_anxiety:.2f}" if not np.isnan(anomaly_anxiety) else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    # Save summary
    summary_path = os.path.join(output_dir, 'lof_detection_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    
    print("\n" + "="*100)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - lof_results_{window}.csv (for each window)")
    print("  - lof_feature_importance_{window}.csv (for each window)")
    print("  - lof_analysis_{window}.png (for each window)")
    print("  - lof_detection_summary.csv (overall comparison)")
    print(f"\nN_neighbors parameter: {n_neighbors}")
    print(f"Contamination parameter: {contamination_rate}")


if __name__ == "__main__":
    main()
