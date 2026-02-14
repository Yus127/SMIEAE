"""
Isolation Forest Anomaly Detection for Stress and Anxiety
Multi-Window Analysis: 5min, 10min, 30min, 60min, and Combined Enriched Dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class StressAnxietyAnomalyDetector:
    """
    Isolation Forest model for detecting anomalies in stress and anxiety patterns
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize the detector
        
        Parameters:
        -----------
        contamination : float
            Expected proportion of outliers in the dataset (default: 0.1 or 10%)
        random_state : int
            Random state for reproducibility
        """
        self.contamination = contamination
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
        Fit the Isolation Forest model
        """
        print(f"Preparing features...")
        X = self.prepare_features(df)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of samples: {len(X)}")
        
        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        print(f"Training Isolation Forest (contamination={self.contamination})...")
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        print("Model training complete!")
        
        return self
    
    def predict(self, df):
        """
        Predict anomalies
        
        Returns:
        --------
        predictions : array
            1 for normal, -1 for anomaly
        anomaly_scores : array
            Anomaly scores (lower is more anomalous)
        """
        X = self.prepare_features(df)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        return predictions, anomaly_scores
    
    def analyze_results(self, df, predictions, anomaly_scores):
        """
        Analyze and visualize anomaly detection results
        """
        results_df = df.copy()
        results_df['anomaly'] = predictions
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Statistics
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        print("\n" + "="*80)
        print("ANOMALY DETECTION RESULTS")
        print("="*80)
        print(f"Total samples: {len(predictions)}")
        print(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")
        print(f"Normal samples: {(predictions == 1).sum()} ({100-anomaly_rate:.2f}%)")
        
        # Analyze stress and anxiety in anomalies
        if self.stress_col and self.stress_col in results_df.columns:
            print(f"\n{'-'*80}")
            print("STRESS LEVELS:")
            print(f"Normal samples - Mean stress: {results_df[results_df['anomaly']==1][self.stress_col].mean():.2f}")
            print(f"Anomalies - Mean stress: {results_df[results_df['anomaly']==-1][self.stress_col].mean():.2f}")
            
        if self.anxiety_col and self.anxiety_col in results_df.columns:
            print(f"\n{'-'*80}")
            print("ANXIETY LEVELS:")
            print(f"Normal samples - Mean anxiety: {results_df[results_df['anomaly']==1][self.anxiety_col].mean():.2f}")
            print(f"Anomalies - Mean anxiety: {results_df[results_df['anomaly']==-1][self.anxiety_col].mean():.2f}")
        
        return results_df
    
    def get_feature_importance(self, df_original, anomaly_scores, top_n=20):
        """
        Analyze feature importance using anomaly scores
        """
        X = self.prepare_features(df_original)
        X = X.fillna(X.median())
        
        feature_importance = {}
        for col in self.feature_names:
            corr = np.abs(np.corrcoef(X[col], anomaly_scores)[0, 1])
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
        
        Parameters:
        -----------
        df_original : DataFrame
            Original dataframe
        results_df : DataFrame
            Results with anomaly labels
        title_suffix : str
            Suffix for the filename
        output_dir : str
            Directory to save the visualization
        """
        if output_dir is None:
            output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection'
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Anomaly Score Distribution
        plt.subplot(2, 3, 1)
        plt.hist(results_df[results_df['anomaly']==1]['anomaly_score'], 
                bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(results_df[results_df['anomaly']==-1]['anomaly_score'], 
                bins=50, alpha=0.7, label='Anomaly', color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Anomaly Scores')
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
            plt.title('Anomaly Rate by User')
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
        
        # 6. PCA Visualization
        plt.subplot(2, 3, 6)
        X = self.prepare_features(df_original)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X_scaled)
        
        normal_mask = results_df['anomaly'] == 1
        anomaly_mask = results_df['anomaly'] == -1
        
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='green', alpha=0.5, s=20, label='Normal')
        plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=50, marker='x', label='Anomaly')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization of Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'anomaly_analysis_{title_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
        plt.close()
        
        return fig


def analyze_dataset(file_path, window_name, contamination=0.1, output_dir=None):
    """
    Analyze a single dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    window_name : str
        Name identifier for the window
    contamination : float
        Expected proportion of outliers
    output_dir : str
        Directory to save outputs (default: current directory)
    """
    if output_dir is None:
        output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection'
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print(f"ANALYZING: {window_name}")
    print("="*100)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize detector
    detector = StressAnxietyAnomalyDetector(contamination=contamination)
    
    # Fit model
    detector.fit(df)
    
    # Predict anomalies
    predictions, anomaly_scores = detector.predict(df)
    
    # Analyze results
    results_df = detector.analyze_results(df, predictions, anomaly_scores)
    
    # Feature importance
    print(f"\n{'-'*80}")
    print("TOP 20 MOST IMPORTANT FEATURES:")
    print(f"{'-'*80}")
    importance_df = detector.get_feature_importance(df, anomaly_scores, top_n=20)
    print(importance_df.to_string(index=False))
    
    # Visualize
    detector.visualize_results(df, results_df, title_suffix=window_name, output_dir=output_dir)
    
    # Save results
    output_file = os.path.join(output_dir, f'anomaly_results_{window_name}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    
    # Save feature importance
    importance_file = os.path.join(output_dir, f'feature_importance_{window_name}.csv')
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved: {importance_file}")
    
    return detector, results_df, importance_df


def main():
    """
    Main execution function
    """
    print("="*100)
    print("ISOLATION FOREST ANOMALY DETECTION FOR STRESS AND ANXIETY")
    print("="*100)
    
    # Define datasets - UPDATE THESE PATHS TO YOUR LOCAL DIRECTORY
    ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
    
    datasets = {
        '5min': f'{ml_dir}/enriched/ml_ready_5min_window_enriched.csv',
        '10min': f'{ml_dir}/enriched/ml_ready_10min_window_enriched.csv',
        '30min': f'{ml_dir}/enriched/ml_ready_30min_window_enriched.csv',
        '60min': f'{ml_dir}/enriched/ml_ready_60min_window_enriched.csv',
        'combined_enriched': f'{ml_dir}/enriched/data_with_exam_features.csv'
    }
    
    # Store results
    all_results = {}
    all_importance = {}
    
    # Set output directory
    output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection'
    import os
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Analyze each dataset
    contamination_rate = 0.1  # 10% expected anomalies
    
    for window_name, file_path in datasets.items():
        try:
            detector, results_df, importance_df = analyze_dataset(
                file_path, window_name, contamination=contamination_rate, output_dir=output_dir
            )
            all_results[window_name] = results_df
            all_importance[window_name] = importance_df
        except Exception as e:
            print(f"\nError processing {window_name}: {str(e)}")
            continue
    
    # Summary comparison
    print("\n" + "="*100)
    print("SUMMARY COMPARISON ACROSS ALL WINDOWS")
    print("="*100)
    
    summary_data = []
    for window_name, results_df in all_results.items():
        results_df.columns = results_df.columns.str.replace('"', '', regex=False).str.strip()

        n_total = len(results_df)
        n_anomalies = (results_df['anomaly'] == -1).sum()
        anomaly_rate = (n_anomalies / n_total) * 100
        
        if 'q_i_stress_sliderNeutralPos' in results_df.columns:
            normal_stress = results_df[results_df['anomaly']==1]['q_i_stress_sliderNeutralPos'].mean()
            anomaly_stress = results_df[results_df['anomaly']==-1]['q_i_stress_sliderNeutralPos'].mean()
        else:
            normal_stress = anomaly_stress = np.nan
            
        if 'q_i_anxiety_sliderNeutralPos' in results_df.columns:
            normal_anxiety = results_df[results_df['anomaly']==1]['q_i_anxiety_sliderNeutralPos'].mean()
            anomaly_anxiety = results_df[results_df['anomaly']==-1]['q_i_anxiety_sliderNeutralPos'].mean()
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
    summary_path = os.path.join(output_dir, 'anomaly_detection_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - anomaly_results_{window}.csv (for each window)")
    print("  - feature_importance_{window}.csv (for each window)")
    print("  - anomaly_analysis_{window}.png (for each window)")
    print("  - anomaly_detection_summary.csv (overall comparison)")


if __name__ == "__main__":
    main()