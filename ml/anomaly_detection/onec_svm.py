"""
One-Class SVM Anomaly Detection for Stress and Anxiety
Multi-Window Analysis: 5min, 10min, 30min, 60min, and Combined Enriched Dataset

One-Class SVM is particularly effective for:
- High-dimensional data
- Non-linear decision boundaries
- Capturing complex patterns in feature space
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class StressAnxietyOCSVM:
    """
    One-Class SVM model for detecting anomalies in stress and anxiety patterns
    """
    
    def __init__(self, nu=0.1, kernel='rbf', gamma='scale', random_state=42):
        """
        Initialize the One-Class SVM detector
        
        Parameters:
        -----------
        nu : float
            Upper bound on the fraction of outliers (default: 0.1 or 10%)
            Also lower bound on the fraction of support vectors
        kernel : str
            Kernel type: 'rbf', 'linear', 'poly', 'sigmoid'
            'rbf' (Radial Basis Function) is most common for anomaly detection
        gamma : str or float
            Kernel coefficient. 'scale' uses 1 / (n_features * X.var())
        random_state : int
            Random state for reproducibility
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
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
        Fit the One-Class SVM model
        """
        print(f"Preparing features...")
        X = self.prepare_features(df)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Number of samples: {len(X)}")
        
        # Scale features (critical for SVM)
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit One-Class SVM
        print(f"Training One-Class SVM (nu={self.nu}, kernel={self.kernel})...")
        print("Note: SVM training may take longer than Isolation Forest...")
        
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            cache_size=1000  # Increase cache for faster training
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
        decision_scores : array
            Decision function values (negative = more anomalous)
        """
        X = self.prepare_features(df)
        X = X.fillna(X.median())
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        decision_scores = self.model.decision_function(X_scaled)
        
        return predictions, decision_scores
    
    def analyze_results(self, df, predictions, decision_scores):
        """
        Analyze and visualize anomaly detection results
        """
        results_df = df.copy()
        results_df['anomaly'] = predictions
        results_df['decision_score'] = decision_scores
        results_df['is_anomaly'] = (predictions == -1).astype(int)
        
        # Statistics
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions) * 100
        
        print("\n" + "="*80)
        print("ONE-CLASS SVM ANOMALY DETECTION RESULTS")
        print("="*80)
        print(f"Total samples: {len(predictions)}")
        print(f"Anomalies detected: {n_anomalies} ({anomaly_rate:.2f}%)")
        print(f"Normal samples: {(predictions == 1).sum()} ({100-anomaly_rate:.2f}%)")
        print(f"Number of support vectors: {len(self.model.support_vectors_)}")
        print(f"Support vector percentage: {len(self.model.support_vectors_) / len(predictions) * 100:.2f}%")
        
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
    
    def get_feature_importance(self, df_original, decision_scores, top_n=20):
        """
        Analyze feature importance using decision scores
        For linear kernel, we can get actual coefficients
        For RBF/other kernels, we use correlation with decision scores
        """
        X = self.prepare_features(df_original)
        X = X.fillna(X.median())
        
        if self.kernel == 'linear':
            # For linear kernel, we can get actual feature coefficients
            feature_importance = {}
            coef = self.model.coef_[0]
            for i, col in enumerate(self.feature_names):
                feature_importance[col] = np.abs(coef[i])
        else:
            # For non-linear kernels, use correlation with decision scores
            feature_importance = {}
            for col in self.feature_names:
                corr = np.abs(np.corrcoef(X[col], decision_scores)[0, 1])
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
            output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_ocsvm'
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Decision Score Distribution
        plt.subplot(2, 3, 1)
        plt.hist(results_df[results_df['anomaly']==1]['decision_score'], 
                bins=50, alpha=0.7, label='Normal', color='green')
        plt.hist(results_df[results_df['anomaly']==-1]['decision_score'], 
                bins=50, alpha=0.7, label='Anomaly', color='red')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
        plt.xlabel('Decision Score (negative = anomalous)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Decision Scores (One-Class SVM)')
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
            plt.title('Anomaly Rate by User (One-Class SVM)')
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
        
        # Plot support vectors if possible
        if hasattr(self.model, 'support_'):
            support_indices = self.model.support_
            plt.scatter(X_pca[support_indices, 0], X_pca[support_indices, 1],
                       s=100, facecolors='none', edgecolors='blue', linewidths=2,
                       label='Support Vectors')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Visualization with Support Vectors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'ocsvm_analysis_{title_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {save_path}")
        plt.close()
        
        return fig


def analyze_dataset(file_path, window_name, nu=0.1, kernel='rbf', output_dir=None):
    """
    Analyze a single dataset with One-Class SVM
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    window_name : str
        Name identifier for the window
    nu : float
        Expected proportion of outliers
    kernel : str
        SVM kernel type
    output_dir : str
        Directory to save outputs
    """
    if output_dir is None:
        output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_ocsvm'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*100)
    print(f"ANALYZING: {window_name}")
    print("="*100)
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Initialize detector
    detector = StressAnxietyOCSVM(nu=nu, kernel=kernel)
    
    # Fit model
    detector.fit(df)
    
    # Predict anomalies
    predictions, decision_scores = detector.predict(df)
    
    # Analyze results
    results_df = detector.analyze_results(df, predictions, decision_scores)
    
    # Feature importance
    print(f"\n{'-'*80}")
    print("TOP 20 MOST IMPORTANT FEATURES:")
    print(f"{'-'*80}")
    importance_df = detector.get_feature_importance(df, decision_scores, top_n=20)
    print(importance_df.to_string(index=False))
    
    # Visualize
    detector.visualize_results(df, results_df, title_suffix=window_name, output_dir=output_dir)
    
    # Save results
    output_file = os.path.join(output_dir, f'ocsvm_results_{window_name}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")
    
    # Save feature importance
    importance_file = os.path.join(output_dir, f'ocsvm_feature_importance_{window_name}.csv')
    importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance saved: {importance_file}")
    
    return detector, results_df, importance_df


def compare_kernels(file_path, window_name, nu=0.1, output_dir=None):
    """
    Compare different SVM kernels on the same dataset
    """
    if output_dir is None:
        output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_ocsvm'
    
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    results = {}
    
    print("\n" + "="*100)
    print(f"KERNEL COMPARISON FOR: {window_name}")
    print("="*100)
    
    df = pd.read_csv(file_path)
    
    for kernel in kernels:
        print(f"\n{'='*80}")
        print(f"Testing kernel: {kernel.upper()}")
        print(f"{'='*80}")
        
        try:
            detector = StressAnxietyOCSVM(nu=nu, kernel=kernel)
            detector.fit(df)
            predictions, decision_scores = detector.predict(df)
            
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            results[kernel] = {
                'anomalies': n_anomalies,
                'rate': anomaly_rate,
                'support_vectors': len(detector.model.support_vectors_)
            }
            
            print(f"Anomalies: {n_anomalies} ({anomaly_rate:.2f}%)")
            print(f"Support Vectors: {len(detector.model.support_vectors_)}")
            
        except Exception as e:
            print(f"Error with {kernel} kernel: {str(e)}")
            results[kernel] = None
    
    # Save comparison
    comparison_df = pd.DataFrame(results).T
    comparison_file = os.path.join(output_dir, f'kernel_comparison_{window_name}.csv')
    comparison_df.to_csv(comparison_file)
    print(f"\nKernel comparison saved: {comparison_file}")
    
    return results


def main():
    """
    Main execution function
    """
    print("="*100)
    print("ONE-CLASS SVM ANOMALY DETECTION FOR STRESS AND ANXIETY")
    print("="*100)
    
    # Define datasets - UPDATE TO YOUR LOCAL PATHS
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
    print("="*100)
    for window_name, file_path in datasets.items():
        if os.path.exists(file_path):
            df_check = pd.read_csv(file_path, nrows=1, quoting=1)
            
            has_stress = 'q_i_stress_sliderNeutralPos' in df_check.columns 
            has_anxiety = 'q_i_anxiety_sliderNeutralPos' in df_check.columns
            print(f"✓ {window_name:20s}: {file_path}")
            print(f"  Columns: {len(df_check.columns)}, Has Stress: {has_stress}, Has Anxiety: {has_anxiety}")
        else:
            print(f"✗ {window_name:20s}: FILE NOT FOUND - {file_path}")
    print("="*100)
    
    # Store results
    all_results = {}
    all_importance = {}
    
    # Set output directory
    output_dir = '/Users/YusMolina/Downloads/smieae/results/anomaly_detection_ocsvm'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Analyze each dataset with RBF kernel (most common for anomaly detection)
    nu_rate = 0.1  # 10% expected anomalies
    kernel_type = 'rbf'  # Can also try: 'linear', 'poly', 'sigmoid'
    
    for window_name, file_path in datasets.items():
        if not os.path.exists(file_path):
            print(f"\nSkipping {window_name}: file not found")
            continue
            
        try:
            detector, results_df, importance_df = analyze_dataset(
                file_path, window_name, nu=nu_rate, kernel=kernel_type, output_dir=output_dir
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
    print("SUMMARY COMPARISON ACROSS ALL WINDOWS (ONE-CLASS SVM)")
    print("="*100)
    
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
    summary_path = os.path.join(output_dir, 'ocsvm_detection_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    
    print("\n" + "="*100)
    print("ONE-CLASS SVM ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - ocsvm_results_{window}.csv (for each window)")
    print("  - ocsvm_feature_importance_{window}.csv (for each window)")
    print("  - ocsvm_analysis_{window}.png (for each window)")
    print("  - ocsvm_detection_summary.csv (overall comparison)")
    print("\nKernel used: " + kernel_type)
    print("Nu parameter: " + str(nu_rate))


if __name__ == "__main__":
    main()