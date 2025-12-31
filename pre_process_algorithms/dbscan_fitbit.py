import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DBSCANProcessor:
    """
    Reusable processor for applying DBSCAN to multiple CSV files with missing data handling.
    """
    
    def __init__(self, missing_threshold=0.20, eps=0.5, min_samples=5):
        """
        Initialize the processor.
        
        Parameters:
        -----------
        missing_threshold : float
            Maximum proportion of missing values allowed (0.20 = 20%)
        eps : float
            DBSCAN epsilon parameter (maximum distance between two samples)
        min_samples : int
            DBSCAN minimum samples in a neighborhood for a core point
        """
        self.missing_threshold = missing_threshold
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        
    def load_and_clean_csv(self, filepath):
        """
        Load CSV and handle missing values according to the threshold.
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe
        dict : Statistics about the cleaning process
        """
        df = pd.read_csv(filepath)
        original_shape = df.shape
        stats = {
            'original_rows': original_shape[0],
            'original_cols': original_shape[1],
            'columns_dropped': [],
            'columns_imputed': [],
            'rows_dropped': 0
        }
        
        # Calculate missing percentage for each column
        missing_pct = df.isnull().sum() / len(df)
        
        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle columns based on missing data percentage
        for col in numeric_cols:
            if missing_pct[col] > self.missing_threshold:
                df = df.drop(columns=[col])
                stats['columns_dropped'].append(col)
            elif missing_pct[col] > 0:
                # Impute with mean for columns within threshold
                df[col].fillna(df[col].mean(), inplace=True)
                stats['columns_imputed'].append(col)
        
        # Drop non-numeric columns (or you can encode them if needed)
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            df = df.drop(columns=non_numeric_cols)
            stats['columns_dropped'].extend(non_numeric_cols)
        
        # Drop rows with any remaining NaN values
        rows_before = len(df)
        df = df.dropna()
        stats['rows_dropped'] = rows_before - len(df)
        stats['final_shape'] = df.shape
        
        return df, stats
    
    def apply_dbscan(self, df):
        """
        Apply DBSCAN clustering to the dataframe.
        
        Returns:
        --------
        np.array : Cluster labels
        pd.DataFrame : Original data with cluster labels
        dict : Clustering statistics
        """
        if df.empty:
            return None, None, {'error': 'Empty dataframe after cleaning'}
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(df)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Add labels to dataframe
        result_df = df.copy()
        result_df['cluster'] = labels
        
        # Calculate statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': (n_noise / len(labels)) * 100,
            'cluster_sizes': {}
        }
        
        for label in unique_labels:
            if label != -1:
                stats['cluster_sizes'][f'cluster_{label}'] = list(labels).count(label)
        
        return labels, result_df, stats
    
    def process_single_file(self, filepath, output_dir=None):
        """
        Complete pipeline for a single CSV file.
        
        Returns:
        --------
        dict : Complete results including data and statistics
        """
        print(f"\nProcessing: {filepath}")
        
        # Load and clean
        df_cleaned, cleaning_stats = self.load_and_clean_csv(filepath)
        print(f"  Cleaned: {cleaning_stats['final_shape'][0]} rows, {cleaning_stats['final_shape'][1]} columns")
        
        if df_cleaned.empty:
            print("  Warning: No data remaining after cleaning")
            return {'error': 'No data after cleaning', 'cleaning_stats': cleaning_stats}
        
        # Apply DBSCAN
        labels, result_df, cluster_stats = self.apply_dbscan(df_cleaned)
        print(f"  Clusters found: {cluster_stats['n_clusters']}")
        print(f"  Noise points: {cluster_stats['n_noise_points']} ({cluster_stats['noise_percentage']:.1f}%)")
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = Path(filepath).stem
            result_df.to_csv(output_path / f"{filename}_clustered.csv", index=False)
            
            # Save statistics
            stats_df = pd.DataFrame({
                'metric': list(cleaning_stats.keys()) + list(cluster_stats.keys()),
                'value': list(cleaning_stats.values()) + list(cluster_stats.values())
            })
            stats_df.to_csv(output_path / f"{filename}_stats.csv", index=False)
        
        return {
            'data': result_df,
            'labels': labels,
            'cleaning_stats': cleaning_stats,
            'cluster_stats': cluster_stats
        }
    
    def process_multiple_files(self, file_list, output_dir='output'):
        """
        Process multiple CSV files.
        
        Parameters:
        -----------
        file_list : list
            List of file paths to process
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        dict : Results for all files
        """
        results = {}
        
        for filepath in file_list:
            try:
                file_result = self.process_single_file(filepath, output_dir)
                results[Path(filepath).name] = file_result
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                results[Path(filepath).name] = {'error': str(e)}
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DBSCANProcessor(
        missing_threshold=0.20,  # 20% missing data threshold
        eps=0.5,                 # Adjust based on your data scale
        min_samples=5            # Adjust based on your dataset size
    )
    
    # Single file example
    # result = processor.process_single_file('your_file.csv', output_dir='output')
    
    # Multiple files example
    csv_files = [
        'file1.csv',
        'file2.csv',
        # Add all 36 files here
    ]
    
    # Or use glob to find all CSV files in a directory
    # from glob import glob
    # csv_files = glob('data/*.csv')
    
    # results = processor.process_multiple_files(csv_files, output_dir='output')
    
    # Print summary
    # for filename, result in results.items():
    #     if 'error' not in result:
    #         print(f"\n{filename}:")
    #         print(f"  Clusters: {result['cluster_stats']['n_clusters']}")
    #         print(f"  Noise: {result['cluster_stats']['noise_percentage']:.1f}%")
