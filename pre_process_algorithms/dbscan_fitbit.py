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
    
    def __init__(self, missing_threshold=0.20, eps=0.5, min_samples=5, 
                 remove_outliers=False, z_threshold=3.0):
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
        remove_outliers : bool
            Whether to remove extreme outliers based on z-score before clustering
        z_threshold : float
            Z-score threshold for outlier removal (typically 3.0)
        """
        self.missing_threshold = missing_threshold
        self.eps = eps
        self.min_samples = min_samples
        self.remove_outliers = remove_outliers
        self.z_threshold = z_threshold
        self.scaler = StandardScaler()  # StandardScaler performs z-score normalization
        
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
    
    def remove_outliers_zscore(self, df):
        """
        Remove extreme outliers using z-score method.
        
        Returns:
        --------
        pd.DataFrame : Dataframe with outliers removed
        int : Number of rows removed
        """
        if not self.remove_outliers:
            return df, 0
        
        rows_before = len(df)
        
        # Calculate z-scores for all numeric columns
        z_scores = np.abs((df - df.mean()) / df.std())
        
        # Keep rows where all z-scores are below threshold
        df_filtered = df[(z_scores < self.z_threshold).all(axis=1)]
        
        rows_removed = rows_before - len(df_filtered)
        
        return df_filtered, rows_removed
    
    def apply_dbscan(self, df):
        """
        Apply DBSCAN clustering to the dataframe with z-score normalization.
        
        Returns:
        --------
        np.array : Cluster labels
        pd.DataFrame : Original data with cluster labels
        dict : Clustering statistics
        """
        if df.empty:
            return None, None, {'error': 'Empty dataframe after cleaning'}
        
        # Apply z-score normalization (StandardScaler)
        # This transforms each feature to have mean=0 and std=1
        X_scaled = self.scaler.fit_transform(df)
        
        # Optional: Create a dataframe to show the z-scores
        z_score_df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        
        # Apply DBSCAN on normalized data
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        # Add labels to original dataframe (not z-scored)
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
            'cluster_sizes': {},
            'z_score_stats': {
                'mean_z_score': np.mean(np.abs(X_scaled)),
                'max_z_score': np.max(np.abs(X_scaled))
            }
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
        
        # Remove outliers if enabled
        df_no_outliers, outliers_removed = self.remove_outliers_zscore(df_cleaned)
        if outliers_removed > 0:
            print(f"  Removed {outliers_removed} outlier rows (z-score > {self.z_threshold})")
            cleaning_stats['outliers_removed'] = outliers_removed
        
        # Apply DBSCAN with z-score normalization
        labels, result_df, cluster_stats = self.apply_dbscan(df_no_outliers)
        print(f"  Z-score normalization applied (mean z-score: {cluster_stats['z_score_stats']['mean_z_score']:.2f})")
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
    from glob import glob
    
    # Initialize processor
    processor = DBSCANProcessor(
        missing_threshold=0.20,  # 20% missing data threshold
        eps=0.5,                 # Adjust based on your data scale after z-score normalization
        min_samples=5,           # Adjust based on your dataset size
        remove_outliers=True,    # Remove extreme outliers before clustering
        z_threshold=3.0          # Consider points with |z-score| > 3 as outliers
    )
    
    # Find all consolidated daily summary CSV files
    data_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit'
    csv_files = glob(f'{data_path}/*_consolidated_daily_summary.csv')
    
    print(f"Found {len(csv_files)} files to process")
    
    # Process all files
    output_dir = f'{data_path}/dbscan_results'
    results = processor.process_multiple_files(csv_files, output_dir=output_dir)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for filename, result in results.items():
        if 'error' not in result:
            successful += 1
            print(f"\n{filename}:")
            print(f"  Clusters: {result['cluster_stats']['n_clusters']}")
            print(f"  Noise points: {result['cluster_stats']['n_noise_points']} ({result['cluster_stats']['noise_percentage']:.1f}%)")
            print(f"  Final dimensions: {result['cleaning_stats']['final_shape']}")
        else:
            failed += 1
            print(f"\n{filename}: FAILED - {result['error']}")
    
    print(f"\n{'='*60}")
    print(f"Successfully processed: {successful}/{len(csv_files)}")
    print(f"Failed: {failed}/{len(csv_files)}")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)