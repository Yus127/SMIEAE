import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class FitbitDBSCANProcessor:
    """
    DBSCAN processor for specific Fitbit health variables with forward-fill imputation.
    """
    
    def __init__(self, eps=0.5, min_samples=5, auto_tune=False):
        """
        Initialize the processor.
        
        Parameters:
        -----------
        eps : float
            DBSCAN epsilon parameter (maximum distance between two samples)
        min_samples : int
            DBSCAN minimum samples in a neighborhood for a core point
        auto_tune : bool
            If True, automatically find optimal eps using k-distance graph
        """
        self.eps = eps
        self.min_samples = min_samples
        self.auto_tune = auto_tune
        self.scaler = StandardScaler()  # For z-score normalization
        
        # Define the specific variables we want to use
        self.target_columns = [
             'hrv_details_rmssd_mean',
            'daily_hrv_summary_rmssd',
            'daily_spo2_average_value',
            'heart_rate_activity_beats per minute_mean',
            'heart_rate_activity_beats per minute_std',
            'minute_spo2_value_mean'
        
        ]
        """    'hrv_details_rmssd_mean',
            'daily_hrv_summary_rmssd',
            'daily_spo2_average_value',
            'heart_rate_activity_beats per minute_mean',
            'heart_rate_activity_beats per minute_std',
            'minute_spo2_value_mean'


            'sleep_global_minutesAsleep',
            'sleep_global_efficiency',
            'sleep_global_minutesAwake',
            'daily_respiratory_rate_daily_respiratory_rate',
            'daily_total_steps',
            
            """
    
    def load_and_prepare_data(self, filepath):
        """
        Load CSV and prepare the specific variables with forward-fill imputation.
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe with selected variables
        dict : Statistics about the preparation process
        """
        df = pd.read_csv(filepath)
        original_shape = df.shape
        
        stats = {
            'original_rows': original_shape[0],
            'original_cols': original_shape[1],
            'columns_found': [],
            'columns_missing': [],
            'missing_values_before': {},
            'missing_values_after': {},
            'rows_dropped': 0
        }
        
        # Check which columns exist in the dataframe
        available_cols = []
        for col in self.target_columns:
            if col in df.columns:
                available_cols.append(col)
                stats['columns_found'].append(col)
                stats['missing_values_before'][col] = df[col].isnull().sum()
            else:
                stats['columns_missing'].append(col)
        
        if not available_cols:
            return pd.DataFrame(), stats
        
        # Select only the available target columns
        df_selected = df[available_cols].copy()
        
        # Fill missing values with forward fill (use previous values)
        # Then backward fill for any remaining NaN at the start
        df_filled = df_selected.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN values remain (entire column is NaN), fill with column mean
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                df_filled[col].fillna(df_filled[col].mean(), inplace=True)
        
        # Track missing values after imputation
        for col in available_cols:
            stats['missing_values_after'][col] = df_filled[col].isnull().sum()
        
        # Drop any remaining rows with NaN (should be rare after the above steps)
        rows_before = len(df_filled)
        df_final = df_filled.dropna()
        stats['rows_dropped'] = rows_before - len(df_final)
        stats['final_shape'] = df_final.shape
        stats['columns_used'] = len(available_cols)
        
        return df_final, stats
    
    def find_optimal_eps(self, X_normalized, k=None):
        """
        Find optimal eps using k-distance graph (elbow method).
        
        Parameters:
        -----------
        X_normalized : array
            Normalized data
        k : int
            Number of neighbors (if None, uses min_samples)
            
        Returns:
        --------
        float : Suggested eps value
        """
        if k is None:
            k = self.min_samples
        
        # Calculate k-nearest neighbors distances
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X_normalized)
        distances, indices = neighbors.kneighbors(X_normalized)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find the elbow point (point of maximum curvature)
        # Simple heuristic: use 90th percentile of distances
        suggested_eps = np.percentile(distances, 90)
        
        return suggested_eps, distances
    
    def apply_dbscan_with_zscore(self, df):
        """
        Apply z-score normalization and DBSCAN clustering.
        
        Returns:
        --------
        np.array : Cluster labels
        pd.DataFrame : Original data with cluster labels and z-scores
        dict : Clustering statistics
        """
        if df.empty:
            return None, None, {'error': 'Empty dataframe'}
        
        # Apply z-score normalization (mean=0, std=1)
        X_normalized = self.scaler.fit_transform(df)
        
        # Create dataframe with z-scores for reference
        z_score_df = pd.DataFrame(
            X_normalized, 
            columns=[f'{col}_zscore' for col in df.columns],
            index=df.index
        )
        
        # Apply DBSCAN on normalized data
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X_normalized)
        
        # Combine original data with z-scores and cluster labels
        result_df = df.copy()
        result_df = pd.concat([result_df, z_score_df], axis=1)
        result_df['cluster'] = labels
        
        # Calculate clustering statistics
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_percentage': (n_noise / len(labels)) * 100 if len(labels) > 0 else 0,
            'total_points': len(labels),
            'cluster_sizes': {},
            'normalization_stats': {}
        }
        
        # Cluster size distribution
        for label in unique_labels:
            if label != -1:
                count = list(labels).count(label)
                stats['cluster_sizes'][f'cluster_{label}'] = count
        
        # Normalization statistics for each feature
        for i, col in enumerate(df.columns):
            stats['normalization_stats'][col] = {
                'original_mean': df[col].mean(),
                'original_std': df[col].std(),
                'z_score_mean': X_normalized[:, i].mean(),
                'z_score_std': X_normalized[:, i].std()
            }
        
        return labels, result_df, stats
    
    def process_single_file(self, filepath, output_dir=None):
        """
        Complete pipeline for a single CSV file.
        
        Returns:
        --------
        dict : Complete results including data and statistics
        """
        filename = Path(filepath).name
        print(f"\nProcessing: {filename}")
        
        try:
            # Load and prepare data
            df_prepared, prep_stats = self.load_and_prepare_data(filepath)
            
            if df_prepared.empty:
                print(f"  Warning: No usable data found")
                return {'error': 'No usable data', 'prep_stats': prep_stats}
            
            print(f"  Variables found: {prep_stats['columns_used']}/{len(self.target_columns)}")
            print(f"  Data shape: {prep_stats['final_shape']}")
            
            # Show imputation details
            total_imputed = sum(prep_stats['missing_values_before'].values())
            if total_imputed > 0:
                print(f"  Imputed {total_imputed} missing values using forward-fill method")
            
            # Apply DBSCAN with z-score normalization
            labels, result_df, cluster_stats = self.apply_dbscan_with_zscore(df_prepared)
            
            if cluster_stats is None or 'error' in cluster_stats:
                error_msg = cluster_stats.get('error', 'Unknown error') if cluster_stats else 'Clustering failed'
                print(f"  Error during clustering: {error_msg}")
                return {'error': error_msg, 'prep_stats': prep_stats}
            
            print(f"  Z-score normalization: Applied (all features scaled)")
            eps_val = cluster_stats.get('eps_used', self.eps)
            min_samp_val = cluster_stats.get('min_samples_used', self.min_samples)
            print(f"  DBSCAN parameters: eps={eps_val:.3f}, min_samples={min_samp_val}")
            print(f"  Clusters found: {cluster_stats['n_clusters']}")
            print(f"  Noise points: {cluster_stats['n_noise_points']} ({cluster_stats['noise_percentage']:.1f}%)")
            
            if cluster_stats['cluster_sizes']:
                print(f"  Cluster distribution: {cluster_stats['cluster_sizes']}")
            elif cluster_stats['n_clusters'] == 0:
                print(f"    WARNING: All points classified as noise. Try increasing eps or decreasing min_samples.")
            
        except Exception as e:
            import traceback
            print(f"  Exception occurred: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            file_stem = Path(filepath).stem
            
            # Save clustered data with z-scores
            result_df.to_csv(output_path / f"{file_stem}_clustered.csv", index=False)
            
            # Save detailed statistics
            stats_summary = {
                'Preparation': prep_stats,
                'Clustering': cluster_stats
            }
            
            # Create a readable stats file
            with open(output_path / f"{file_stem}_stats.txt", 'w') as f:
                f.write(f"Analysis Report for {filename}\n")
                f.write("="*60 + "\n\n")
                
                f.write("DATA PREPARATION\n")
                f.write(f"Original dimensions: {prep_stats['original_rows']} rows x {prep_stats['original_cols']} cols\n")
                f.write(f"Variables used: {prep_stats['columns_used']}\n")
                f.write(f"Variables found: {', '.join(prep_stats['columns_found'])}\n")
                if prep_stats['columns_missing']:
                    f.write(f"Variables missing: {', '.join(prep_stats['columns_missing'])}\n")
                f.write(f"\nMissing values imputed:\n")
                for col, count in prep_stats['missing_values_before'].items():
                    f.write(f"  {col}: {count}\n")
                
                f.write(f"\n\nCLUSTERING RESULTS\n")
                f.write(f"DBSCAN parameters: eps={cluster_stats.get('eps_used', 'N/A')}, min_samples={cluster_stats.get('min_samples_used', 'N/A')}\n")
                f.write(f"Number of clusters: {cluster_stats['n_clusters']}\n")
                f.write(f"Noise points: {cluster_stats['n_noise_points']} ({cluster_stats['noise_percentage']:.1f}%)\n")
                f.write(f"Total points: {cluster_stats['total_points']}\n")
                
                if cluster_stats['cluster_sizes']:
                    f.write(f"\nCluster sizes:\n")
                    for cluster, size in cluster_stats['cluster_sizes'].items():
                        f.write(f"  {cluster}: {size} points\n")
                
                f.write(f"\n\nNORMALIZATION DETAILS (Z-SCORES)\n")
                for var, norm_stats in cluster_stats['normalization_stats'].items():
                    f.write(f"\n{var}:\n")
                    f.write(f"  Original: mean={norm_stats['original_mean']:.2f}, std={norm_stats['original_std']:.2f}\n")
                    f.write(f"  Z-score: mean={norm_stats['z_score_mean']:.4f}, std={norm_stats['z_score_std']:.4f}\n")
        
        return {
            'data': result_df,
            'labels': labels,
            'prep_stats': prep_stats,
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


# Main execution
if __name__ == "__main__":
    # Initialize processor with auto-tuning enabled
    processor = FitbitDBSCANProcessor(
        eps=2.0,            # Starting eps (will be auto-tuned if auto_tune=True)
        min_samples=3,      # Lower min_samples for smaller datasets (try 2-5)
        auto_tune=True      # Automatically find optimal eps
    )
    
    # Find all consolidated daily summary CSV files
    data_path = '/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit'
    csv_files = glob(f'{data_path}/*_consolidated_daily_summary.csv')
    
    print(f"Found {len(csv_files)} files to process")
    print(f"\nDBSCAN Configuration:")
    print(f"  Initial eps: {processor.eps}")
    print(f"  Min samples: {processor.min_samples}")
    print(f"  Auto-tune: {processor.auto_tune}")
    print(f"\nTarget variables:")
    for i, var in enumerate(processor.target_columns, 1):
        print(f"  {i}. {var}")
    
    # Process all files
    output_dir = f'{data_path}/dbscan_results2'
    results = processor.process_multiple_files(csv_files, output_dir=output_dir)
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    
    successful = 0
    failed = 0
    total_clusters = 0
    
    for filename, result in results.items():
        if 'error' not in result:
            successful += 1
            n_clusters = result['cluster_stats']['n_clusters']
            total_clusters += n_clusters
            print(f"\n{filename}:")
            print(f"   Clusters: {n_clusters}")
            print(f"   Noise: {result['cluster_stats']['noise_percentage']:.1f}%")
            print(f"   Variables: {result['prep_stats']['columns_used']}")
        else:
            failed += 1
            print(f"\n{filename}:  FAILED - {result['error']}")
    
    print(f"\n{'='*60}")
    print(f"Successfully processed: {successful}/{len(csv_files)}")
    print(f"Failed: {failed}/{len(csv_files)}")
    print(f"Average clusters per file: {total_clusters/successful:.1f}" if successful > 0 else "N/A")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - *_clustered.csv: Original data + z-scores + cluster labels")
    print(f"  - *_stats.txt: Detailed analysis report")


    ## WHAT IF I CHOOSE THE VARIABLES AND DO THE DBSCAN IN THE STRESS LEVELS
    """
    Para DBSCAN (outliers) necesitas **pocas variables (4–10 máximo)**, **bien escaladas**, y en tu caso casi siempre **por usuario** (o normalizadas por usuario). Si metes las ~100 columnas, DBSCAN va a fallar o va a detectar “missingness” en vez de eventos reales.

Abajo te dejo **tres conjuntos de variables** (elige según el tipo de outlier que quieres reportar). En la práctica, yo usaría **A + C** y luego cruzaría resultados.

---

## A) DBSCAN para outliers de **estrés/ansiedad** (eventos psicológicos)

Úsalo cuando quieras detectar “días raros” en los reportes (0–100).

**Variables (recomendado):**

1. `stress`
2. `anxiety`
3. `stress_delta = stress(t) - stress(t-1)`
4. `anxiety_delta = anxiety(t) - anxiety(t-1)`

**Notas:**

* Corre DBSCAN **por usuario**.
* Antes de DBSCAN: estandariza por usuario (z-score) para que “alto estrés” de un alumno no se compare con “alto estrés” de otro.
* Este set casi no sufre por faltantes porque tus respuestas son relativamente completas.

---

## B) DBSCAN biométrico “denso” (cubre la mayoría de días)

Sirve para detectar días físicamente raros incluso cuando no hay sueño/HRV.

**Variables (elige 6–8):**

* `daily_total_steps`
* `activity_level_sedentary_count`
* `activity_level_lightly_active_count`
* `activity_level_moderately_active_count`
* `activity_level_very_active_count`
* `heart_rate_activity_beats per minute_mean`
* `heart_rate_activity_beats per minute_std`
  (opcional) `heart_rate_activity_beats per minute_max`

**Por qué este set:** suele tener mucha cobertura (casi diario) y DBSCAN funciona mejor.

**Importante:** crea un proxy de uso del wearable:

* `wear_time_proxy = sedentary + lightly + moderately + very_active`
  y si `wear_time_proxy` es muy bajo, **no llames outlier a ese día** (es día de baja calidad).

---

## C) DBSCAN biométrico “recovery nocturno” (mejor señal, menos días)

Este es el más útil para relacionar fisiología con estrés/ansiedad, pero solo en días con datos de sueño/HRV/resp confiables.

**Variables (elige 6–9):**
**Sueño (calidad/cantidad):**

* `sleep_global_minutesAsleep`
* `sleep_global_efficiency`
* `sleep_global_minutesAwake`  *(o `micro_awakening_per_hour` si está disponible suficiente)*
* `sleep_global_minutesToFallAsleep`

**HRV / autonómico:**

* `daily_hrv_summary_rmssd`
* `daily_hrv_summary_nremhr`  *(o ambos si puedes)*

**Respiración:**

* `daily_respiratory_rate_daily_respiratory_rate`
  (o alternativamente `respiratory_rate_summary_full_sleep_breathing_rate_mean`)

**Actividad (control):**

* `daily_total_steps`  **o** `wear_time_proxy` (uno de los dos)

**Filtro recomendado (antes de correr DBSCAN):**

* usar solo días con `has_sleep=1` y `has_hrv=1` y `has_resp=1` (según el set que uses)
* y `wear_time_proxy` arriba de un umbral razonable

---

## SpO₂: úsalo con cautela

En muchos datasets Fitbit, `minute_spo2_value_min` puede ser un “floor” constante o artefacto.
Si no has verificado que **varía** y tiene rangos realistas, yo haría:

* **No incluir SpO₂** en DBSCAN biométrico, o
* usar solo `daily_spo2_average_value` (si pasa control de calidad), nunca el `min` si es sospechoso.

---

## Regla de oro para DBSCAN con tus datos

* **No mezcles usuarios sin normalizar.** Mejor: DBSCAN **por usuario**.
* **Escala siempre** (StandardScaler o RobustScaler) antes de DBSCAN.
* **Pocas variables** (4–10). Si quieres más señal, usa PCA a 3–5 componentes y corre DBSCAN en esos componentes.

---

## Recomendación final (lo más defendible en tu reporte)

1. Detecta outliers en el **espacio mental** con Set A.
2. Detecta outliers en **recovery** con Set C (solo días de alta calidad).
3. Reporta “outliers de alta confianza” como la intersección:
   **outlier_high_conf = outlier_A AND outlier_C**.

Si me dices si vas a correr DBSCAN **por usuario** (recomendado) o global con z-score por usuario, te doy el esquema exacto de preprocesamiento (incluyendo cómo alinear `dateOfSleep` con el estrés reportado al final del día) y los parámetros iniciales (`min_samples`, estrategia para `eps`).
"""