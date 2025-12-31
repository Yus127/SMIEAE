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