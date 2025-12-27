import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_daily_aggregations(df):
    """
    Calculate daily aggregated metrics from time-series data.
    
    Args:
        df: DataFrame with timestamp and time-series columns
    
    Returns:
        DataFrame with daily aggregated metrics
    """
    # Parse timestamp and extract date - handle mixed formats
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df['date'] = df['timestamp'].dt.date
    
    # Initialize aggregation dictionary
    agg_dict = {}
    
    # Heart Rate metrics
    hr_cols = [col for col in df.columns if 'heart_rate' in col.lower() and 'bpm' in str(df[col].dtype)]
    if hr_cols:
        for col in hr_cols:
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
            agg_dict[f'{col}_min'] = (col, 'min')
            agg_dict[f'{col}_max'] = (col, 'max')
    
    # HRV metrics (RMSSD)
    rmssd_cols = [col for col in df.columns if 'rmssd' in col.lower()]
    for col in rmssd_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
            agg_dict[f'{col}_min'] = (col, 'min')
            agg_dict[f'{col}_max'] = (col, 'max')
    
    # HRV frequency domain metrics
    lf_cols = [col for col in df.columns if 'low_frequency' in col.lower()]
    hf_cols = [col for col in df.columns if 'high_frequency' in col.lower()]
    
    for col in lf_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
    
    for col in hf_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
    
    # Calculate LF/HF ratio if both exist
    if lf_cols and hf_cols:
        for lf_col, hf_col in zip(lf_cols, hf_cols):
            ratio_col = f'lf_hf_ratio_{lf_col.split("_")[0]}'
            df[ratio_col] = df[lf_col] / df[hf_col]
            agg_dict[f'{ratio_col}_mean'] = (ratio_col, 'mean')
            agg_dict[f'{ratio_col}_std'] = (ratio_col, 'std')
    
    # Respiratory rate metrics
    resp_cols = [col for col in df.columns if 'respiratory' in col.lower() or 'breathing' in col.lower()]
    for col in resp_cols:
        if df[col].notna().any() and 'standard_deviation' not in col.lower():
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
    
    # SpO2 metrics
    spo2_cols = [col for col in df.columns if 'spo2' in col.lower() and 'value' in col.lower()]
    for col in spo2_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
            agg_dict[f'{col}_min'] = (col, 'min')
    
    # Steps
    steps_cols = [col for col in df.columns if 'steps' in col.lower() and 'steps' == col.lower().split('_')[-1]]
    for col in steps_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_total'] = (col, 'sum')
    
    # Activity minutes
    activity_cols = [col for col in df.columns if 'active_minutes' in col.lower()]
    for col in activity_cols:
        if df[col].notna().any():
            agg_dict[f'{col}_total'] = (col, 'sum')
    
    # Activity level counts
    level_cols = [col for col in df.columns if 'activity_level' in col.lower() or col.lower() == 'level']
    if level_cols:
        for col in level_cols:
            if df[col].notna().any():
                # Count occurrences of each activity level
                activity_counts = df.groupby('date')[col].value_counts().unstack(fill_value=0)
                activity_counts.columns = [f'activity_level_{level.lower()}_count' for level in activity_counts.columns]
    
    # Perform aggregation
    if agg_dict:
        daily_agg = df.groupby('date').agg(**agg_dict).reset_index()
        
        # Add activity level counts if available
        if level_cols:
            for col in level_cols:
                if df[col].notna().any():
                    activity_counts = df.groupby('date')[col].value_counts().unstack(fill_value=0)
                    activity_counts.columns = [f'activity_level_{str(level).lower()}_count' for level in activity_counts.columns]
                    activity_counts = activity_counts.reset_index()
                    daily_agg = daily_agg.merge(activity_counts, on='date', how='left')
        
        return daily_agg
    else:
        return pd.DataFrame({'date': df['date'].unique()})


def extract_sleep_metrics(df, midnight_mask):
    """
    Extract additional sleep metrics from sleep data.
    
    Args:
        df: DataFrame with sleep data
        midnight_mask: Boolean mask for midnight timestamps
    
    Returns:
        DataFrame with sleep metrics per day
    """
    # Get midnight rows
    midnight_df = df[midnight_mask].copy()
    
    if midnight_df.empty or 'timestamp' not in midnight_df.columns:
        return None
    
    midnight_df['date'] = pd.to_datetime(midnight_df['timestamp']).dt.date
    
    # Initialize list to store sleep metrics for each row
    sleep_records = []
    
    # Extract sleep stage durations from levels summary
    levels_col = [col for col in midnight_df.columns if 'levels' in col.lower()]
    
    if levels_col:
        for idx, row in midnight_df.iterrows():
            record = {'date': row['date']}
            
            if pd.notna(row[levels_col[0]]):
                try:
                    levels_data = eval(row[levels_col[0]])
                    if 'summary' in levels_data:
                        summary = levels_data['summary']
                        
                        if 'deep' in summary:
                            record['deep_sleep_minutes'] = summary['deep'].get('minutes', np.nan)
                            record['deep_sleep_count'] = summary['deep'].get('count', np.nan)
                        
                        if 'light' in summary:
                            record['light_sleep_minutes'] = summary['light'].get('minutes', np.nan)
                            record['light_sleep_count'] = summary['light'].get('count', np.nan)
                        
                        if 'rem' in summary:
                            record['rem_sleep_minutes'] = summary['rem'].get('minutes', np.nan)
                            record['rem_sleep_count'] = summary['rem'].get('count', np.nan)
                        
                        if 'wake' in summary:
                            record['wake_count'] = summary['wake'].get('count', np.nan)
                            record['wake_minutes'] = summary['wake'].get('minutes', np.nan)
                except Exception as e:
                    # If parsing fails, just add the date
                    pass
            
            sleep_records.append(record)
    
    if sleep_records:
        return pd.DataFrame(sleep_records)
    return None


def process_csv_file(input_path, output_dir="processed_data"):
    """
    Process a single CSV file to separate daily summary columns from time-series data
    and calculate additional daily aggregations.
    
    Args:
        input_path: Path to the input CSV file
        output_dir: Directory to save processed files
    """
    # Read the CSV with low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_path, low_memory=False)
    
    # Parse timestamp column (assuming it's the first column)
    # Use format='mixed' to handle different timestamp formats
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='mixed', errors='coerce')
    
    # Identify rows where time is exactly 00:00:00
    midnight_mask = df['timestamp'].dt.time == pd.Timestamp('00:00:00').time()
    
    # Find columns that only have data at midnight (00:00:00)
    daily_columns = ['timestamp']
    timeseries_columns = ['timestamp']
    
    for col in df.columns:
        if col == 'timestamp' or col == df.columns[0]:
            continue
            
        # Check if column has non-null values
        has_data = df[col].notna()
        
        if has_data.any():
            # Check if all non-null values occur only at midnight
            non_null_at_midnight = (has_data & midnight_mask).sum()
            non_null_total = has_data.sum()
            
            if non_null_at_midnight == non_null_total:
                daily_columns.append(col)
            else:
                timeseries_columns.append(col)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(input_path).stem
    
    # Create daily summary DataFrame from midnight data
    daily_df = df[midnight_mask][daily_columns].copy()
    daily_df = daily_df.dropna(how='all', subset=[col for col in daily_columns if col != 'timestamp'])
    
    # Add date column
    daily_df['date'] = pd.to_datetime(daily_df['timestamp']).dt.date
    
    # Extract additional sleep metrics
    sleep_metrics_df = extract_sleep_metrics(df, midnight_mask)
    if sleep_metrics_df is not None:
        daily_df = daily_df.merge(sleep_metrics_df, on='date', how='left')
    
    # Calculate daily aggregations from time-series data
    timeseries_df_full = df[timeseries_columns].copy()
    daily_agg = calculate_daily_aggregations(timeseries_df_full)
    
    # Merge aggregated metrics with daily summary
    if not daily_agg.empty:
        daily_df = daily_df.merge(daily_agg, on='date', how='outer')
    
    # Create time-series DataFrame (without daily-only columns)
    timeseries_df = df[timeseries_columns].copy()
    timeseries_df = timeseries_df.dropna(how='all', subset=[col for col in timeseries_columns if col != 'timestamp'])
    
    # Save the files
    daily_output = os.path.join(output_dir, f"{base_name}_daily_summary.csv")
    timeseries_output = os.path.join(output_dir, f"{base_name}_timeseries.csv")
    
    daily_df.to_csv(daily_output, index=False)
    timeseries_df.to_csv(timeseries_output, index=False)
    
    print(f"Processed: {input_path}")
    print(f"  - Daily summary: {daily_output} ({len(daily_df.columns)} columns, {len(daily_df)} rows)")
    print(f"  - Time-series: {timeseries_output} ({len(timeseries_columns)-1} columns, {len(timeseries_df)} rows)")
    print(f"  - Added {len(daily_agg.columns)-1 if not daily_agg.empty else 0} aggregated metrics")
    print()
    
    return daily_output, timeseries_output


def process_multiple_csvs(input_dir, output_dir="processed_data"):
    """
    Process all CSV files in a directory.
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Directory to save processed files
    """
    csv_files = list(Path(input_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process\n")
    
    for csv_file in csv_files:
        try:
            process_csv_file(str(csv_file), output_dir)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}\n")
    
    print("Processing complete!")


# Example usage:
if __name__ == "__main__":
    # Option 1: Process a single file
    # process_csv_file("your_file.csv", output_dir="processed_data")
    
    # Option 2: Process all CSV files in a directory
    # process_multiple_csvs("path/to/your/csv/directory", output_dir="processed_data")
    
    # Example with current directory
    process_multiple_csvs("/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output", output_dir="/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit")