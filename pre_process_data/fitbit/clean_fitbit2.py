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
    
    # Special handling for heart_rate_activity_beats per minute column
    hr_activity_col = 'heart_rate_activity_beats per minute'
    if hr_activity_col in df.columns:
        agg_dict[f'{hr_activity_col}_mean'] = (hr_activity_col, 'mean')
        agg_dict[f'{hr_activity_col}_std'] = (hr_activity_col, 'std')
        agg_dict[f'{hr_activity_col}_min'] = (hr_activity_col, 'min')
        agg_dict[f'{hr_activity_col}_max'] = (hr_activity_col, 'max')
    
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
    
    # SpO2 metrics - filter out values of 50 (likely sensor errors)
    spo2_cols = [col for col in df.columns if 'spo2' in col.lower() and 'value' in col.lower()]
    for col in spo2_cols:
        if df[col].notna().any():
            # Create a filtered version excluding values of 50
            filtered_col = f'{col}_filtered'
            df[filtered_col] = df[col].apply(lambda x: x if pd.notna(x) and x != 50 else np.nan)
            
            agg_dict[f'{col}_mean'] = (filtered_col, 'mean')
            agg_dict[f'{col}_std'] = (filtered_col, 'std')
            agg_dict[f'{col}_min'] = (filtered_col, 'min')
    
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
        
        # Clean up: remove the temporary filtered SpO2 columns from the result
        filtered_cols = [col for col in daily_agg.columns if '_filtered' in col]
        daily_agg = daily_agg.drop(columns=filtered_cols, errors='ignore')
        
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
    Extract comprehensive sleep metrics from sleep data.
    
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
                    
                    # Extract from 'summary'
                    if 'summary' in levels_data:
                        summary = levels_data['summary']
                        
                        # Deep sleep metrics
                        if 'deep' in summary:
                            record['deep_sleep_minutes'] = summary['deep'].get('minutes', np.nan)
                            record['deep_sleep_count'] = summary['deep'].get('count', np.nan)
                            record['deep_sleep_30day_avg'] = summary['deep'].get('thirtyDayAvgMinutes', np.nan)
                            # Deviation from 30-day average
                            if summary['deep'].get('minutes') and summary['deep'].get('thirtyDayAvgMinutes'):
                                record['deep_sleep_deviation_from_avg'] = summary['deep']['minutes'] - summary['deep']['thirtyDayAvgMinutes']
                        
                        # Light sleep metrics
                        if 'light' in summary:
                            record['light_sleep_minutes'] = summary['light'].get('minutes', np.nan)
                            record['light_sleep_count'] = summary['light'].get('count', np.nan)
                            record['light_sleep_30day_avg'] = summary['light'].get('thirtyDayAvgMinutes', np.nan)
                            if summary['light'].get('minutes') and summary['light'].get('thirtyDayAvgMinutes'):
                                record['light_sleep_deviation_from_avg'] = summary['light']['minutes'] - summary['light']['thirtyDayAvgMinutes']
                        
                        # REM sleep metrics
                        if 'rem' in summary:
                            record['rem_sleep_minutes'] = summary['rem'].get('minutes', np.nan)
                            record['rem_sleep_count'] = summary['rem'].get('count', np.nan)
                            record['rem_sleep_30day_avg'] = summary['rem'].get('thirtyDayAvgMinutes', np.nan)
                            if summary['rem'].get('minutes') and summary['rem'].get('thirtyDayAvgMinutes'):
                                record['rem_sleep_deviation_from_avg'] = summary['rem']['minutes'] - summary['rem']['thirtyDayAvgMinutes']
                        
                        # Wake metrics
                        if 'wake' in summary:
                            record['wake_count'] = summary['wake'].get('count', np.nan)
                            record['wake_minutes'] = summary['wake'].get('minutes', np.nan)
                            record['wake_30day_avg'] = summary['wake'].get('thirtyDayAvgMinutes', np.nan)
                            if summary['wake'].get('minutes') and summary['wake'].get('thirtyDayAvgMinutes'):
                                record['wake_deviation_from_avg'] = summary['wake']['minutes'] - summary['wake']['thirtyDayAvgMinutes']
                        
                        # Restless metrics (if present)
                        if 'restless' in summary:
                            record['restless_count'] = summary['restless'].get('count', np.nan)
                            record['restless_minutes'] = summary['restless'].get('minutes', np.nan)
                        
                        # Asleep metrics (if present)
                        if 'asleep' in summary:
                            record['asleep_count'] = summary['asleep'].get('count', np.nan)
                            record['asleep_minutes'] = summary['asleep'].get('minutes', np.nan)
                    
                    # Extract from 'data' (main sleep stage transitions)
                    if 'data' in levels_data and levels_data['data']:
                        data = levels_data['data']
                        
                        # Total number of sleep stage transitions
                        record['sleep_stage_transitions'] = len(data)
                        
                        # Find time to first deep sleep and first REM
                        sleep_start_time = pd.to_datetime(data[0]['dateTime']) if data else None
                        
                        first_deep_time = None
                        first_rem_time = None
                        
                        for segment in data:
                            seg_time = pd.to_datetime(segment['dateTime'])
                            if segment['level'] == 'deep' and first_deep_time is None:
                                first_deep_time = seg_time
                            if segment['level'] == 'rem' and first_rem_time is None:
                                first_rem_time = seg_time
                        
                        if sleep_start_time and first_deep_time:
                            record['minutes_to_first_deep_sleep'] = (first_deep_time - sleep_start_time).total_seconds() / 60
                        
                        if sleep_start_time and first_rem_time:
                            record['minutes_to_first_rem_sleep'] = (first_rem_time - sleep_start_time).total_seconds() / 60
                        
                        # Calculate average segment duration per sleep stage
                        stage_durations = {'deep': [], 'light': [], 'rem': [], 'wake': []}
                        for segment in data:
                            level = segment['level']
                            if level in stage_durations:
                                stage_durations[level].append(segment['seconds'] / 60)  # Convert to minutes
                        
                        for stage, durations in stage_durations.items():
                            if durations:
                                record[f'{stage}_avg_segment_duration_minutes'] = np.mean(durations)
                                record[f'{stage}_segment_duration_std'] = np.std(durations) if len(durations) > 1 else 0
                        
                        # Last sleep stage before waking
                        if data:
                            record['last_sleep_stage'] = data[-1]['level']
                    
                    # Extract from 'shortData' (micro-awakenings)
                    if 'shortData' in levels_data and levels_data['shortData']:
                        short_data = levels_data['shortData']
                        
                        # Count micro-awakenings
                        record['micro_awakening_count'] = len(short_data)
                        
                        # Total duration of micro-awakenings
                        total_micro_wake_seconds = sum(segment['seconds'] for segment in short_data)
                        record['micro_awakening_total_minutes'] = total_micro_wake_seconds / 60
                        
                        # Calculate micro-awakening frequency per hour
                        if 'data' in levels_data and levels_data['data']:
                            total_sleep_duration_hours = sum(seg['seconds'] for seg in levels_data['data']) / 3600
                            if total_sleep_duration_hours > 0:
                                record['micro_awakening_per_hour'] = len(short_data) / total_sleep_duration_hours
                        
                        # Average micro-awakening duration
                        if short_data:
                            record['micro_awakening_avg_duration_seconds'] = np.mean([seg['seconds'] for seg in short_data])
                            record['micro_awakening_max_duration_seconds'] = max(seg['seconds'] for seg in short_data)
                    
                except Exception as e:
                    # If parsing fails, just add the date
                    print(f"Warning: Could not parse sleep levels for date {record['date']}: {str(e)}")
            
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
    # Columns to remove from the dataset
    columns_to_remove = [
        'active_minutes_data source',
        'activity_level_data source',
        'heart_rate_activity_data source',
        'spo2_activity_data source',
        'hrv_activity_data source',
        'heart_rate_activity_heart rate notification type',
        'heart_rate_activity_heart rate threshold beats per minute',
        'heart_rate_activity_heart rate trigger value beats per minute'
    ]
    
    df = pd.read_csv(input_path, low_memory=False)
    
    # Remove unwanted columns if they exist
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
    
    # Remove any columns containing "stress" (case-insensitive)
    stress_columns = [col for col in df.columns if 'stress' in col.lower()]
    df = df.drop(columns=stress_columns, errors='ignore')
    
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
    
    # Remove any columns containing "stress" from the final daily summary (case-insensitive)
    stress_columns_daily = [col for col in daily_df.columns if 'stress' in col.lower()]
    daily_df = daily_df.drop(columns=stress_columns_daily, errors='ignore')
    
    # Remove active_minutes columns (they contain mostly zeros and are redundant)
    active_minutes_cols = [
        'active_minutes_light_total',
        'active_minutes_moderate_total', 
        'active_minutes_very_total',
        'respiratory_rate_summary_deep_sleep_breathing_rate_std',
        'respiratory_rate_summary_rem_sleep_signal_to_noise_std',
        'respiratory_rate_summary_rem_sleep_breathing_rate_std',
        'respiratory_rate_summary_light_sleep_signal_to_noise_std',
        'respiratory_rate_summary_light_sleep_breathing_rate_std',
        'respiratory_rate_summary_deep_sleep_signal_to_noise_std',
        'respiratory_rate_summary_full_sleep_breathing_rate_std',
        'respiratory_rate_summary_full_sleep_signal_to_noise_std',
        'asleep_minutes',
        'asleep_count',
        'restless_minutes',
        'restless_count'


    ]
    daily_df = daily_df.drop(columns=[col for col in active_minutes_cols if col in daily_df.columns], errors='ignore')
    
    # Remove duplicate 'date' columns if they exist (keep only one)
    date_cols = [col for col in daily_df.columns if col == 'date']
    if len(date_cols) > 1:
        # Keep the first one, remove duplicates
        daily_df = daily_df.loc[:, ~daily_df.columns.duplicated()]
    
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


if __name__ == "__main__":
    process_multiple_csvs("/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output", output_dir="/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit")