import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import glob
import gc

def load_json_file(filepath):
    """Load and parse JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def load_csv_file(filepath):
    """Load CSV file"""
    try:
        return pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def normalize_timestamp(df, time_cols=['date', 'time', 'timestamp', 'datetime']):
    """Convert various timestamp formats to standard datetime"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # Try to find timestamp column
    for col in df.columns:
        col_lower = col.lower()
        if any(tc in col_lower for tc in time_cols):
            try:
                df['timestamp'] = pd.to_datetime(df[col], utc=True, format='mixed')
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                return df
            except:
                try:
                    df['timestamp'] = pd.to_datetime(df[col], utc=True)
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                    return df
                except:
                    continue
    
    # If no timestamp found, try combining date and time columns
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)
    
    if date_col and time_col:
        try:
            df['timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except:
            df['timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
    elif date_col:
        try:
            df['timestamp'] = pd.to_datetime(df[date_col], utc=True)
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except:
            df['timestamp'] = pd.to_datetime(df[date_col])
    
    return df

def process_files_in_folder(folder_path, pattern, source_name):
    """Process files matching a pattern in a folder"""
    files = glob.glob(os.path.join(folder_path, pattern))
    dfs = []
    
    for file in files:
        print(f"    - Loading {os.path.basename(file)}")
        if file.endswith('.json'):
            df = load_json_file(file)
        else:
            df = load_csv_file(file)
        
        if not df.empty:
            df = normalize_timestamp(df)
            if 'timestamp' in df.columns:
                df['source'] = source_name
                # Keep only essential columns to reduce memory
                cols_to_keep = ['timestamp', 'source'] + [c for c in df.columns 
                                if c not in ['timestamp', 'source'] and not c.lower() in ['date', 'time', 'datetime']]
                df = df[cols_to_keep]
                dfs.append(df)
        
        del df
        gc.collect()
    
    return dfs

def process_global_export_data(base_path):
    """Process files in Global Export Data folder"""
    dfs = []
    dfs.extend(process_files_in_folder(base_path, 'estimated_oxygen_variation-*.csv', 'oxygen_variation'))
    dfs.extend(process_files_in_folder(base_path, 'heart_rate-*.json', 'heart_rate_global'))
    dfs.extend(process_files_in_folder(base_path, 'sleep-*.json', 'sleep_global'))
    return dfs

def process_hrv_folder(base_path):
    """Process Heart Rate Variability folder"""
    dfs = []
    dfs.extend(process_files_in_folder(base_path, 'Daily Heart Rate Variability Summary*.csv', 'daily_hrv_summary'))
    dfs.extend(process_files_in_folder(base_path, 'Daily Respiratory Rate Summary*.csv', 'daily_respiratory_rate'))
    dfs.extend(process_files_in_folder(base_path, 'Heart Rate Variability Details*.csv', 'hrv_details'))
    dfs.extend(process_files_in_folder(base_path, 'Respiratory Rate Summary*.csv', 'respiratory_rate_summary'))
    return dfs

def process_spo2_folder(base_path):
    """Process Oxygen Saturation folder"""
    dfs = []
    dfs.extend(process_files_in_folder(base_path, 'Daily SpO2*.csv', 'daily_spo2'))
    dfs.extend(process_files_in_folder(base_path, 'Minute SpO2*.csv', 'minute_spo2'))
    return dfs

def process_physical_activity_folder(base_path):
    """Process Physical Activity Google Data folder"""
    dfs = []
    activity_files = {
        'active_minutes_*.csv': 'active_minutes',
        'activity_level_*.csv': 'activity_level',
        'heart_rate_*.csv': 'heart_rate_activity',
        'heart_rate_variability_*.csv': 'hrv_activity',
        'oxygen_saturation_*.csv': 'spo2_activity',
        'steps_*.csv': 'steps'
    }
    
    for pattern, source_name in activity_files.items():
        dfs.extend(process_files_in_folder(base_path, pattern, source_name))
    
    return dfs

def process_sleep_score_folder(base_path):
    """Process Sleep Score folder"""
    return process_files_in_folder(base_path, 'sleep_score.csv', 'sleep_score')

def process_stress_score_folder(base_path):
    """Process Stress Score folder"""
    return process_files_in_folder(base_path, 'Stress Score.csv', 'stress_score')

def consolidate_and_save_incremental(all_dfs, output_file):
    """Consolidate dataframes incrementally and save to avoid memory issues"""
    if not all_dfs:
        return None
    
    print("  Consolidating data incrementally...")
    
    # Group dataframes by source to reduce memory
    source_groups = {}
    for df in all_dfs:
        if df.empty or 'timestamp' not in df.columns:
            continue
        
        source = df['source'].iloc[0] if 'source' in df.columns else 'unknown'
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(df)
    
    # Process each source group
    processed_sources = {}
    daily_steps_df = None
    
    for source, dfs in source_groups.items():
        print(f"    Processing source: {source}")
        combined = pd.concat(dfs, ignore_index=True)
        combined = combined[combined['timestamp'].notna()]
        
        if not combined.empty:
            # Special handling for steps data - aggregate by day
            if source == 'steps':
                print(f"      Aggregating steps by day...")
                combined['date'] = combined['timestamp'].dt.date
                
                # Find step count column (could be 'steps', 'step_count', 'value', etc.)
                step_col = None
                for col in combined.columns:
                    if col.lower() in ['steps', 'step_count', 'value', 'count']:
                        step_col = col
                        break
                
                if step_col:
                    # Group by date and sum steps
                    daily_steps = combined.groupby('date')[step_col].sum().reset_index()
                    daily_steps['timestamp'] = pd.to_datetime(daily_steps['date']) + pd.Timedelta(hours=0, minutes=0, seconds=0)
                    daily_steps = daily_steps.rename(columns={step_col: 'daily_total_steps'})
                    daily_steps = daily_steps[['timestamp', 'daily_total_steps']]
                    daily_steps_df = daily_steps
                    print(f"      Created daily step totals: {len(daily_steps)} days")
                
                # Skip adding the original detailed steps data
                del combined, dfs
                gc.collect()
                continue
            
            # Remove duplicates within this source
            combined = combined.sort_values('timestamp')
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            
            # Rename columns to include source prefix
            cols_to_rename = [c for c in combined.columns if c not in ['timestamp', 'source']]
            rename_dict = {c: f"{source}_{c}" for c in cols_to_rename}
            combined = combined.rename(columns=rename_dict)
            combined = combined.drop(columns=['source'])
            
            processed_sources[source] = combined
        
        del combined, dfs
        gc.collect()
    
    # Add daily steps as a separate source
    if daily_steps_df is not None:
        processed_sources['daily_steps'] = daily_steps_df
    
    # Merge all sources on timestamp
    if not processed_sources:
        return None
    
    print("    Merging all sources...")
    result = list(processed_sources.values())[0]
    
    for i, df in enumerate(list(processed_sources.values())[1:], 1):
        print(f"    Merging source {i+1}/{len(processed_sources)}")
        result = pd.merge(result, df, on='timestamp', how='outer')
        del df
        gc.collect()
    
    result = result.sort_values('timestamp')
    
    # Save incrementally
    print(f"    Saving to {output_file}")
    result.to_csv(output_file, index=False)
    
    rows = len(result)
    cols = len(result.columns)
    date_range = f"{result['timestamp'].min()} to {result['timestamp'].max()}"
    
    del result
    gc.collect()
    
    return {'rows': rows, 'cols': cols, 'date_range': date_range}

def consolidate_user_data(download_folder, output_file):
    """Consolidate all data for a single download folder"""
    print(f"  Processing folder: {download_folder.name}")
    
    all_dfs = []
    
    folders = {
        'Global Export Data': process_global_export_data,
        'Heart Rate Variability': process_hrv_folder,
        'Oxygen Saturation (SpO2)': process_spo2_folder,
        'Physical Activity_GoogleData': process_physical_activity_folder,
        'Sleep Score': process_sleep_score_folder,
        'Stress Score': process_stress_score_folder
    }
    
    for folder_name, process_func in folders.items():
        folder_path = os.path.join(download_folder, folder_name)
        if os.path.exists(folder_path):
            print(f"    Processing {folder_name}...")
            dfs = process_func(folder_path)
            all_dfs.extend(dfs)
            del dfs
            gc.collect()
    
    # Consolidate and save incrementally
    stats = consolidate_and_save_incremental(all_dfs, output_file)
    
    del all_dfs
    gc.collect()
    
    return stats

def main():
    """Main execution function"""
    base_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit"
    base_dir = Path(base_dir)
    
    user_folders = [item for item in base_dir.iterdir() if item.is_dir()]
    
    if not user_folders:
        print("No user folders found!")
        return
    
    print(f"Found {len(user_folders)} user folders")
    
    output_dir = base_dir / 'consolidated_output'
    output_dir.mkdir(exist_ok=True)
    
    for user_folder in user_folders:
        if user_folder.name == 'consolidated_output':
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing User: {user_folder.name}")
        print(f"{'='*60}")
        
        download_folders = [item for item in user_folder.iterdir() if item.is_dir()]
        
        if not download_folders:
            print(f"  No download folders found")
            continue
        
        print(f"  Found {len(download_folders)} download folders")
        
        # Create temporary files for each download folder
        temp_files = []
        for i, download_folder in enumerate(download_folders):
            temp_file = output_dir / f"temp_user_{user_folder.name}_download_{i}.csv"
            stats = consolidate_user_data(download_folder, temp_file)
            
            if stats:
                temp_files.append(temp_file)
                print(f"    ✓ Processed: {stats['rows']} rows, {stats['cols']} columns")
            
            gc.collect()
        
        # Combine all temporary files for this user
        if temp_files:
            print(f"\n  Combining {len(temp_files)} download folders...")
            output_file = output_dir / f"user_{user_folder.name}_consolidated.csv"
            
            dfs = []
            for temp_file in temp_files:
                print(f"    Loading {temp_file.name}")
                df = pd.read_csv(temp_file, low_memory=False)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                dfs.append(df)
                os.remove(temp_file)  # Clean up temp file
            
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.sort_values('timestamp')
            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
            
            combined.to_csv(output_file, index=False)
            
            print(f"\n  ✓ Final file saved: {output_file}")
            print(f"    Total Rows: {len(combined)}, Columns: {len(combined.columns)}")
            print(f"    Date Range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
            
            del dfs, combined
            gc.collect()
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete! Output files in: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()