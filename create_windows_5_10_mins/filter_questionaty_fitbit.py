import pandas as pd
import numpy as np
from datetime import timedelta
import os
import glob
import warnings

# Suppress the DtypeWarning
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Define paths
fitbit_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output"
questionnaire_path = "/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"
output_dir = "/Users/YusMolina/Downloads/smieae/data/processed/5_10min"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Columns to delete from Fitbit data
columns_to_delete = [
    'stress_score_UPDATED_AT',
    'stress_score_STRESS_SCORE',
    'stress_score_SLEEP_POINTS',
    'stress_score_MAX_SLEEP_POINTS',
    'stress_score_RESPONSIVENESS_POINTS',
    'stress_score_MAX_RESPONSIVENESS_POINTS',
    'stress_score_EXERTION_POINTS',
    'stress_score_MAX_EXERTION_POINTS',
    'stress_score_STATUS',
    'stress_score_CALCULATION_FAILED'
]

# Define numerical columns for aggregation
numerical_cols = [
    'oxygen_variation_Infrared to Red Signal Ratio',
    'heart_rate_global_value',
    'daily_hrv_summary_rmssd',
    'daily_hrv_summary_nremhr',
    'daily_hrv_summary_entropy',
    'daily_respiratory_rate_daily_respiratory_rate',
    'hrv_details_rmssd',
    'hrv_details_coverage',
    'hrv_details_low_frequency',
    'hrv_details_high_frequency',
    'daily_spo2_average_value',
    'minute_spo2_value',
    'heart_rate_activity_beats per minute',
    'daily_total_steps',
    'active_minutes_light',
    'active_minutes_moderate	active_minutes_very	activity_level_level',
  	
    'heart_rate_activity_root mean square of successive differences milliseconds',
    'heart_rate_activity_standard deviation milliseconds',
    'hrv_activity_root mean square of successive differences milliseconds',
    'hrv_activity_standard deviation milliseconds',
    'spo2_activity_oxygen saturation percentage'
]

def parse_timestamp_flexible(timestamp_series):
    """
    Parse timestamps with flexible formatting to handle various formats including nanoseconds
    """
    try:
        # Try standard parsing first
        return pd.to_datetime(timestamp_series, format='mixed', errors='coerce')
    except:
        pass
    
    try:
        # Try ISO8601 format
        return pd.to_datetime(timestamp_series, format='ISO8601', errors='coerce')
    except:
        pass
    
    try:
        # If that fails, try without format specification
        return pd.to_datetime(timestamp_series, errors='coerce')
    except Exception as e:
        print(f"    ⚠ Warning: Some timestamps could not be parsed: {e}")
        return pd.to_datetime(timestamp_series, errors='coerce')

print("="*70)
print("FITBIT-QUESTIONNAIRE DATA MERGER (ROBUST VERSION)")
print("="*70)

print("\nStep 1: Loading questionnaire data...")
questionnaire_df = pd.read_csv(questionnaire_path)

# Convert timestamp to datetime with flexible parsing
questionnaire_df['timestamp'] = parse_timestamp_flexible(questionnaire_df['timestamp'])
questionnaire_df['date_only'] = pd.to_datetime(questionnaire_df['date_only'], errors='coerce')

# Normalize userid to string
questionnaire_df['userid'] = questionnaire_df['userid'].astype(str)

# Remove any rows where timestamp parsing failed
original_count = len(questionnaire_df)
questionnaire_df = questionnaire_df.dropna(subset=['timestamp', 'date_only'])
if len(questionnaire_df) < original_count:
    print(f"⚠ Removed {original_count - len(questionnaire_df)} rows with invalid timestamps")

print(f"✓ Loaded {len(questionnaire_df)} questionnaire responses")
print(f"✓ Users in questionnaire: {sorted(questionnaire_df['userid'].unique())}")

# Find all consolidated Fitbit files
print("\nStep 2: Finding Fitbit files...")
fitbit_files = glob.glob(os.path.join(fitbit_dir, "user_*_consolidated.csv"))
print(f"✓ Found {len(fitbit_files)} Fitbit files")

# Extract userids from filenames and create mapping
print("\nStep 3: Analyzing userid formats...")
fitbit_userid_map = {}
for fitbit_file in fitbit_files:
    filename = os.path.basename(fitbit_file)
    # Extract the original userid from filename (e.g., "1" from "user_1_consolidated.csv")
    user_id_original = filename.replace('user_', '').replace('_consolidated.csv', '')
    
    # Try multiple format conversions
    user_id_as_is = user_id_original
    user_id_int = str(int(user_id_original))  # Remove leading zeros
    user_id_padded = user_id_original.zfill(2)  # Add leading zero if needed
    
    fitbit_userid_map[fitbit_file] = {
        'original': user_id_original,
        'as_is': user_id_as_is,
        'int': user_id_int,
        'padded': user_id_padded
    }

# Create a reverse mapping: questionnaire userid -> fitbit file
quest_to_fitbit_map = {}
questionnaire_users = set(questionnaire_df['userid'].unique())

print(f"Questionnaire userids: {sorted(questionnaire_users)}")

for fitbit_file, userid_formats in fitbit_userid_map.items():
    for format_type, user_id in userid_formats.items():
        if user_id in questionnaire_users:
            if user_id not in quest_to_fitbit_map:
                quest_to_fitbit_map[user_id] = fitbit_file
                print(f"✓ Matched: questionnaire '{user_id}' -> Fitbit '{os.path.basename(fitbit_file)}' (using {format_type} format)")

if len(quest_to_fitbit_map) == 0:
    print("\n⚠ ERROR: No matching userids found!")
    print("Please run 'diagnose_userid_mismatch.py' to identify the issue.")
    exit(1)

print(f"\n✓ Successfully matched {len(quest_to_fitbit_map)} users")
unmatched_users = questionnaire_users - set(quest_to_fitbit_map.keys())
if unmatched_users:
    print(f"⚠ Questionnaire users without Fitbit data: {sorted(unmatched_users)}")

# Initialize lists to store results
results_5min_detailed = []
results_10min_detailed = []
results_5min_aggregated = []
results_10min_aggregated = []

print("\n" + "="*70)
print("PROCESSING DATA")
print("="*70)

# Track statistics
total_responses_processed = 0
total_responses_with_data = 0
users_processed = 0
users_with_errors = []

# Process each questionnaire user
for quest_userid, fitbit_file in sorted(quest_to_fitbit_map.items()):
    print(f"\nProcessing user {quest_userid}...")
    print(f"  Fitbit file: {os.path.basename(fitbit_file)}")
    
    try:
        # Load Fitbit data with low_memory=False to avoid DtypeWarning
        fitbit_df = pd.read_csv(fitbit_file, low_memory=False)
        
        # Convert timestamp to datetime with flexible parsing
        print(f"  Parsing timestamps...")
        fitbit_df['timestamp'] = parse_timestamp_flexible(fitbit_df['timestamp'])
        
        # Remove rows where timestamp parsing failed
        original_fitbit_count = len(fitbit_df)
        fitbit_df = fitbit_df.dropna(subset=['timestamp'])
        if len(fitbit_df) < original_fitbit_count:
            print(f"  ⚠ Removed {original_fitbit_count - len(fitbit_df)} rows with invalid timestamps")
        
        # Delete specified columns if they exist
        cols_to_drop = [col for col in columns_to_delete if col in fitbit_df.columns]
        if cols_to_drop:
            fitbit_df = fitbit_df.drop(columns=cols_to_drop)
            print(f"  ✓ Removed {len(cols_to_drop)} stress-related columns")
        
        print(f"  ✓ Loaded {len(fitbit_df):,} valid Fitbit records")
        
        # Get questionnaire responses for this user
        user_questions = questionnaire_df[questionnaire_df['userid'] == quest_userid].copy()
        
        if len(user_questions) == 0:
            print(f"  ⚠ No questionnaire responses found for user {quest_userid}")
            continue
            
        print(f"  ✓ Found {len(user_questions)} questionnaire responses")
        
        # Process each questionnaire response
        response_count = 0
        responses_with_data = 0
        
        for idx, question_row in user_questions.iterrows():
            response_time = question_row['timestamp']
            response_date = question_row['date_only']
            total_responses_processed += 1
            
            # Calculate time windows
            time_5min_before = response_time - timedelta(minutes=5)
            time_10min_before = response_time - timedelta(minutes=10)
            
            # Filter Fitbit data for the same date
            fitbit_same_date = fitbit_df[fitbit_df['timestamp'].dt.date == response_date.date()].copy()
            
            if len(fitbit_same_date) == 0:
                continue
            
            # 5-minute window
            fitbit_5min = fitbit_same_date[
                (fitbit_same_date['timestamp'] >= time_5min_before) & 
                (fitbit_same_date['timestamp'] < response_time)
            ].copy()
            
            # 1-hour window
            fitbit_10min = fitbit_same_date[
                (fitbit_same_date['timestamp'] >= time_10min_before) & 
                (fitbit_same_date['timestamp'] < response_time)
            ].copy()
            
            # Process 5-minute window
            if len(fitbit_5min) > 0:
                # Detailed records
                fitbit_5min_detail = fitbit_5min.copy()
                for col in question_row.index:
                    if col not in ['timestamp']:
                        fitbit_5min_detail[f'q_{col}'] = question_row[col]
                fitbit_5min_detail['response_timestamp'] = response_time
                fitbit_5min_detail['time_window'] = '5min'
                fitbit_5min_detail['window_start'] = time_5min_before
                fitbit_5min_detail['window_end'] = response_time
                results_5min_detailed.append(fitbit_5min_detail)
                
                # Aggregated statistics
                agg_dict = {
                    'userid': quest_userid, 
                    'response_timestamp': response_time, 
                    'window_start': time_5min_before, 
                    'window_end': response_time,
                    'record_count': len(fitbit_5min)
                }
                
                # Add questionnaire data
                for col in question_row.index:
                    if col not in ['timestamp']:
                        agg_dict[f'q_{col}'] = question_row[col]
                
                # Calculate statistics for numerical columns
                for col in fitbit_5min.columns:
                    if col in numerical_cols:
                        try:
                            col_data = pd.to_numeric(fitbit_5min[col], errors='coerce')
                            non_null = col_data.dropna()
                            if len(non_null) > 0:
                                agg_dict[f'{col}_mean'] = non_null.mean()
                                agg_dict[f'{col}_std'] = non_null.std()
                                agg_dict[f'{col}_min'] = non_null.min()
                                agg_dict[f'{col}_max'] = non_null.max()
                                agg_dict[f'{col}_median'] = non_null.median()
                                agg_dict[f'{col}_count'] = len(non_null)
                        except:
                            pass
                
                results_5min_aggregated.append(agg_dict)
            
            # Process 1-hour window
            if len(fitbit_10min) > 0:
                # Detailed records
                fitbit_10min_detail = fitbit_10min.copy()
                for col in question_row.index:
                    if col not in ['timestamp']:
                        fitbit_10min_detail[f'q_{col}'] = question_row[col]
                fitbit_10min_detail['response_timestamp'] = response_time
                fitbit_10min_detail['time_window'] = '10min'
                fitbit_10min_detail['window_start'] = time_10min_before
                fitbit_10min_detail['window_end'] = response_time
                results_10min_detailed.append(fitbit_10min_detail)
                
                # Aggregated statistics
                agg_dict = {
                    'userid': quest_userid,
                    'response_timestamp': response_time,
                    'window_start': time_10min_before,
                    'window_end': response_time,
                    'record_count': len(fitbit_10min)
                }
                
                # Add questionnaire data
                for col in question_row.index:
                    if col not in ['timestamp']:
                        agg_dict[f'q_{col}'] = question_row[col]
                
                # Calculate statistics for numerical columns
                for col in fitbit_10min.columns:
                    if col in numerical_cols:
                        try:
                            col_data = pd.to_numeric(fitbit_10min[col], errors='coerce')
                            non_null = col_data.dropna()
                            if len(non_null) > 0:
                                agg_dict[f'{col}_mean'] = non_null.mean()
                                agg_dict[f'{col}_std'] = non_null.std()
                                agg_dict[f'{col}_min'] = non_null.min()
                                agg_dict[f'{col}_max'] = non_null.max()
                                agg_dict[f'{col}_median'] = non_null.median()
                                agg_dict[f'{col}_count'] = len(non_null)
                        except:
                            pass
                
                results_10min_aggregated.append(agg_dict)
            
            if len(fitbit_5min) > 0 or len(fitbit_10min) > 0:
                response_count += 1
                responses_with_data += 1
                total_responses_with_data += 1
                if response_count <= 5:  # Show first 5 responses
                    print(f"    ✓ Response {response_count} at {response_time.strftime('%Y-%m-%d %H:%M')}: "
                          f"{len(fitbit_5min)} records (5min), {len(fitbit_10min)} records (10min)")
        
        if response_count > 5:
            print(f"    ... and {response_count - 5} more responses")
        
        if response_count == 0:
            print(f"  ⚠ No Fitbit data found in any time windows for this user")
        else:
            print(f"  ✓ Total: {responses_with_data}/{len(user_questions)} responses had matching Fitbit data")
            users_processed += 1
    
    except Exception as e:
        print(f"  ✗ Error processing user {quest_userid}: {e}")
        users_with_errors.append(quest_userid)
        import traceback
        traceback.print_exc()
        continue

# Save results
print("\n" + "="*70)
print("PROCESSING SUMMARY")
print("="*70)
print(f"Users processed successfully: {users_processed}/{len(quest_to_fitbit_map)}")
print(f"Questionnaire responses processed: {total_responses_processed}")
print(f"Responses with matching Fitbit data: {total_responses_with_data}")
if users_with_errors:
    print(f"Users with errors: {sorted(users_with_errors)}")

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Detailed data
if results_5min_detailed:
    final_5min_detailed = pd.concat(results_5min_detailed, ignore_index=True)
    output_file = os.path.join(output_dir, "fitbit_5min_window_detailed.csv")
    final_5min_detailed.to_csv(output_file, index=False)
    print(f"\n✓ 5-minute window (detailed): {output_file}")
    print(f"  Records: {len(final_5min_detailed):,}")
    print(f"  Unique responses: {final_5min_detailed['response_timestamp'].nunique()}")
    print(f"  Users: {final_5min_detailed['q_userid'].nunique()}")
else:
    print("\n⚠ No data for 5-minute window")

if results_10min_detailed:
    final_10min_detailed = pd.concat(results_10min_detailed, ignore_index=True)
    output_file = os.path.join(output_dir, "fitbit_10min_window_detailed.csv")
    final_10min_detailed.to_csv(output_file, index=False)
    print(f"\n✓ 1-hour window (detailed): {output_file}")
    print(f"  Records: {len(final_10min_detailed):,}")
    print(f"  Unique responses: {final_10min_detailed['response_timestamp'].nunique()}")
    print(f"  Users: {final_10min_detailed['q_userid'].nunique()}")
else:
    print("\n⚠ No data for 1-hour window")

# Aggregated data
if results_5min_aggregated:
    final_5min_agg = pd.DataFrame(results_5min_aggregated)
    output_file = os.path.join(output_dir, "fitbit_5min_window_aggregated.csv")
    final_5min_agg.to_csv(output_file, index=False)
    print(f"\n✓ 5-minute window (aggregated): {output_file}")
    print(f"  Rows: {len(final_5min_agg)} (one per questionnaire response)")
    print(f"  Users: {final_5min_agg['userid'].nunique()}")

if results_10min_aggregated:
    final_10min_agg = pd.DataFrame(results_10min_aggregated)
    output_file = os.path.join(output_dir, "fitbit_10min_window_aggregated.csv")
    final_10min_agg.to_csv(output_file, index=False)
    print(f"\n✓ 1-hour window (aggregated): {output_file}")
    print(f"  Rows: {len(final_10min_agg)} (one per questionnaire response)")
    print(f"  Users: {final_10min_agg['userid'].nunique()}")

# Combined detailed data
if results_5min_detailed and results_10min_detailed:
    combined_detailed = pd.concat([final_5min_detailed, final_10min_detailed], ignore_index=True)
    output_file = os.path.join(output_dir, "fitbit_combined_windows_detailed.csv")
    combined_detailed.to_csv(output_file, index=False)
    print(f"\n✓ Combined windows (detailed): {output_file}")
    print(f"  Records: {len(combined_detailed):,}")

print("\n" + "="*70)
print("✓ PROCESSING COMPLETE!")
print("="*70)