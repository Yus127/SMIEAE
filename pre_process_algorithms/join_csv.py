import pandas as pd
import numpy as np
import glob
from pathlib import Path

# Configuration
daily_summaries_folder = "/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit"  # Folder with *_consolidated_daily_summary.csv files
daily_questions_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"  # Path to the questions file
output_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data.csv"

# Users to exclude
excluded_users = [39, 54]

# Read and combine all daily summary CSV files
def load_all_daily_summaries(folder_path):
    csv_files = glob.glob(f"{folder_path}/*_consolidated_daily_summary.csv")
    print(f"Found {len(csv_files)} daily summary CSV files")
    
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Extract user ID from filename
            filename = Path(file).stem
            # Handle formats like "user_13_consolidated_daily_summary" or "13_consolidated_daily_summary"
            parts = filename.split('_')
            user_id = None
            for part in parts:
                try:
                    user_id = int(part)
                    break
                except ValueError:
                    continue
            
            if user_id is None:
                print(f"Warning: Could not extract user ID from {filename}, skipping")
                continue
                
            df['userid'] = user_id
            dfs.append(df)
            print(f"Loaded: {file} (User {user_id}, {len(df)} records)")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal daily summary records: {len(combined_df)}")
        print(f"Total users: {combined_df['userid'].nunique()}")
        return combined_df
    else:
        print("No files loaded!")
        return None

# Load daily summaries
print("="*60)
print("LOADING DAILY SUMMARIES")
print("="*60)
daily_summaries = load_all_daily_summaries(daily_summaries_folder)

if daily_summaries is None:
    print("Failed to load daily summaries. Exiting.")
    exit()

# Load daily questions (anxiety and stress)
print("\n" + "="*60)
print("LOADING DAILY QUESTIONS")
print("="*60)
try:
    daily_questions = pd.read_csv(daily_questions_file)
    print(f"Loaded daily questions: {len(daily_questions)} records")
    print(f"Users in questions: {daily_questions['userid'].nunique()}")
    print(f"Columns: {list(daily_questions.columns)}")
except Exception as e:
    print(f"Error loading daily questions: {e}")
    exit()

# Exclude users 39 and 54
print("\n" + "="*60)
print("FILTERING DATA")
print("="*60)
print(f"Excluding users: {excluded_users}")

daily_summaries_filtered = daily_summaries[~daily_summaries['userid'].isin(excluded_users)]
daily_questions_filtered = daily_questions[~daily_questions['userid'].isin(excluded_users)]

print(f"Daily summaries after filtering: {len(daily_summaries_filtered)} records ({daily_summaries_filtered['userid'].nunique()} users)")
print(f"Daily questions after filtering: {len(daily_questions_filtered)} records ({daily_questions_filtered['userid'].nunique()} users)")

# Prepare date columns for merging
print("\n" + "="*60)
print("PREPARING DATA FOR MERGE")
print("="*60)

# Check what date column exists in daily_summaries
date_cols_summary = [col for col in daily_summaries_filtered.columns if 'date' in col.lower()]
print(f"Date columns in daily summaries: {date_cols_summary}")

# Check what date column exists in daily_questions
date_cols_questions = [col for col in daily_questions_filtered.columns if 'date' in col.lower()]
print(f"Date columns in daily questions: {date_cols_questions}")

# Convert date columns to standard format
# Assuming daily_questions has 'date_only' column
if 'date_only' in daily_questions_filtered.columns:
    daily_questions_filtered['merge_date'] = pd.to_datetime(daily_questions_filtered['date_only']).dt.date
else:
    print("Warning: 'date_only' column not found in daily_questions")
    # Try to find another date column
    if date_cols_questions:
        daily_questions_filtered['merge_date'] = pd.to_datetime(daily_questions_filtered[date_cols_questions[0]]).dt.date

# For daily summaries, find the appropriate date column
# Common names: sleep_global_dateOfSleep, date, dateOfSleep, etc.
if 'sleep_global_dateOfSleep' in daily_summaries_filtered.columns:
    daily_summaries_filtered['merge_date'] = pd.to_datetime(daily_summaries_filtered['sleep_global_dateOfSleep']).dt.date
elif 'date_only' in daily_summaries_filtered.columns:
    daily_summaries_filtered['merge_date'] = pd.to_datetime(daily_summaries_filtered['date_only']).dt.date
elif date_cols_summary:
    daily_summaries_filtered['merge_date'] = pd.to_datetime(daily_summaries_filtered[date_cols_summary[0]]).dt.date
else:
    print("ERROR: Cannot find date column in daily summaries!")
    exit()

# Select relevant columns from daily_questions (stress and anxiety)
questions_cols = ['userid', 'merge_date', 'i_stress_sliderNeutralPos', 'i_anxiety_sliderNeutralPos']
daily_questions_subset = daily_questions_filtered[questions_cols].copy()

# Rename for clarity
daily_questions_subset.rename(columns={
    'i_stress_sliderNeutralPos': 'stress_level',
    'i_anxiety_sliderNeutralPos': 'anxiety_level'
}, inplace=True)

print(f"\nDaily summaries shape before merge: {daily_summaries_filtered.shape}")
print(f"Daily questions shape before merge: {daily_questions_subset.shape}")

# Merge the dataframes
print("\n" + "="*60)
print("MERGING DATA")
print("="*60)
combined_data = pd.merge(
    daily_summaries_filtered,
    daily_questions_subset,
    on=['userid', 'merge_date'],
    how='left'  # Keep all daily summaries, even if no stress/anxiety data
)

print(f"Combined data shape: {combined_data.shape}")
print(f"Users in combined data: {combined_data['userid'].nunique()}")

# Check merge quality
print("\n" + "="*60)
print("MERGE QUALITY CHECK")
print("="*60)
total_records = len(combined_data)
with_stress = combined_data['stress_level'].notna().sum()
with_anxiety = combined_data['anxiety_level'].notna().sum()

print(f"Total records: {total_records}")
print(f"Records with stress data: {with_stress} ({with_stress/total_records*100:.1f}%)")
print(f"Records with anxiety data: {with_anxiety} ({with_anxiety/total_records*100:.1f}%)")

# Remove unwanted columns
print("\n" + "="*60)
print("REMOVING REDUNDANT COLUMNS")
print("="*60)
columns_to_remove = [
    'sleep_global_timeInBed',
    'wake_minutes',
    'respiratory_rate_summary_deep_sleep_signal_to_noise_mean',
    'micro_awakening_count',
    'timestamp_fmt',
    'hour',
    'sleep_global_logId',
    'sleep_global_startTime',
    'sleep_global_endTime',
    'sleep_global_dateOfSleep',
    'respiratory_rate_summary_full_sleep_breathing_rate_mean',
    'sleep_global_minutesAsleep',

    'merge_date'  # temporary merge column
]

# Check which columns actually exist before removing
existing_cols_to_remove = [col for col in columns_to_remove if col in combined_data.columns]
missing_cols = [col for col in columns_to_remove if col not in combined_data.columns]

if existing_cols_to_remove:
    print(f"Removing {len(existing_cols_to_remove)} columns: {existing_cols_to_remove}")
    combined_data = combined_data.drop(existing_cols_to_remove, axis=1)
else:
    print("No columns to remove (already absent)")

if missing_cols:
    print(f"Columns not found (already absent): {missing_cols}")

# Encode categorical variables
print("\n" + "="*60)
print("ENCODING CATEGORICAL VARIABLES")
print("="*60)

# Define encoding mappings
encoding_mappings = {
    'sleep_global_type': {
        'stages': 0,
        'classic': 1
    },
    'sleep_global_mainSleep': {
        'TRUE': 0,
        True: 0,
        'FALSE': 1,
        False: 1
    },
    'last_sleep_stage': {
        'light': 0,
        'rem': 1,
        'wake': 2,
        'asleep': 3,
        'deep': 4,
        'restless': 5,
        'awake': 6,
        'unknown': 7
    }
}

# Apply encodings
for column, mapping in encoding_mappings.items():
    if column in combined_data.columns:
        # Create encoded column
        combined_data[f'{column}_encoded'] = combined_data[column].map(mapping)
        
        # Check for unmapped values
        unmapped = combined_data[column][combined_data[f'{column}_encoded'].isna() & combined_data[column].notna()].unique()
        if len(unmapped) > 0:
            print(f"Warning: '{column}' has unmapped values: {unmapped}")
        
        print(f"✓ Encoded '{column}' -> '{column}_encoded'")
        print(f"  Original values: {combined_data[column].value_counts().to_dict()}")
        print(f"  Encoded distribution: {combined_data[f'{column}_encoded'].value_counts().sort_index().to_dict()}")
    else:
        print(f"✗ Column '{column}' not found in dataset")

# Consolidate date columns
print("\n" + "="*60)
print("CONSOLIDATING DATE COLUMNS")
print("="*60)

date_columns = ['timestamp', 'sleep_global_dateOfSleep', 'date']
existing_date_cols = [col for col in date_columns if col in combined_data.columns]

if existing_date_cols:
    print(f"Found date columns: {existing_date_cols}")
    
    # Check missing values in each column
    for col in existing_date_cols:
        missing = combined_data[col].isna().sum()
        total = len(combined_data)
        print(f"  '{col}': {missing}/{total} missing ({missing/total*100:.1f}%)")
    
    # Create unified date column using coalesce logic (first non-null value)
    combined_data['unified_date'] = None
    for col in existing_date_cols:
        # Convert to datetime if not already
        try:
            combined_data[col] = pd.to_datetime(combined_data[col], errors='coerce')
        except:
            pass
        
        # Fill unified_date with first non-null value
        combined_data['unified_date'] = combined_data['unified_date'].fillna(combined_data[col])
    
    # Convert unified_date to date format (remove time component)
    combined_data['unified_date'] = pd.to_datetime(combined_data['unified_date']).dt.date
    
    # Check result
    missing_unified = combined_data['unified_date'].isna().sum()
    print(f"\n✓ Created 'unified_date' column")
    print(f"  Missing values: {missing_unified}/{len(combined_data)} ({missing_unified/len(combined_data)*100:.1f}%)")
    
    # Drop original date columns
    print(f"\nDropping original date columns: {existing_date_cols}")
    combined_data = combined_data.drop(existing_date_cols, axis=1)
else:
    print("No date columns found to consolidate")

# Drop original categorical columns (keep only encoded versions)
categorical_cols_to_drop = [col for col in encoding_mappings.keys() if col in combined_data.columns]
if categorical_cols_to_drop:
    print(f"\nDropping original categorical columns: {categorical_cols_to_drop}")
    combined_data = combined_data.drop(categorical_cols_to_drop, axis=1)

# Save combined data
print("\n" + "="*60)
print("SAVING COMBINED DATA")
print("="*60)
combined_data.to_csv(output_file, index=False)
print(f"Combined data saved to: {output_file}")
print(f"Final dataset: {len(combined_data)} records, {len(combined_data.columns)} columns")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY BY USER")
print("="*60)
user_summary = combined_data.groupby('userid').agg({
    'stress_level': ['count', 'mean'],
    'anxiety_level': ['count', 'mean']
}).round(2)
print(user_summary)

print("\n" + "="*60)
print("PROCESS COMPLETE!")
print("="*60)


"""
columns_to_remove = [
    'sleep_global_timeInBed',
    'wake_minutes',
    'respiratory_rate_summary_deep_sleep_signal_to_noise_mean',
    'micro_awakening_count',
    'timestamp_fmt',
    'hour',
    'sleep_global_logId',
    'sleep_global_startTime',
    'sleep_global_endTime',
    'sleep_global_dateOfSleep'
    'merge_date'  # temporary merge column
]

# Configuration
daily_summaries_folder = "/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit"  # Folder with *_consolidated_daily_summary.csv files
daily_questions_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"  # Path to the questions file
output_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data.csv"


"""