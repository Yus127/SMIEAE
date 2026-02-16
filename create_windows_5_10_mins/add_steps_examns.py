import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

fitbit_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output"
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"

os.makedirs(output_dir, exist_ok=True)

# User-course mapping
user_courses = {
    1: {'university': 1, 'course': 'A1'},
    2: {'university': 1, 'course': 'A1'},
    3: {'university': 1, 'course': 'A1'},
    4: {'university': 1, 'course': 'A1'},
    5: {'university': 1, 'course': 'A1'},
    6: {'university': 1, 'course': 'A1'},
    7: {'university': 1, 'course': 'A1'},
    8: {'university': 1, 'course': 'A1'},
    9: {'university': 1, 'course': 'A1'},
    10: {'university': 1, 'course': 'A1'},
    11: {'university': 1, 'course': 'A2'},
    12: {'university': 1, 'course': 'A2'},
    13: {'university': 1, 'course': 'A2'},
    14: {'university': 1, 'course': 'A2'},
    15: {'university': 1, 'course': 'A2'},
    16: {'university': 1, 'course': 'A2'},
    17: {'university': 1, 'course': 'A2'},
    18: {'university': 1, 'course': 'A2'},
    19: {'university': 1, 'course': 'A2'},
    20: {'university': 1, 'course': 'A2'},
    21: {'university': 2, 'course': 'B'},
    22: {'university': 2, 'course': 'B'},
    23: {'university': 2, 'course': 'B'},
    24: {'university': 2, 'course': 'B'},
    25: {'university': 2, 'course': 'B'},
    26: {'university': 2, 'course': 'B'},
    27: {'university': 2, 'course': 'B'},
    28: {'university': 2, 'course': 'B'},
    29: {'university': 2, 'course': 'B'},
    30: {'university': 2, 'course': 'B'},
    31: {'university': 2, 'course': 'B'},
    32: {'university': 2, 'course': 'B'},
    33: {'university': 2, 'course': 'B'},
    34: {'university': 2, 'course': 'B'},
    35: {'university': 2, 'course': 'B'},
    36: {'university': 1, 'course': 'C'},
    37: {'university': 1, 'course': 'C'},
    38: {'university': 1, 'course': 'C'},
    39: {'university': 1, 'course': 'C'},
    40: {'university': 1, 'course': 'C'},
    41: {'university': 1, 'course': 'C'},
    42: {'university': 1, 'course': 'C'},
    43: {'university': 1, 'course': 'C'},
    44: {'university': 1, 'course': 'C'},
    45: {'university': 1, 'course': 'C'},
    46: {'university': 1, 'course': 'C'},
    47: {'university': 1, 'course': 'C'},
    48: {'university': 1, 'course': 'C'},
    49: {'university': 1, 'course': 'C'},
    50: {'university': 1, 'course': 'C'},
    51: {'university': 1, 'course': 'C'},
    52: {'university': 1, 'course': 'C'},
    53: {'university': 1, 'course': 'C'},
    54: {'university': 1, 'course': 'C'},
    55: {'university': 1, 'course': 'C'}
}

# Exam periods for University 1 (2025)
exam_periods_uni1 = [
    ('2025-03-17', '2025-03-21'),  # March
    ('2025-05-12', '2025-05-16'),  # May
    ('2025-06-23', '2025-06-27'),  # June
    ('2025-07-14', '2025-07-18'),  # July (Suficiencia)
    ('2025-09-29', '2025-10-03'),  # October
    ('2025-11-10', '2025-11-14'),  # November
    ('2025-12-15', '2025-12-19'),  # December
    ('2026-01-04', '2026-01-09'),  # January
    ('2026-01-19', '2026-01-23'),  # January (Suficiencia)
]

# Exam periods for University 2 (2025)
exam_periods_uni2 = [
    ('2025-03-17', '2025-03-22'),  # March
    ('2025-05-05', '2025-05-10'),  # May
    ('2025-06-09', '2025-06-14'),  # June
]

# Easter break (Semana Santa) - students were more relaxed
easter_break = [
    ('2025-04-14', '2025-04-19')
]

def date_in_period(date, periods):
    """Check if a date falls within any of the given periods"""
    date = pd.to_datetime(date).date()
    for start, end in periods:
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        if start_date <= date <= end_date:
            return True
    return False

def get_exam_features(date, university):
    """Get exam-related features for a given date and university"""
    if university == 1:
        is_exam = date_in_period(date, exam_periods_uni1)
    elif university == 2:
        is_exam = date_in_period(date, exam_periods_uni2)
    else:
        is_exam = False
    
    is_easter = date_in_period(date, easter_break)
    
    # Days until next exam
    date_obj = pd.to_datetime(date).date()
    days_to_exam = None
    
    if university == 1:
        periods = exam_periods_uni1
    elif university == 2:
        periods = exam_periods_uni2
    else:
        periods = []
    
    min_days = float('inf')
    for start, end in periods:
        start_date = pd.to_datetime(start).date()
        if start_date >= date_obj:
            days_diff = (start_date - date_obj).days
            if days_diff < min_days:
                min_days = days_diff
    
    if min_days != float('inf'):
        days_to_exam = min_days
    
    return {
        'is_exam_period': 1 if is_exam else 0,
        'is_easter_break': 1 if is_easter else 0,
        'days_until_exam': days_to_exam if days_to_exam is not None else -1,
        'is_pre_exam_week': 1 if (days_to_exam is not None and 0 <= days_to_exam <= 7) else 0
    }

print("ENRICHING ML DATASETS WITH STEPS AND EXAM DATA")

# EXTRACT DAILY STEPS FROM FITBIT DATA


# Find all Fitbit files
fitbit_files = glob.glob(os.path.join(fitbit_dir, "user_*_consolidated.csv"))
print(f"Found {len(fitbit_files)} Fitbit files")

# Dictionary to store daily steps by user and date
daily_steps_data = {}

for fitbit_file in fitbit_files:
    # Extract user ID
    filename = os.path.basename(fitbit_file)
    user_id = str(int(filename.replace('user_', '').replace('_consolidated.csv', '')))
    
    try:
        # Load Fitbit data
        df = pd.read_csv(fitbit_file, low_memory=False)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Extract date
        df['date'] = df['timestamp'].dt.date
        
        # Check if daily_total_steps column exists
        if 'daily_total_steps' in df.columns:
            # Get daily steps (take the max value per day since it's cumulative)
            daily_steps = df.groupby('date')['daily_total_steps'].max().to_dict()
            
            if user_id not in daily_steps_data:
                daily_steps_data[user_id] = {}
            
            for date, steps in daily_steps.items():
                if pd.notna(steps):
                    daily_steps_data[user_id][date] = steps
            
            print(f"  User {user_id}: {len(daily_steps)} days with step data")
        else:
            print(f"  User {user_id}: No daily_total_steps column found")
    
    except Exception as e:
        print(f"  Error processing user {user_id}: {e}")

print(f"\n Extracted step data for {len(daily_steps_data)} users")

# ENRICH ML DATASETS

print("\n" + "="*70)

# Find all ML-ready CSV files
ml_files = [
    "ml_ready_5min_window.csv",
    "ml_ready_10min_window.csv"
]

# Also check for optimized files if they exist
optimized_dir = os.path.join(ml_dir, "optimized")
if os.path.exists(optimized_dir):
    optimized_files = glob.glob(os.path.join(optimized_dir, "ml_*.csv"))
    for opt_file in optimized_files:
        ml_files.append(os.path.join("optimized", os.path.basename(opt_file)))

processed_count = 0

for ml_file in ml_files:
    ml_path = os.path.join(ml_dir, ml_file)
    
    if not os.path.exists(ml_path):
        continue
    
    print(f"\nProcessing: {ml_file}")
    
    try:
        # Load ML dataset
        df = pd.read_csv(ml_path)
        original_shape = df.shape
        
        # Convert response_timestamp to datetime
        df['response_timestamp'] = pd.to_datetime(df['response_timestamp'])
        df['date'] = df['response_timestamp'].dt.date
        
        # Convert userid to string
        df['userid'] = df['userid'].astype(str)
        
        # Add daily steps
        df['daily_total_steps'] = df.apply(
            lambda row: daily_steps_data.get(row['userid'], {}).get(row['date'], np.nan),
            axis=1
        )
        
        # Add university and course information
        df['university'] = df['userid'].apply(lambda x: user_courses.get(int(x), {}).get('university', np.nan))
        df['course'] = df['userid'].apply(lambda x: user_courses.get(int(x), {}).get('course', 'Unknown'))
        
        # Add exam period features
        exam_features = df.apply(
            lambda row: pd.Series(get_exam_features(row['response_timestamp'], row['university'])),
            axis=1
        )
        
        df = pd.concat([df, exam_features], axis=1)
        
        # Remove temporary date column
        df = df.drop('date', axis=1)
        
        # Reorder columns: metadata, new features, original features, targets
        metadata_cols = ['userid', 'response_timestamp', 'university', 'course']
        new_feature_cols = ['daily_total_steps', 'is_exam_period', 'is_easter_break', 
                           'days_until_exam', 'is_pre_exam_week']
        
        # Get remaining columns
        other_cols = [col for col in df.columns 
                     if col not in metadata_cols + new_feature_cols]
        
        # Reorder
        final_cols = metadata_cols + new_feature_cols + other_cols
        df = df[final_cols]
        
        # ── Drop columns with <50% completeness ──
        completeness = df.notna().mean()
        low_completeness_cols = completeness[completeness < 0.50].index.tolist()
        if low_completeness_cols:
            print(f"   Dropping {len(low_completeness_cols)} columns with <50% completeness:")
            for col in low_completeness_cols:
                print(f"      {col:40s} {completeness[col]*100:5.1f}%")
            df = df.drop(columns=low_completeness_cols)
        else:
            print(f"   All columns have ≥50% completeness")

        # Save enriched dataset
        output_file = os.path.join(output_dir, os.path.basename(ml_file).replace('.csv', '_enriched.csv'))
        df.to_csv(output_file, index=False)
        
        print(f"   Original shape: {original_shape}")
        print(f"   New shape: {df.shape}")
        print(f"   Added 5 new features")
        print(f"   Step data coverage: {df['daily_total_steps'].notna().sum()}/{len(df)} ({df['daily_total_steps'].notna().mean()*100:.1f}%)")
        print(f"   Exam periods marked: {df['is_exam_period'].sum()} responses")
        print(f"   Easter break marked: {df['is_easter_break'].sum()} responses")
        print(f"   Saved: {output_file}")
        
        processed_count += 1
        
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()



# FINAL SUMMARY

print("\n" + "="*70)

print(f"\nProcessed {processed_count} ML datasets")

print(f"\nAll enriched files saved to: {output_dir}")

