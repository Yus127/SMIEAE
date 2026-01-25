import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

# Define paths
fitbit_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output"
ml_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready"
output_dir = "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched"

# Create output directory
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

print("="*70)
print("ENRICHING ML DATASETS WITH STEPS AND EXAM DATA")
print("="*70)

# ==============================================================================
# STEP 1: EXTRACT DAILY STEPS FROM FITBIT DATA
# ==============================================================================

print("\nStep 1: Extracting daily step counts from Fitbit data...")

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
        
        # Convert timestamp to datetime
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

print(f"\n✓ Extracted step data for {len(daily_steps_data)} users")

# ==============================================================================
# STEP 2: ENRICH ML DATASETS
# ==============================================================================

print("\n" + "="*70)
print("Step 2: Enriching ML datasets...")
print("="*70)

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
        
        # Save enriched dataset
        output_file = os.path.join(output_dir, os.path.basename(ml_file).replace('.csv', '_enriched.csv'))
        df.to_csv(output_file, index=False)
        
        print(f"  ✓ Original shape: {original_shape}")
        print(f"  ✓ New shape: {df.shape}")
        print(f"  ✓ Added 5 new features")
        print(f"  ✓ Step data coverage: {df['daily_total_steps'].notna().sum()}/{len(df)} ({df['daily_total_steps'].notna().mean()*100:.1f}%)")
        print(f"  ✓ Exam periods marked: {df['is_exam_period'].sum()} responses")
        print(f"  ✓ Easter break marked: {df['is_easter_break'].sum()} responses")
        print(f"  ✓ Saved: {output_file}")
        
        processed_count += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# STEP 3: CREATE SUMMARY REPORTS
# ==============================================================================

print("\n" + "="*70)
print("Step 3: Creating summary reports...")
print("="*70)

# Load one of the enriched datasets for analysis
main_enriched_file = os.path.join(output_dir, "ml_ready_combined_windows_enriched.csv")

if os.path.exists(main_enriched_file):
    df_summary = pd.read_csv(main_enriched_file)
    
    # Summary by university
    print("\nSummary by University:")
    uni_summary = df_summary.groupby('university').agg({
        'userid': 'nunique',
        'response_timestamp': 'count',
        'daily_total_steps': lambda x: x.notna().sum(),
        'is_exam_period': 'sum',
        'is_easter_break': 'sum'
    }).rename(columns={
        'userid': 'unique_users',
        'response_timestamp': 'total_responses',
        'daily_total_steps': 'responses_with_steps',
        'is_exam_period': 'exam_period_responses',
        'is_easter_break': 'easter_break_responses'
    })
    
    print(uni_summary)
    
    uni_summary_file = os.path.join(output_dir, "summary_by_university.csv")
    uni_summary.to_csv(uni_summary_file)
    print(f"\n✓ Saved: {uni_summary_file}")
    
    # Summary by course
    print("\nSummary by Course:")
    course_summary = df_summary.groupby('course').agg({
        'userid': 'nunique',
        'response_timestamp': 'count',
        'daily_total_steps': lambda x: x.notna().sum(),
        'is_exam_period': 'sum'
    }).rename(columns={
        'userid': 'unique_users',
        'response_timestamp': 'total_responses',
        'daily_total_steps': 'responses_with_steps',
        'is_exam_period': 'exam_period_responses'
    })
    
    print(course_summary)
    
    course_summary_file = os.path.join(output_dir, "summary_by_course.csv")
    course_summary.to_csv(course_summary_file)
    print(f"\n✓ Saved: {course_summary_file}")
    
    # Exam period analysis
    print("\nExam Period Analysis:")
    exam_analysis = pd.DataFrame({
        'Period': ['Exam Period', 'Pre-Exam Week (1-7 days before)', 'Easter Break', 'Regular Period'],
        'Response Count': [
            df_summary['is_exam_period'].sum(),
            df_summary['is_pre_exam_week'].sum(),
            df_summary['is_easter_break'].sum(),
            len(df_summary) - df_summary['is_exam_period'].sum() - 
            df_summary['is_pre_exam_week'].sum() - df_summary['is_easter_break'].sum()
        ],
        'Percentage': [
            df_summary['is_exam_period'].mean() * 100,
            df_summary['is_pre_exam_week'].mean() * 100,
            df_summary['is_easter_break'].mean() * 100,
            (1 - df_summary['is_exam_period'].mean() - 
             df_summary['is_pre_exam_week'].mean() - 
             df_summary['is_easter_break'].mean()) * 100
        ]
    })
    
    print(exam_analysis)
    
    exam_analysis_file = os.path.join(output_dir, "exam_period_analysis.csv")
    exam_analysis.to_csv(exam_analysis_file, index=False)
    print(f"\n✓ Saved: {exam_analysis_file}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "="*70)
print("✓ ENRICHMENT COMPLETE!")
print("="*70)

print(f"\nProcessed {processed_count} ML datasets")
print(f"\nNew features added:")
print("  1. daily_total_steps - Total steps taken on that day")
print("  2. university - University identifier (1 or 2)")
print("  3. course - Course identifier (A1, A2, B, C)")
print("  4. is_exam_period - Binary indicator for exam weeks")
print("  5. is_easter_break - Binary indicator for Easter break (relaxed period)")
print("  6. days_until_exam - Number of days until next exam (-1 if no upcoming exam)")
print("  7. is_pre_exam_week - Binary indicator for 1-7 days before exam")

print(f"\nAll enriched files saved to: {output_dir}")

print("\nKey files:")
print("  • ml_ready_combined_windows_enriched.csv - Main dataset with all features")
print("  • ml_heart_rate_focused_enriched.csv - Heart rate only (if exists)")
print("  • summary_by_university.csv - Statistics by university")
print("  • summary_by_course.csv - Statistics by course")
print("  • exam_period_analysis.csv - Breakdown of exam vs regular periods")

print("\nML Considerations:")
print("  • Use 'is_exam_period' to analyze stress differences during exams")
print("  • Use 'is_easter_break' as a control period (expected lower stress)")
print("  • Use 'days_until_exam' to model stress buildup before exams")
print("  • Use 'daily_total_steps' as a physical activity indicator")
print("  • Consider 'university' and 'course' as categorical features or for stratification")
