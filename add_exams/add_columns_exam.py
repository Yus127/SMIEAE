import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# User-university mapping
user_university_mapping = {
    **{i: 1 for i in range(1, 11)},   # University 1, Course A1
    **{i: 1 for i in range(11, 21)},  # University 1, Course A2
    **{i: 2 for i in range(21, 36)},  # University 2, Course B
    **{i: 1 for i in range(36, 56)}   # University 1, Course C
}

# Define exam periods
university_1_exams = [
    ('2025-03-17', '2025-03-21', 'Marzo'),
    ('2025-05-12', '2025-05-16', 'Mayo'),
    ('2025-06-23', '2025-06-27', 'Junio'),
    ('2025-07-14', '2025-07-18', 'Julio - Suficiencia'),
    ('2025-09-29', '2025-10-03', 'Sept-Oct'),
    ('2025-11-10', '2025-11-14', 'Noviembre'),
    ('2025-12-15', '2025-12-19', 'Dic-Ene parte 1'),
    ('2026-01-04', '2026-01-09', 'Dic-Ene parte 2'),
    ('2026-01-19', '2026-01-23', 'Enero - Suficiencia')
]

university_2_exams = [
    ('2025-03-17', '2025-03-22', 'Marzo'),
    ('2025-05-05', '2025-05-10', 'Mayo'),
    ('2025-06-09', '2025-06-14', 'Junio')
]

semana_santa = ('2025-04-14', '2025-04-19')

# HARDCODED PATHS
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data_with_log_transforms.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv'
DATE_COL = 'unified_date'
USER_COL = 'userid'

print(f"Reading data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# Ensure date column is datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

# Initialize new columns
df['university'] = df[USER_COL].map(user_university_mapping)
df['is_exam_period'] = 0
df['exam_period_name'] = ''
df['is_semana_santa'] = 0
df['is_holiday_period'] = 0
df['days_to_next_exam'] = np.nan
df['days_since_last_exam'] = np.nan
df['weeks_to_next_exam'] = np.nan
df['weeks_since_last_exam'] = np.nan
df['exam_proximity_category'] = ''

# Semana Santa dates
semana_start = pd.to_datetime(semana_santa[0])
semana_end = pd.to_datetime(semana_santa[1])

print("Processing exam features...")

# Process by university
for university in [1, 2]:
    university_mask = df['university'] == university
    exam_list = university_1_exams if university == 1 else university_2_exams
    
    print(f"Processing University {university} students...")
    
    for idx in df[university_mask].index:
        current_date = df.at[idx, DATE_COL]
        
        # Check Semana Santa
        if semana_start <= current_date <= semana_end:
            df.at[idx, 'is_semana_santa'] = 1
            df.at[idx, 'is_holiday_period'] = 1
            df.at[idx, 'exam_proximity_category'] = 'holiday'
        
        # Check exam periods
        for exam_start, exam_end, exam_name in exam_list:
            exam_start_dt = pd.to_datetime(exam_start)
            exam_end_dt = pd.to_datetime(exam_end)
            
            if exam_start_dt <= current_date <= exam_end_dt:
                df.at[idx, 'is_exam_period'] = 1
                df.at[idx, 'exam_period_name'] = exam_name
                df.at[idx, 'is_holiday_period'] = 1
                df.at[idx, 'exam_proximity_category'] = 'exam_week'
                break
        
        # Calculate days to next exam
        future_exams = []
        for exam_start, exam_end, exam_name in exam_list:
            exam_start_dt = pd.to_datetime(exam_start)
            if exam_start_dt > current_date:
                future_exams.append((exam_start_dt, exam_name))
        
        if future_exams:
            next_exam_date, next_exam_name = min(future_exams, key=lambda x: x[0])
            days_to_exam = (next_exam_date - current_date).days
            df.at[idx, 'days_to_next_exam'] = days_to_exam
            df.at[idx, 'weeks_to_next_exam'] = days_to_exam / 7
            
            if df.at[idx, 'exam_proximity_category'] == '':
                if days_to_exam <= 7:
                    df.at[idx, 'exam_proximity_category'] = 'pre_exam_week'
                elif days_to_exam <= 14:
                    df.at[idx, 'exam_proximity_category'] = 'pre_exam_2weeks'
                else:
                    df.at[idx, 'exam_proximity_category'] = 'normal'
        
        # Calculate days since last exam
        past_exams = []
        for exam_start, exam_end, exam_name in exam_list:
            exam_end_dt = pd.to_datetime(exam_end)
            if exam_end_dt < current_date:
                past_exams.append((exam_end_dt, exam_name))
        
        if past_exams:
            last_exam_date, last_exam_name = max(past_exams, key=lambda x: x[0])
            days_since_exam = (current_date - last_exam_date).days
            df.at[idx, 'days_since_last_exam'] = days_since_exam
            df.at[idx, 'weeks_since_last_exam'] = days_since_exam / 7
            
            if df.at[idx, 'exam_proximity_category'] == '':
                if days_since_exam <= 7:
                    df.at[idx, 'exam_proximity_category'] = 'post_exam_week'
                elif days_since_exam <= 14:
                    df.at[idx, 'exam_proximity_category'] = 'post_exam_2weeks'
                else:
                    df.at[idx, 'exam_proximity_category'] = 'normal'

print("Saving processed data...")
df.to_csv(OUTPUT_PATH, index=False)
print(f"Data saved to: {OUTPUT_PATH}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total rows processed: {len(df)}")
print(f"Exam period observations: {df['is_exam_period'].sum()}")
print(f"Semana Santa observations: {df['is_semana_santa'].sum()}")
print(f"Percentage in exam periods: {(df['is_exam_period'].sum()/len(df))*100:.2f}%")
print("\nExam proximity distribution:")
print(df['exam_proximity_category'].value_counts())
print("\nNew columns added:")
new_cols = ['university', 'is_exam_period', 'exam_period_name', 'is_semana_santa',
           'is_holiday_period', 'days_to_next_exam', 'days_since_last_exam',
           'weeks_to_next_exam', 'weeks_since_last_exam', 'exam_proximity_category']
for col in new_cols:
    print(f"  - {col}")
print("="*80)
print("DONE!")