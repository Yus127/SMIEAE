import pandas as pd
import numpy as np

# Configuration
INPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data.csv'
OUTPUT_PATH = '/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/combined_daily_data_with_log_transforms.csv'

# Skewness threshold: columns with |skew| > 1.0 get a log transform
SKEW_THRESHOLD = 1.0

# Columns to exclude from log transforms (non-numeric, identifiers, targets, encoded)
EXCLUDE_COLS = [
    'userid',
    'stress_level',
    'anxiety_level',
    'sleep_global_type_encoded',
    'sleep_global_mainSleep_encoded',
    'last_sleep_stage_encoded',
    'sleep_global_efficiency',
    'sleep_global_infoCode',
    'sleep_global_logType',
    'sleep_global_levels',
    'unified_date',
]

# Columns with negative values (deviations) — skip these, log is not appropriate
DEVIATION_COLS = [
    'deep_sleep_deviation_from_avg',
    'light_sleep_deviation_from_avg',
    'rem_sleep_deviation_from_avg',
    'wake_deviation_from_avg',
]

# Columns that use -1 as a sentinel for missing — skip these
SENTINEL_COLS = [
    'respiratory_rate_summary_deep_sleep_breathing_rate_mean',
    'respiratory_rate_summary_light_sleep_breathing_rate_mean',
    'respiratory_rate_summary_rem_sleep_breathing_rate_mean',
]

print("=" * 60)
print("ADDING LOG TRANSFORMS TO SKEWED COLUMNS")
print("=" * 60)

print(f"\nReading data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

# Identify numeric columns eligible for log transform
numeric_cols = df.select_dtypes(include='number').columns.tolist()
candidates = [
    c for c in numeric_cols
    if c not in EXCLUDE_COLS
    and c not in DEVIATION_COLS
    and c not in SENTINEL_COLS
]

print(f"\nAnalyzing skewness of {len(candidates)} numeric columns...")
print(f"Threshold: |skew| > {SKEW_THRESHOLD}")

log_transformed = []

for col in candidates:
    skew = df[col].skew()
    col_min = df[col].min()

    if abs(skew) > SKEW_THRESHOLD and col_min >= 0:
        log_col_name = f"{col}_log"
        df[log_col_name] = np.log1p(df[col])
        log_transformed.append((col, skew))
        print(f"  + {col} (skew={skew:.3f}) -> {log_col_name}")

print(f"\nLog-transformed {len(log_transformed)} columns")
print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to: {OUTPUT_PATH}")

print("\n" + "=" * 60)
print("LOG-TRANSFORMED COLUMNS:")
print("=" * 60)
for col, skew in log_transformed:
    print(f"  {col}_log  (original skew: {skew:.3f})")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
