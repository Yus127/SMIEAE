import pandas as pd
import numpy as np

# Your data - note the delimiter is semicolon, not comma
data_file = "/Users/YusMolina/Downloads/smieae/data/original_data/daily_questions 2025-12-03.csv"
df = pd.read_csv(data_file, delimiter=';')

print("Data loaded successfully:")
print(df.head())
print(f"\nTotal number of responses: {len(df)}")

# Calculate response time (time between sent and stop)
df['response_time_seconds'] = df['timeStampStop'] - df['timeStampSent']

# Convert to minutes for easier interpretation
df['response_time_minutes'] = df['response_time_seconds'] / 60

# Filter out negative response times
print(f"\nTotal responses before filtering: {len(df)}")
print(f"Negative response times found: {(df['response_time_seconds'] < 0).sum()}")

# Print full rows with negative response times
negative_rows = df[df['response_time_seconds'] < 0].copy()
if len(negative_rows) > 0:
    print("\n" + "=" * 70)
    print("Rows with NEGATIVE response times:")
    print("=" * 70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(negative_rows.to_string())
    print("\n")

# Convert negative values to their absolute values
df['response_time_seconds'] = df['response_time_seconds'].abs()
df['response_time_minutes'] = df['response_time_seconds'] / 60

print(f"Total responses (using absolute values): {len(df)}")

print("\n" + "=" * 70)
print("Response Times Analysis")
print("=" * 70)

print(f"\nMean response time: {df['response_time_seconds'].mean():.2f} seconds ({df['response_time_minutes'].mean():.2f} minutes)")
print(f"Median response time: {df['response_time_seconds'].median():.2f} seconds ({df['response_time_minutes'].median():.2f} minutes)")

print("\nAdditional Statistics:")
print(f"Standard deviation: {df['response_time_seconds'].std():.2f} seconds ({df['response_time_minutes'].std():.2f} minutes)")
print(f"Min response time: {df['response_time_seconds'].min():.2f} seconds ({df['response_time_minutes'].min():.2f} minutes)")
print(f"Max response time: {df['response_time_seconds'].max():.2f} seconds ({df['response_time_minutes'].max():.2f} minutes)")
print(f"25th percentile: {df['response_time_seconds'].quantile(0.25):.2f} seconds")
print(f"75th percentile: {df['response_time_seconds'].quantile(0.75):.2f} seconds")

# Show distribution
print("\n" + "=" * 70)
print("Response Time Distribution:")
print("=" * 70)
print(df['response_time_seconds'].describe())