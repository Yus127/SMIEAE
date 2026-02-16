import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_dataframe(df, user_name):
    #Perform comprehensive exploratory data analysis on a dataframe
    
    print(f"\n{'='*80}")
    print(f"EXPLORATORY DATA ANALYSIS - {user_name}")
    print(f"{'='*80}\n")
    
    # Basic Information
    print("1. BASIC INFORMATION")
    print(f"Total number of rows: {len(df):,}")
    print(f"Total number of columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Total days covered: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Column Information
    print(f"\n{'=' * 80}")
    print("2. VARIABLES (COLUMNS)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print(f"Numeric variables: {len(numeric_cols)}")
    print(f"Categorical variables: {len(categorical_cols)}")
    print(f"Datetime variables: {len(datetime_cols)}")
    
    # List all variables by category
    print(f"\n--- Numeric Variables ({len(numeric_cols)}) ---")
    for col in numeric_cols[:50]:  # Show first 50
        print(f"  - {col}")
    if len(numeric_cols) > 50:
        print(f"  ... and {len(numeric_cols) - 50} more")
    
    if categorical_cols:
        print(f"\n--- Categorical Variables ({len(categorical_cols)}) ---")
        for col in categorical_cols[:20]:
            print(f"  - {col}")
        if len(categorical_cols) > 20:
            print(f"  ... and {len(categorical_cols) - 20} more")
    
    if datetime_cols:
        print(f"\n--- Datetime Variables ({len(datetime_cols)}) ---")
        for col in datetime_cols:
            print(f"  - {col}")
    
    # Missing Data Analysis
    print(f"\n{'=' * 80}")
    print("3. MISSING DATA ANALYSIS")
    
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_data) > 0:
        print(f"Columns with missing data: {len(missing_data)} out of {len(df.columns)}")
        print(f"\nTop 20 columns with most missing data:")
        print(missing_data.head(20).to_string(index=False))
    else:
        print("No missing data found!")
    
    # Numeric Variables Statistics
    print(f"\n{'=' * 80}")
    print("4. NUMERIC VARIABLES - DESCRIPTIVE STATISTICS")
    
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        stats['missing_%'] = (df[numeric_cols].isnull().sum() / len(df) * 100).round(2)
        stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_%']]
        
        # Show all statistics
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print("\nComplete Statistics for All Numeric Variables:")
        print(stats.to_string())
        
        # Reset display options
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
    
    # Data Distribution Summary
    print(f"\n{'=' * 80}")
    print("5. DATA DISTRIBUTION SUMMARY")
    
    if len(numeric_cols) > 0:
        print("\nVariables with potential outliers (values beyond 3 std from mean):")
        outlier_summary = []
        
        for col in numeric_cols[:30]:  # Check first 30 numeric columns
            if df[col].notna().sum() > 0:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    outliers = ((df[col] < mean - 3*std) | (df[col] > mean + 3*std)).sum()
                    outlier_pct = (outliers / df[col].notna().sum() * 100)
                    if outlier_pct > 0:
                        outlier_summary.append({
                            'Variable': col,
                            'Outliers': outliers,
                            'Percentage': f"{outlier_pct:.2f}%"
                        })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary).sort_values('Outliers', ascending=False)
            print(outlier_df.head(15).to_string(index=False))
        else:
            print("No significant outliers detected in sampled variables.")
    
    # Temporal Analysis (if timestamp exists)
    if 'timestamp' in df.columns:
        print(f"\n{'=' * 80}")
        print("6. TEMPORAL ANALYSIS")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        
        print("\nData points by hour of day:")
        hour_dist = df['hour'].value_counts().sort_index()
        for hour, count in hour_dist.items():
            print(f"  Hour {hour:02d}:00 - {count:,} records")
        
        print("\nData points by day of week:")
        dow_dist = df['day_of_week'].value_counts()
        for day, count in dow_dist.items():
            print(f"  {day}: {count:,} records")
        
        print("\nData points by month:")
        month_dist = df['month'].value_counts()
        for month, count in month_dist.items():
            print(f"  {month}: {count:,} records")
    
    # Completeness Score
    print(f"\n{'=' * 80}")
    print("7. DATA COMPLETENESS SCORE")
    
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    print(f"Overall data completeness: {completeness:.2f}%")
    
    # Variable type breakdown
    print(f"\n{'=' * 80}")
    print("8. VARIABLE CATEGORIES (BY NAME PREFIX)")
    
    # Group variables by their prefix (source)
    prefixes = {}
    for col in df.columns:
        if col == 'timestamp':
            continue
        prefix = col.split('_')[0] if '_' in col else 'other'
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(col)
    
    print("\nVariables grouped by data source:")
    for prefix, cols in sorted(prefixes.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {prefix}: {len(cols)} variables")
    
    print(f"\n{'=' * 80}")
    print("END OF ANALYSIS")
    print(f"{'=' * 80}\n")



def aggregate_all_users():
    """Create a master summary aggregating all user data"""
    
    base_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output"
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return
    
    # Find all consolidated CSV files
    csv_files = list(base_dir.glob("user_*_consolidated.csv"))
    
    if not csv_files:
        print("No consolidated CSV files found!")
        return
    
    print(f"Found {len(csv_files)} user files")
    print("\n" + "="*80)
    print("MASTER SUMMARY - ALL USERS AGGREGATE ANALYSIS")
    print("="*80 + "\n")
    
    # Collect aggregate statistics
    all_users_stats = []
    all_columns = set()
    variable_presence = {}
    total_records = 0
    date_ranges = []
    
    
    for csv_file in csv_files:
        user_name = csv_file.stem.replace('_consolidated', '')
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Basic stats for this user
            user_stats = {
                'User': user_name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory_MB': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Date range
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                user_stats['Start_Date'] = df['timestamp'].min()
                user_stats['End_Date'] = df['timestamp'].max()
                user_stats['Days_Covered'] = (df['timestamp'].max() - df['timestamp'].min()).days
                date_ranges.append((df['timestamp'].min(), df['timestamp'].max()))
            
            # Completeness
            user_stats['Completeness_%'] = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            
            # Count numeric vs categorical
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            user_stats['Numeric_Variables'] = len(numeric_cols)
            user_stats['Categorical_Variables'] = len(categorical_cols)
            
            all_users_stats.append(user_stats)
            total_records += len(df)
            
            # Track which variables are present
            for col in df.columns:
                all_columns.add(col)
                if col not in variable_presence:
                    variable_presence[col] = {'count': 0, 'total_non_null': 0}
                variable_presence[col]['count'] += 1
                variable_presence[col]['total_non_null'] += df[col].notna().sum()
            
        except Exception as e:
            print(f"    Error processing {csv_file.name}: {e}")
            continue
    
    # Create summary dataframe
    summary_df = pd.DataFrame(all_users_stats)
    
    # Print overall statistics
    print("\n" + "="*80)
    print("1. OVERALL DATASET STATISTICS")
    print(f"Total number of users: {len(csv_files)}")
    print(f"Total records across all users: {total_records:,}")
    print(f"Total unique variables: {len(all_columns)}")
    print(f"Average records per user: {total_records/len(csv_files):,.0f}")
    print(f"Total memory usage: {summary_df['Memory_MB'].sum():.2f} MB")
    
    if date_ranges:
        overall_start = min([d[0] for d in date_ranges])
        overall_end = max([d[1] for d in date_ranges])
        print(f"Overall date range: {overall_start} to {overall_end}")
        print(f"Total days covered: {(overall_end - overall_start).days} days")
    
    # Per-user summary
    print("\n" + "="*80)
    print("2. PER-USER SUMMARY")
    print(summary_df.to_string(index=False))
    
    # Statistics about statistics
    print("\n" + "="*80)
    print("3. CROSS-USER STATISTICS")
    print(f"\nRows per user:")
    print(f"  Mean: {summary_df['Rows'].mean():,.0f}")
    print(f"  Median: {summary_df['Rows'].median():,.0f}")
    print(f"  Min: {summary_df['Rows'].min():,} ({summary_df.loc[summary_df['Rows'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Rows'].max():,} ({summary_df.loc[summary_df['Rows'].idxmax(), 'User']})")
    print(f"  Std Dev: {summary_df['Rows'].std():,.0f}")
    
    print(f"\nColumns per user:")
    print(f"  Mean: {summary_df['Columns'].mean():.0f}")
    print(f"  Median: {summary_df['Columns'].median():.0f}")
    print(f"  Min: {summary_df['Columns'].min()} ({summary_df.loc[summary_df['Columns'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Columns'].max()} ({summary_df.loc[summary_df['Columns'].idxmax(), 'User']})")
    
    print(f"\nData completeness across users:")
    print(f"  Mean: {summary_df['Completeness_%'].mean():.2f}%")
    print(f"  Median: {summary_df['Completeness_%'].median():.2f}%")
    print(f"  Min: {summary_df['Completeness_%'].min():.2f}% ({summary_df.loc[summary_df['Completeness_%'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Completeness_%'].max():.2f}% ({summary_df.loc[summary_df['Completeness_%'].idxmax(), 'User']})")
    
    if 'Days_Covered' in summary_df.columns:
        print(f"\nDays covered per user:")
        print(f"  Mean: {summary_df['Days_Covered'].mean():.0f}")
        print(f"  Median: {summary_df['Days_Covered'].median():.0f}")
        print(f"  Min: {summary_df['Days_Covered'].min()} ({summary_df.loc[summary_df['Days_Covered'].idxmin(), 'User']})")
        print(f"  Max: {summary_df['Days_Covered'].max()} ({summary_df.loc[summary_df['Days_Covered'].idxmax(), 'User']})")
    
    # Variable presence analysis
    print("\n" + "="*80)
    print("4. VARIABLE PRESENCE ACROSS USERS")
    
    var_presence_df = pd.DataFrame([
        {
            'Variable': var,
            'Present_in_N_Users': info['count'],
            'Presence_%': (info['count'] / len(csv_files) * 100),
            'Total_Non_Null_Values': info['total_non_null']
        }
        for var, info in variable_presence.items()
    ]).sort_values('Presence_%', ascending=False)
    
    print(f"\nTotal unique variables across all users: {len(var_presence_df)}")
    
    # Variables present in all users
    universal_vars = var_presence_df[var_presence_df['Presence_%'] == 100]
    print(f"\nVariables present in ALL users ({len(universal_vars)}):")
    for var in universal_vars['Variable'].head(30):
        print(f"  - {var}")
    if len(universal_vars) > 30:
        print(f"  ... and {len(universal_vars) - 30} more")
    
    # Variables present in only some users
    partial_vars = var_presence_df[(var_presence_df['Presence_%'] < 100) & (var_presence_df['Presence_%'] > 0)]
    print(f"\nVariables present in SOME users ({len(partial_vars)}):")
    print(partial_vars.head(20).to_string(index=False))
    
    # Variable categories
    print("\n" + "="*80)
    print("5. VARIABLE CATEGORIES (BY PREFIX)")
    
    prefixes = {}
    for var in all_columns:
        if var == 'timestamp':
            continue
        prefix = var.split('_')[0] if '_' in var else 'other'
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(var)
    
    print("\nVariables grouped by data source:")
    for prefix, vars in sorted(prefixes.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {prefix}: {len(vars)} variables")
    
    # Data quality summary
    print("\n" + "="*80)
    print("6. DATA QUALITY SUMMARY")
    
    high_quality = summary_df[summary_df['Completeness_%'] >= 80]
    medium_quality = summary_df[(summary_df['Completeness_%'] >= 60) & (summary_df['Completeness_%'] < 80)]
    low_quality = summary_df[summary_df['Completeness_%'] < 60]
    
    print(f"High quality (≥80% complete): {len(high_quality)} users")
    print(f"Medium quality (60-80% complete): {len(medium_quality)} users")
    print(f"Low quality (<60% complete): {len(low_quality)} users")
    
    if len(low_quality) > 0:
        print(f"\nUsers with low data quality:")
        for _, row in low_quality.iterrows():
            print(f"  - {row['User']}: {row['Completeness_%']:.2f}% complete")
    
    # Save master summary
    output_file = base_dir / "MASTER_SUMMARY_ALL_USERS.txt"
    print(f"\n" + "="*80)
    print(f"Saving master summary to: {output_file}")
    
    # Save detailed CSV
    summary_csv = base_dir / "users_summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved detailed statistics to: {summary_csv}")
    
    var_presence_csv = base_dir / "variable_presence_across_users.csv"
    var_presence_df.to_csv(var_presence_csv, index=False)
    print(f"Saved variable presence data to: {var_presence_csv}")
    
    print("\n" + "="*80)
    print("MASTER SUMMARY COMPLETE")

def main():
    #Main execution function
    base_dir = "/Users/YusMolina/Downloads/smieae/data/original_data/fitbit/consolidated_output"
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return
    
    # Find all consolidated CSV files
    csv_files = list(base_dir.glob("user_*_consolidated.csv"))
    
    if not csv_files:
        print("No consolidated CSV files found!")
        return
    
    print(f"Found {len(csv_files)} consolidated CSV files")
    
    # Analyze each user's data
    for csv_file in csv_files:
        user_name = csv_file.stem.replace('_consolidated', '')
        print(f"\nLoading {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            analyze_dataframe(df, user_name)
            
            output_file = base_dir / f"{user_name}_analysis_summary.txt"
            print(f"\nSaving analysis summary to: {output_file}")
            
        except Exception as e:
            print(f"Error analyzing {csv_file.name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE")
    print("GENERATE SUMMARY OF ALL THE SUMMARIES")
    aggregate_all_users()

if __name__ == "__main__":
    main()
