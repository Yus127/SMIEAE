import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_daily_summary_df(df, user_name):
    """Perform exploratory data analysis on a daily summary dataframe"""
    
    print(f"\n{'='*80}")
    print(f"DAILY SUMMARY ANALYSIS - {user_name}")
    print(f"{'='*80}\n")
    
    # Basic Information
    print("=" * 80)
    print("1. BASIC INFORMATION")
    print("=" * 80)
    print(f"Total days: {len(df):,}")
    print(f"Total variables: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'timestamp' in df.columns or 'date' in df.columns:
        date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
        df[date_col] = pd.to_datetime(df[date_col])
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        print(f"Days covered: {(df[date_col].max() - df[date_col].min()).days + 1} days")
    
    # Variable Categories
    print(f"\n{'=' * 80}")
    print("2. VARIABLE CATEGORIES")
    print("=" * 80)
    
    # Group variables by category
    sleep_vars = [col for col in df.columns if 'sleep' in col.lower()]
    hrv_vars = [col for col in df.columns if 'hrv' in col.lower() or 'rmssd' in col.lower()]
    heart_vars = [col for col in df.columns if 'heart' in col.lower() and 'hrv' not in col.lower()]
    respiratory_vars = [col for col in df.columns if 'respiratory' in col.lower() or 'breathing' in col.lower()]
    spo2_vars = [col for col in df.columns if 'spo2' in col.lower() or 'oxygen' in col.lower()]
    activity_vars = [col for col in df.columns if 'activity' in col.lower() or 'steps' in col.lower()]
    other_vars = [col for col in df.columns if col not in sleep_vars + hrv_vars + heart_vars + respiratory_vars + spo2_vars + activity_vars]
    
    print(f"Sleep-related variables: {len(sleep_vars)}")
    print(f"Heart Rate Variability variables: {len(hrv_vars)}")
    print(f"Heart Rate variables: {len(heart_vars)}")
    print(f"Respiratory variables: {len(respiratory_vars)}")
    print(f"SpO2/Oxygen variables: {len(spo2_vars)}")
    print(f"Activity/Steps variables: {len(activity_vars)}")
    print(f"Other variables: {len(other_vars)}")
    
    # Missing Data Analysis
    print(f"\n{'=' * 80}")
    print("3. MISSING DATA ANALYSIS")
    print("=" * 80)
    
    missing_data = pd.DataFrame({
        'Variable': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_%': (df.isnull().sum() / len(df) * 100).round(2),
        'Available_Days': len(df) - df.isnull().sum()
    })
    missing_data = missing_data.sort_values('Missing_%', ascending=False)
    
    print(f"\nVariables with complete data (0% missing): {len(missing_data[missing_data['Missing_%'] == 0])}")
    print(f"Variables with any missing data: {len(missing_data[missing_data['Missing_%'] > 0])}")
    
    print(f"\nTop 20 variables with most missing data:")
    print(missing_data.head(20).to_string(index=False))
    
    # Numeric Statistics by Category
    print(f"\n{'=' * 80}")
    print("4. SLEEP METRICS SUMMARY")
    print("=" * 80)
    
    sleep_numeric = [col for col in sleep_vars if df[col].dtype in ['float64', 'int64']]
    if sleep_numeric:
        stats = df[sleep_numeric].describe().T
        stats['missing_%'] = (df[sleep_numeric].isnull().sum() / len(df) * 100).round(2)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_%']].to_string())
    
    print(f"\n{'=' * 80}")
    print("5. HEART RATE VARIABILITY METRICS SUMMARY")
    print("=" * 80)
    
    hrv_numeric = [col for col in hrv_vars if df[col].dtype in ['float64', 'int64']]
    if hrv_numeric:
        stats = df[hrv_numeric].describe().T
        stats['missing_%'] = (df[hrv_numeric].isnull().sum() / len(df) * 100).round(2)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_%']].to_string())
    
    print(f"\n{'=' * 80}")
    print("6. RESPIRATORY METRICS SUMMARY")
    print("=" * 80)
    
    resp_numeric = [col for col in respiratory_vars if df[col].dtype in ['float64', 'int64']]
    if resp_numeric:
        stats = df[resp_numeric].describe().T
        stats['missing_%'] = (df[resp_numeric].isnull().sum() / len(df) * 100).round(2)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_%']].to_string())
    
    print(f"\n{'=' * 80}")
    print("7. SpO2/OXYGEN METRICS SUMMARY")
    print("=" * 80)
    
    spo2_numeric = [col for col in spo2_vars if df[col].dtype in ['float64', 'int64']]
    if spo2_numeric:
        stats = df[spo2_numeric].describe().T
        stats['missing_%'] = (df[spo2_numeric].isnull().sum() / len(df) * 100).round(2)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_%']].to_string())
    
    print(f"\n{'=' * 80}")
    print("8. ACTIVITY METRICS SUMMARY")
    print("=" * 80)
    
    activity_numeric = [col for col in activity_vars if df[col].dtype in ['float64', 'int64']]
    if activity_numeric:
        stats = df[activity_numeric].describe().T
        stats['missing_%'] = (df[activity_numeric].isnull().sum() / len(df) * 100).round(2)
        print(stats[['count', 'mean', 'std', 'min', '50%', 'max', 'missing_%']].to_string())
    
    # Data Completeness
    print(f"\n{'=' * 80}")
    print("9. DATA COMPLETENESS")
    print("=" * 80)
    
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    print(f"Overall data completeness: {completeness:.2f}%")
    
    # Category-specific completeness
    categories = {
        'Sleep': sleep_vars,
        'HRV': hrv_vars,
        'Heart Rate': heart_vars,
        'Respiratory': respiratory_vars,
        'SpO2': spo2_vars,
        'Activity': activity_vars
    }
    
    print("\nCompleteness by category:")
    for cat_name, cat_vars in categories.items():
        if cat_vars:
            cat_completeness = (1 - df[cat_vars].isnull().sum().sum() / (len(df) * len(cat_vars))) * 100
            print(f"  {cat_name}: {cat_completeness:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("END OF ANALYSIS")
    print(f"{'=' * 80}\n")

def aggregate_daily_summaries():
    """Create master summary of all daily summary files"""
    
    base_dir = "/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit"
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist!")
        return
    
    # Find all daily summary CSV files
    csv_files = list(base_dir.glob("user_*_consolidated_daily_summary.csv"))
    
    if not csv_files:
        print("No daily summary CSV files found!")
        return
    
    print(f"Found {len(csv_files)} daily summary files")
    print("\n" + "="*80)
    print("MASTER SUMMARY - ALL USERS DAILY SUMMARIES")
    print("="*80 + "\n")
    
    # Individual analysis
    print("="*80)
    print("INDIVIDUAL USER ANALYSES")
    print("="*80)
    
    all_users_stats = []
    all_columns = set()
    variable_presence = {}
    
    for csv_file in csv_files:
        user_name = csv_file.stem.replace('_consolidated_daily_summary', '')
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Perform individual analysis
            analyze_daily_summary_df(df, user_name)
            
            # Collect stats
            user_stats = {
                'User': user_name,
                'Days': len(df),
                'Variables': len(df.columns),
                'Memory_MB': df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Date range
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date' if 'date' in df.columns else None
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                user_stats['Start_Date'] = df[date_col].min()
                user_stats['End_Date'] = df[date_col].max()
                user_stats['Days_Covered'] = (df[date_col].max() - df[date_col].min()).days + 1
            
            # Completeness
            user_stats['Completeness_%'] = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            
            # Category counts
            sleep_vars = [col for col in df.columns if 'sleep' in col.lower()]
            hrv_vars = [col for col in df.columns if 'hrv' in col.lower() or 'rmssd' in col.lower()]
            heart_vars = [col for col in df.columns if 'heart' in col.lower() and 'hrv' not in col.lower()]
            respiratory_vars = [col for col in df.columns if 'respiratory' in col.lower() or 'breathing' in col.lower()]
            spo2_vars = [col for col in df.columns if 'spo2' in col.lower() or 'oxygen' in col.lower()]
            activity_vars = [col for col in df.columns if 'activity' in col.lower() or 'steps' in col.lower()]
            
            user_stats['Sleep_Vars'] = len(sleep_vars)
            user_stats['HRV_Vars'] = len(hrv_vars)
            user_stats['Heart_Vars'] = len(heart_vars)
            user_stats['Respiratory_Vars'] = len(respiratory_vars)
            user_stats['SpO2_Vars'] = len(spo2_vars)
            user_stats['Activity_Vars'] = len(activity_vars)
            
            all_users_stats.append(user_stats)
            
            # Track variable presence
            for col in df.columns:
                all_columns.add(col)
                if col not in variable_presence:
                    variable_presence[col] = {'count': 0, 'total_non_null': 0}
                variable_presence[col]['count'] += 1
                variable_presence[col]['total_non_null'] += df[col].notna().sum()
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Create aggregate summary
    summary_df = pd.DataFrame(all_users_stats)
    
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS - ALL USERS")
    print("="*80)
    
    print("\n1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total users: {len(csv_files)}")
    print(f"Total days across all users: {summary_df['Days'].sum():,}")
    print(f"Total unique variables: {len(all_columns)}")
    print(f"Average days per user: {summary_df['Days'].mean():.1f}")
    print(f"Total memory usage: {summary_df['Memory_MB'].sum():.2f} MB")
    
    if 'Start_Date' in summary_df.columns:
        overall_start = summary_df['Start_Date'].min()
        overall_end = summary_df['End_Date'].max()
        print(f"Overall date range: {overall_start} to {overall_end}")
    
    print("\n2. PER-USER SUMMARY")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    print("\n3. CROSS-USER STATISTICS")
    print("-" * 80)
    
    print(f"\nDays per user:")
    print(f"  Mean: {summary_df['Days'].mean():.1f}")
    print(f"  Median: {summary_df['Days'].median():.1f}")
    print(f"  Min: {summary_df['Days'].min()} ({summary_df.loc[summary_df['Days'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Days'].max()} ({summary_df.loc[summary_df['Days'].idxmax(), 'User']})")
    print(f"  Std Dev: {summary_df['Days'].std():.1f}")
    
    print(f"\nVariables per user:")
    print(f"  Mean: {summary_df['Variables'].mean():.1f}")
    print(f"  Median: {summary_df['Variables'].median():.1f}")
    print(f"  Min: {summary_df['Variables'].min()} ({summary_df.loc[summary_df['Variables'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Variables'].max()} ({summary_df.loc[summary_df['Variables'].idxmax(), 'User']})")
    
    print(f"\nData completeness:")
    print(f"  Mean: {summary_df['Completeness_%'].mean():.2f}%")
    print(f"  Median: {summary_df['Completeness_%'].median():.2f}%")
    print(f"  Min: {summary_df['Completeness_%'].min():.2f}% ({summary_df.loc[summary_df['Completeness_%'].idxmin(), 'User']})")
    print(f"  Max: {summary_df['Completeness_%'].max():.2f}% ({summary_df.loc[summary_df['Completeness_%'].idxmax(), 'User']})")
    
    print("\n4. VARIABLE CATEGORIES ACROSS USERS")
    print("-" * 80)
    
    for cat in ['Sleep_Vars', 'HRV_Vars', 'Heart_Vars', 'Respiratory_Vars', 'SpO2_Vars', 'Activity_Vars']:
        if cat in summary_df.columns:
            cat_name = cat.replace('_Vars', '').replace('_', ' ')
            print(f"\n{cat_name} variables per user:")
            print(f"  Mean: {summary_df[cat].mean():.1f}")
            print(f"  Range: {summary_df[cat].min()}-{summary_df[cat].max()}")
    
    print("\n5. VARIABLE PRESENCE ANALYSIS")
    print("-" * 80)
    
    var_presence_df = pd.DataFrame([
        {
            'Variable': var,
            'Present_in_N_Users': info['count'],
            'Presence_%': (info['count'] / len(csv_files) * 100),
            'Total_Non_Null_Days': info['total_non_null']
        }
        for var, info in variable_presence.items()
    ]).sort_values('Presence_%', ascending=False)
    
    universal_vars = var_presence_df[var_presence_df['Presence_%'] == 100]
    print(f"\nVariables present in ALL users: {len(universal_vars)}")
    
    partial_vars = var_presence_df[(var_presence_df['Presence_%'] < 100) & (var_presence_df['Presence_%'] >= 50)]
    print(f"Variables present in ≥50% of users: {len(partial_vars)}")
    
    rare_vars = var_presence_df[var_presence_df['Presence_%'] < 50]
    print(f"Variables present in <50% of users: {len(rare_vars)}")
    
    print("\n6. DATA QUALITY SUMMARY")
    print("-" * 80)
    
    high_quality = summary_df[summary_df['Completeness_%'] >= 80]
    medium_quality = summary_df[(summary_df['Completeness_%'] >= 60) & (summary_df['Completeness_%'] < 80)]
    low_quality = summary_df[summary_df['Completeness_%'] < 60]
    
    print(f"High quality (≥80% complete): {len(high_quality)} users")
    print(f"Medium quality (60-80% complete): {len(medium_quality)} users")
    print(f"Low quality (<60% complete): {len(low_quality)} users")
    
    # Save outputs
    output_dir = base_dir
    summary_csv = output_dir / "daily_summary_users_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved user statistics to: {summary_csv}")
    
    var_presence_csv = output_dir / "daily_summary_variable_presence.csv"
    var_presence_df.to_csv(var_presence_csv, index=False)
    print(f"✓ Saved variable presence to: {var_presence_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    aggregate_daily_summaries()