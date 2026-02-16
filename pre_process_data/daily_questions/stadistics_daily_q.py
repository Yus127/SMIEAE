import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

def check_date_continuity_single_user(df, user_id):
    """Check for missing days for a single user"""
    
    print(f"\n{'='*80}")
    print(f"DATE CONTINUITY ANALYSIS - USER {user_id}")
    print(f"{'='*80}\n")
    
    # Determine date column
    date_col = None
    if 'date_only' in df.columns:
        date_col = 'date_only'
    elif 'date' in df.columns:
        date_col = 'date'
    elif 'timestamp' in df.columns:
        date_col = 'timestamp'
    
    if date_col is None:
        print("ERROR: No date column found!")
        return None
    
    # Convert to datetime and extract date only
    df[date_col] = pd.to_datetime(df[date_col])
    df['date_extracted'] = df[date_col].dt.date
    
    # Get unique dates with responses
    response_dates = sorted(df['date_extracted'].unique())
    
    if len(response_dates) == 0:
        print("No response dates found!")
        return None
    
    # Basic stats
    start_date = response_dates[0]
    end_date = response_dates[-1]
    total_days_in_range = (end_date - start_date).days + 1
    days_with_responses = len(response_dates)
    missing_days = total_days_in_range - days_with_responses
    completion_rate = (days_with_responses / total_days_in_range) * 100
    
    print("1. BASIC STATISTICS")
    print(f"First response: {start_date}")
    print(f"Last response: {end_date}")
    print(f"Total days in range: {total_days_in_range}")
    print(f"Days with responses: {days_with_responses}")
    print(f"Days without responses (missing): {missing_days}")
    print(f"Completion rate: {completion_rate:.2f}%")
    
    # Check for multiple responses per day
    responses_per_day = df.groupby('date_extracted').size()
    days_with_multiple = responses_per_day[responses_per_day > 1]
    
    if len(days_with_multiple) > 0:
        print(f"\nDays with multiple responses: {len(days_with_multiple)}")
        print(f"Max responses in a single day: {responses_per_day.max()}")
        print("\nDays with multiple responses:")
        for date, count in days_with_multiple.items():
            print(f"  {date}: {count} responses")
    else:
        print("\nAll days have at most 1 response (no duplicates)")
    
    # Find missing dates
    print(f"\n{'=' * 80}")
    print("2. MISSING DATES ANALYSIS")
    
    if missing_days > 0:
        # Generate complete date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_dates = set(date_range.date)
        response_dates_set = set(response_dates)
        missing_dates = sorted(all_dates - response_dates_set)
        
        print(f"\nTotal missing days: {len(missing_dates)}")
        print(f"\nMissing dates:")
        for date in missing_dates:
            weekday = pd.Timestamp(date).day_name()
            print(f"  {date} ({weekday})")
        
        # Analyze missing date patterns
        print(f"\n{'=' * 80}")
        print("3. MISSING DATE PATTERNS")
        
        # Missing days by weekday
        missing_weekdays = [pd.Timestamp(date).day_name() for date in missing_dates]
        weekday_counts = pd.Series(missing_weekdays).value_counts()
        
        print("\nMissing days by weekday:")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            count = weekday_counts.get(day, 0)
            if count > 0:
                print(f"  {day}: {count} missing days")
        
        # Find consecutive missing days (gaps)
        gaps = []
        current_gap = []
        
        for i, date in enumerate(missing_dates):
            if i == 0:
                current_gap = [date]
            else:
                prev_date = missing_dates[i-1]
                if (date - prev_date).days == 1:
                    current_gap.append(date)
                else:
                    if len(current_gap) > 0:
                        gaps.append(current_gap)
                    current_gap = [date]
            
            if i == len(missing_dates) - 1 and len(current_gap) > 0:
                gaps.append(current_gap)
        
        if gaps:
            print(f"\nConsecutive missing day periods (gaps):")
            print(f"Total gaps: {len(gaps)}")
            
            # Sort gaps by length
            gaps_sorted = sorted(gaps, key=len, reverse=True)
            
            print(f"\nLongest gaps:")
            for i, gap in enumerate(gaps_sorted[:10], 1):
                print(f"  Gap {i}: {gap[0]} to {gap[-1]} ({len(gap)} days)")
        
        # Calculate longest streak of consecutive responses
        print(f"\n{'=' * 80}")
        print("4. RESPONSE STREAKS")
        
        response_dates_sorted = sorted(response_dates)
        current_streak = 1
        max_streak = 1
        max_streak_start = response_dates_sorted[0]
        max_streak_end = response_dates_sorted[0]
        current_streak_start = response_dates_sorted[0]
        
        for i in range(1, len(response_dates_sorted)):
            if (response_dates_sorted[i] - response_dates_sorted[i-1]).days == 1:
                current_streak += 1
            else:
                if current_streak > max_streak:
                    max_streak = current_streak
                    max_streak_start = current_streak_start
                    max_streak_end = response_dates_sorted[i-1]
                current_streak = 1
                current_streak_start = response_dates_sorted[i]
        
        # Check final streak
        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_start = current_streak_start
            max_streak_end = response_dates_sorted[-1]
        
        print(f"Longest consecutive response streak: {max_streak} days")
        print(f"  From: {max_streak_start}")
        print(f"  To: {max_streak_end}")
        
    else:
        print("\nNo missing days! User answered every day in the date range.")
        print(f"Perfect completion: {days_with_responses} consecutive days")
    
    # Summary statistics
    print(f"\n{'=' * 80}")
    print("5. SUMMARY")
    print(f"Completion rate: {completion_rate:.2f}%")
    print(f"Response consistency: ", end="")
    
    if completion_rate == 100:
        print("PERFECT ")
    elif completion_rate >= 90:
        print("EXCELLENT (≥90%)")
    elif completion_rate >= 75:
        print("GOOD (≥75%)")
    elif completion_rate >= 50:
        print("FAIR (≥50%)")
    else:
        print("POOR (<50%)")
    
    return {
        'user_id': user_id,
        'start_date': start_date,
        'end_date': end_date,
        'total_days': total_days_in_range,
        'days_with_responses': days_with_responses,
        'missing_days': missing_days,
        'completion_rate': completion_rate,
        'days_with_multiple_responses': len(days_with_multiple),
        'max_responses_per_day': responses_per_day.max() if len(responses_per_day) > 0 else 0
    }

def analyze_continuity_per_user():
    """Analyze date continuity for each user in the CSV"""
    
    # Path to the daily questions CSV file
    csv_file = "/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Error: File {csv_file} does not exist!")
        print("Please update the csv_file path in the script.")
        return
    
    print("DATE CONTINUITY ANALYSIS - PER USER")
    print(f"Analyzing file: {csv_path.name}\n")
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"Loaded {len(df)} total responses")
        
        # Check if userid column exists
        if 'userid' not in df.columns:
            print("ERROR: 'userid' column not found in CSV!")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Get unique users
        users = sorted(df['userid'].unique())
        print(f"Found {len(users)} unique users: {users}\n")
        
        # Analyze each user
        all_stats = []
        
        for user_id in users:
            user_df = df[df['userid'] == user_id].copy()
            print(f"\nAnalyzing User {user_id} ({len(user_df)} responses)...")
            
            stats = check_date_continuity_single_user(user_df, user_id)
            
            if stats:
                all_stats.append(stats)
        
        # Create aggregate summary
        if all_stats:
            print("\n" + "="*80)
            print("AGGREGATE SUMMARY - ALL USERS")
            print("="*80 + "\n")
            
            summary_df = pd.DataFrame(all_stats)
            
            print("1. OVERVIEW")
            print("-" * 80)
            print(summary_df.to_string(index=False))
            
            print("\n2. COMPLETION STATISTICS")
            print("-" * 80)
            print(f"Average completion rate: {summary_df['completion_rate'].mean():.2f}%")
            print(f"Median completion rate: {summary_df['completion_rate'].median():.2f}%")
            print(f"Best completion rate: {summary_df['completion_rate'].max():.2f}% (User {summary_df.loc[summary_df['completion_rate'].idxmax(), 'user_id']})")
            print(f"Worst completion rate: {summary_df['completion_rate'].min():.2f}% (User {summary_df.loc[summary_df['completion_rate'].idxmin(), 'user_id']})")
            
            print(f"\nTotal missing days across all users: {summary_df['missing_days'].sum()}")
            print(f"Average missing days per user: {summary_df['missing_days'].mean():.1f}")
            
            # Categorize users
            perfect = summary_df[summary_df['completion_rate'] == 100]
            excellent = summary_df[(summary_df['completion_rate'] >= 90) & (summary_df['completion_rate'] < 100)]
            good = summary_df[(summary_df['completion_rate'] >= 75) & (summary_df['completion_rate'] < 90)]
            fair = summary_df[(summary_df['completion_rate'] >= 50) & (summary_df['completion_rate'] < 75)]
            poor = summary_df[summary_df['completion_rate'] < 50]
            
            print("\n3. USER CATEGORIES BY COMPLETION")
            print("-" * 80)
            print(f"Perfect (100%): {len(perfect)} users")
            if len(perfect) > 0:
                for _, row in perfect.iterrows():
                    print(f"  - User {row['user_id']}: {row['days_with_responses']} days")
            
            print(f"\nExcellent (90-99%): {len(excellent)} users")
            if len(excellent) > 0:
                for _, row in excellent.iterrows():
                    print(f"  - User {row['user_id']}: {row['completion_rate']:.2f}% ({row['missing_days']} missing)")
            
            print(f"\nGood (75-89%): {len(good)} users")
            if len(good) > 0:
                for _, row in good.iterrows():
                    print(f"  - User {row['user_id']}: {row['completion_rate']:.2f}% ({row['missing_days']} missing)")
            
            print(f"\nFair (50-74%): {len(fair)} users")
            if len(fair) > 0:
                for _, row in fair.iterrows():
                    print(f"  - User {row['user_id']}: {row['completion_rate']:.2f}% ({row['missing_days']} missing)")
            
            print(f"\nPoor (<50%): {len(poor)} users")
            if len(poor) > 0:
                for _, row in poor.iterrows():
                    print(f"  - User {row['user_id']}: {row['completion_rate']:.2f}% ({row['missing_days']} missing)")
            
            # Multiple responses
            users_with_duplicates = summary_df[summary_df['days_with_multiple_responses'] > 0]
            if len(users_with_duplicates) > 0:
                print("\n4. USERS WITH MULTIPLE RESPONSES PER DAY")
                print("-" * 80)
                for _, row in users_with_duplicates.iterrows():
                    print(f"  - User {row['user_id']}: {row['days_with_multiple_responses']} days with multiple responses")
            
            # Save summary
            output_dir = csv_path.parent
            output_file = output_dir / "date_continuity_summary_per_user.csv"
            summary_df.to_csv(output_file, index=False)
            print(f"\n Saved continuity summary to: {output_file}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_continuity_per_user()