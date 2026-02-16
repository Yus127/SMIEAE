import pandas as pd
import numpy as np
from pathlib import Path
import os

def analyze_completeness(input_dir, output_file="data_completeness_report.csv", file_pattern="*_daily_summary.csv"):
    """
    Analyze data completeness across all user files.
    
    Args:
        input_dir: Directory containing the CSV files
        output_file: Name of the output report file
        file_pattern: Pattern to match files (default: daily summary files)
    
    Returns:
        DataFrame with completeness statistics
    """
    csv_files = list(Path(input_dir).glob(file_pattern))
    
    if not csv_files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return None
    
    print(f"Found {len(csv_files)} files to analyze\n")
    
    # Dictionary to store completeness data
    completeness_data = {}
    total_rows_per_user = {}
    
    # Process each file
    for csv_file in csv_files:
        try:
            # Extract user identifier from filename
            user_id = csv_file.stem.replace('_consolidated_daily_summary', '').replace('_daily_summary', '')
            
            df = pd.read_csv(csv_file)
            
            # Store total rows for this user
            total_rows_per_user[user_id] = len(df)
            
            # Calculate completeness for each column
            for column in df.columns:
                non_null_count = df[column].notna().sum()
                
                if column not in completeness_data:
                    completeness_data[column] = {
                        'total_values': 0,
                        'non_null_values': 0,
                        'users_with_data': 0,
                        'users_checked': 0
                    }
                
                completeness_data[column]['total_values'] += len(df)
                completeness_data[column]['non_null_values'] += non_null_count
                completeness_data[column]['users_checked'] += 1
                
                if non_null_count > 0:
                    completeness_data[column]['users_with_data'] += 1
            
            print(f"Processed: {csv_file.name} ({len(df)} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    # Create summary DataFrame
    summary_data = []
    
    for column, stats in completeness_data.items():
        total_values = stats['total_values']
        non_null_values = stats['non_null_values']
        users_with_data = stats['users_with_data']
        users_checked = stats['users_checked']
        
        # Calculate percentages
        completeness_pct = (non_null_values / total_values * 100) if total_values > 0 else 0
        users_with_data_pct = (users_with_data / users_checked * 100) if users_checked > 0 else 0
        
        summary_data.append({
            'column_name': column,
            'total_possible_values': total_values,
            'non_null_values': non_null_values,
            'null_values': total_values - non_null_values,
            'completeness_percentage': round(completeness_pct, 2),
            'users_with_data': users_with_data,
            'total_users': users_checked,
            'users_with_data_percentage': round(users_with_data_pct, 2)
        })
    
    # Convert to DataFrame and sort by completeness
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('completeness_percentage', ascending=True)
    
    # Save to CSV
    output_path = os.path.join(input_dir, output_file)
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"COMPLETENESS ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total users analyzed: {len(csv_files)}")
    print(f"Total columns found: {len(summary_df)}")
    print(f"\nReport saved to: {output_path}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("COMPLETENESS DISTRIBUTION:")
    print(f"{'='*80}")
    
    ranges = [
        (0, 10, "0-10% complete (Consider removing)"),
        (10, 25, "10-25% complete (Very sparse)"),
        (25, 50, "25-50% complete (Sparse)"),
        (50, 75, "50-75% complete (Moderate)"),
        (75, 90, "75-90% complete (Good)"),
        (90, 100, "90-100% complete (Excellent)")
    ]
    
    for low, high, label in ranges:
        count = len(summary_df[(summary_df['completeness_percentage'] >= low) & 
                                (summary_df['completeness_percentage'] < high)])
        print(f"{label:40s}: {count:4d} columns")
    
    # Show columns with very low completeness (< 10%)
    low_completeness = summary_df[summary_df['completeness_percentage'] < 10]
    if len(low_completeness) > 0:
        print(f"\n{'='*80}")
        print(f"COLUMNS WITH < 10% COMPLETENESS (CONSIDER REMOVING):")
        print(f"{'='*80}")
        for _, row in low_completeness.iterrows():
            print(f"{row['column_name']:60s} {row['completeness_percentage']:6.2f}%  ({row['users_with_data']}/{row['total_users']} users)")
    
    # Show columns with excellent completeness (> 90%)
    high_completeness = summary_df[summary_df['completeness_percentage'] > 90]
    if len(high_completeness) > 0:
        print(f"\n{'='*80}")
        print(f"COLUMNS WITH > 90% COMPLETENESS (RELIABLE):")
        print(f"{'='*80}")
        for _, row in high_completeness.head(20).iterrows():
            print(f"{row['column_name']:60s} {row['completeness_percentage']:6.2f}%  ({row['users_with_data']}/{row['total_users']} users)")
    
    return summary_df


def analyze_timeseries_completeness(input_dir, output_file="timeseries_completeness_report.csv"):
    """
    Analyze completeness for time-series data files.
    
    Args:
        input_dir: Directory containing the CSV files
        output_file: Name of the output report file
    
    Returns:
        DataFrame with completeness statistics
    """
    return analyze_completeness(input_dir, output_file, file_pattern="*_consolidated_daily_summary.csv")


def create_recommendations(summary_df, completeness_threshold=10, users_threshold=0.25):
    """
    Create recommendations for which columns to keep or remove.
    
    Args:
        summary_df: DataFrame from analyze_completeness
        completeness_threshold: Minimum completeness percentage to keep
        users_threshold: Minimum fraction of users that must have data
    
    Returns:
        Dictionary with recommendations
    """
    total_users = summary_df['total_users'].iloc[0]
    min_users = int(total_users * users_threshold)
    
    # Columns to remove
    remove_cols = summary_df[
        (summary_df['completeness_percentage'] < completeness_threshold) |
        (summary_df['users_with_data'] < min_users)
    ]['column_name'].tolist()
    
    # Columns to keep
    keep_cols = summary_df[
        (summary_df['completeness_percentage'] >= completeness_threshold) &
        (summary_df['users_with_data'] >= min_users)
    ]['column_name'].tolist()
    
    recommendations = {
        'columns_to_remove': remove_cols,
        'columns_to_keep': keep_cols,
        'removal_criteria': {
            'completeness_threshold': completeness_threshold,
            'min_users_required': min_users,
            'min_users_percentage': users_threshold * 100
        },
        'summary': {
            'total_columns': len(summary_df),
            'columns_to_remove_count': len(remove_cols),
            'columns_to_keep_count': len(keep_cols),
            'removal_percentage': round(len(remove_cols) / len(summary_df) * 100, 2)
        }
    }
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print(f"Criteria:")
    print(f"  - Minimum completeness: {completeness_threshold}%")
    print(f"  - Minimum users with data: {min_users} ({users_threshold*100}% of {total_users} users)")
    print(f"\nResults:")
    print(f"  - Columns to REMOVE: {len(remove_cols)} ({recommendations['summary']['removal_percentage']}%)")
    print(f"  - Columns to KEEP: {len(keep_cols)}")
    
    return recommendations


# Example usage:
if __name__ == "__main__":
    # Analyze daily summary files
    print("ANALYZING DAILY SUMMARY FILES:")
    daily_summary = analyze_completeness("processed_data", output_file="daily_completeness_report.csv")
    
    if daily_summary is not None:
        # Create recommendations
        recommendations = create_recommendations(
            daily_summary, 
            completeness_threshold=10,  # Remove columns with < 10% data
            users_threshold=0.25  # Remove columns present in < 25% of users
        )
        
        # Save recommendations to file
        with open("processed_data/column_removal_recommendations.txt", "w") as f:
            f.write("COLUMNS TO REMOVE:\n")
            f.write("="*80 + "\n")
            for col in recommendations['columns_to_remove']:
                f.write(f"{col}\n")
            f.write(f"\nTotal: {len(recommendations['columns_to_remove'])} columns\n")
        
        print(f"\nRecommendations saved to: processed_data/column_removal_recommendations.txt")
    
    # Optionally analyze time-series files
    print("\n\n")
    print("ANALYZING TIME-SERIES FILES:")
    timeseries_summary = analyze_timeseries_completeness("/Users/YusMolina/Downloads/smieae/data/data_clean/fitbit", output_file="timeseries_completeness_report.csv")