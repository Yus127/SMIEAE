import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create the dataset
data = {
    'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39,
                41, 42, 44, 46, 47, 50, 54, 55],
    'start_date': ['2025-02-19', '2025-02-18', '2025-02-26', '2025-02-14', '2025-02-14',
                   '2025-02-21', '2025-02-17', '2025-02-17', '2025-02-22', '2025-02-20',
                   '2025-02-25', '2025-02-16', '2025-02-19', '2025-02-20', '2025-02-24',
                   '2025-02-17', '2025-02-24', '2025-02-24', '2025-02-19', '2025-02-19',
                   '2025-03-13', '2025-03-05', '2025-03-09', '2025-03-19', '2025-03-13',
                   '2025-03-06', '2025-03-07', '2025-03-10', '2025-03-13', '2025-03-16',
                   '2025-03-13', '2025-03-06', '2025-03-04', '2025-03-06', '2025-04-06',
                   '2025-09-27', '2025-10-14', '2025-09-26', '2025-09-26', '2025-09-25',
                   '2025-09-27', '2025-10-09', '2025-10-10', '2025-10-15', '2025-10-21',
                   '2025-10-13'],
    'end_date': ['2025-06-22', '2025-06-13', '2025-04-29', '2025-06-13', '2025-06-29',
                 '2025-06-20', '2025-06-11', '2025-06-11', '2025-06-13', '2025-06-22',
                 '2025-05-15', '2025-06-03', '2025-06-07', '2025-04-12', '2025-06-07',
                 '2025-06-19', '2025-06-21', '2025-06-09', '2025-06-23', '2025-06-20',
                 '2025-06-29', '2025-07-09', '2025-06-30', '2025-07-04', '2025-05-15',
                 '2025-07-02', '2025-07-03', '2025-06-30', '2025-07-08', '2025-07-04',
                 '2025-05-14', '2025-07-03', '2025-07-01', '2025-07-04', '2025-06-30',
                 '2025-11-22', '2025-12-03', '2025-12-03', '2025-12-01', '2025-12-03',
                 '2025-10-28', '2025-12-03', '2025-12-02', '2025-12-03', '2025-10-21',
                 '2025-12-02'],
    'total_days': [124, 116, 63, 120, 136, 120, 115, 115, 112, 123, 80, 108, 109, 52, 104,
                   123, 118, 106, 125, 122, 109, 127, 114, 108, 64, 119, 119, 113, 118, 111,
                   63, 120, 120, 121, 86, 57, 51, 69, 67, 70, 32, 56, 54, 50, 1, 51],
    'days_with_responses': [121, 102, 14, 118, 135, 105, 103, 104, 60, 115, 39, 55, 88, 40, 59,
                            71, 116, 83, 116, 100, 59, 117, 76, 108, 7, 102, 116, 108, 115, 108,
                            50, 116, 115, 111, 81, 28, 49, 65, 56, 68, 17, 55, 42, 48, 1, 41],
    'missing_days': [3, 14, 49, 2, 1, 15, 12, 11, 52, 8, 41, 53, 21, 12, 45,
                     52, 2, 23, 9, 22, 50, 10, 38, 0, 57, 17, 3, 5, 3, 3,
                     13, 4, 5, 10, 5, 29, 2, 4, 11, 2, 15, 1, 12, 2, 0, 10],
    'completion_rate': [97.581, 87.931, 22.222, 98.333, 99.265, 87.500, 89.565, 90.435, 53.571, 93.496,
                        48.750, 50.926, 80.734, 76.923, 56.731, 57.724, 98.305, 78.302, 92.800, 81.967,
                        54.128, 92.126, 66.667, 100.000, 10.938, 85.714, 97.479, 95.575, 97.458, 97.297,
                        79.365, 96.667, 95.833, 91.736, 94.186, 49.123, 96.078, 94.203, 83.582, 97.143,
                        53.125, 98.214, 77.778, 96.000, 100.000, 80.392]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert date columns to datetime
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Create performance categories
def categorize_performance(rate):
    if rate >= 90:
        return 'Excellent (≥90%)'
    elif rate >= 70:
        return 'Good (70-89%)'
    elif rate >= 50:
        return 'Fair (50-69%)'
    else:
        return 'Needs Improvement (<50%)'

df['performance_category'] = df['completion_rate'].apply(categorize_performance)

# Calculate additional metrics
df['engagement_score'] = df['days_with_responses'] / df['total_days'] * 100
df['start_month'] = df['start_date'].dt.to_period('M')

# Print summary statistics
print("=" * 80)
print("USER ENGAGEMENT ANALYSIS - SUMMARY STATISTICS")
print("=" * 80)
print(f"\nTotal Users: {len(df)}")
print(f"Average Completion Rate: {df['completion_rate'].mean():.2f}%")
print(f"Median Completion Rate: {df['completion_rate'].median():.2f}%")
print(f"Standard Deviation: {df['completion_rate'].std():.2f}%")
print(f"Highest Completion Rate: {df['completion_rate'].max():.2f}% (User {df.loc[df['completion_rate'].idxmax(), 'user_id']})")
print(f"Lowest Completion Rate: {df['completion_rate'].min():.2f}% (User {df.loc[df['completion_rate'].idxmin(), 'user_id']})")
print(f"\nAverage Tracking Duration: {df['total_days'].mean():.1f} days")
print(f"Total Days Tracked (all users): {df['total_days'].sum()} days")
print(f"Total Responses: {df['days_with_responses'].sum()}")
print(f"Total Missing Days: {df['missing_days'].sum()}")

print("\n" + "=" * 80)
print("PERFORMANCE DISTRIBUTION")
print("=" * 80)
print(df['performance_category'].value_counts().sort_index())

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Completion Rate Distribution (Histogram)
ax1 = plt.subplot(3, 3, 1)
plt.hist(df['completion_rate'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(df['completion_rate'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["completion_rate"].mean():.1f}%')
plt.axvline(df['completion_rate'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["completion_rate"].median():.1f}%')
plt.xlabel('Completion Rate (%)', fontsize=10)
plt.ylabel('Number of Users', fontsize=10)
plt.title('Distribution of Completion Rates', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Top 10 Users by Completion Rate
ax2 = plt.subplot(3, 3, 2)
top_10 = df.nlargest(10, 'completion_rate')
colors = ['#10b981' if x >= 95 else '#3b82f6' for x in top_10['completion_rate']]
plt.barh(top_10['user_id'].astype(str), top_10['completion_rate'], color=colors, edgecolor='black')
plt.xlabel('Completion Rate (%)', fontsize=10)
plt.ylabel('User ID', fontsize=10)
plt.title('Top 10 Users by Completion Rate', fontsize=12, fontweight='bold')
plt.xlim(0, 105)
for i, v in enumerate(top_10['completion_rate']):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
plt.grid(True, alpha=0.3, axis='x')

# 3. Bottom 10 Users by Completion Rate
ax3 = plt.subplot(3, 3, 3)
bottom_10 = df.nsmallest(10, 'completion_rate')
colors = ['#ef4444' if x < 50 else '#f97316' for x in bottom_10['completion_rate']]
plt.barh(bottom_10['user_id'].astype(str), bottom_10['completion_rate'], color=colors, edgecolor='black')
plt.xlabel('Completion Rate (%)', fontsize=10)
plt.ylabel('User ID', fontsize=10)
plt.title('Bottom 10 Users by Completion Rate', fontsize=12, fontweight='bold')
plt.xlim(0, 105)
for i, v in enumerate(bottom_10['completion_rate']):
    plt.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
plt.grid(True, alpha=0.3, axis='x')

# 4. Performance Category Distribution (Pie Chart)
ax4 = plt.subplot(3, 3, 4)
performance_counts = df['performance_category'].value_counts()
colors_pie = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
explode = (0.05, 0, 0, 0)
plt.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
        colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 9})
plt.title('Performance Category Distribution', fontsize=12, fontweight='bold')

# 5. Completion Rate vs Total Days (Scatter)
ax5 = plt.subplot(3, 3, 5)
scatter_colors = df['completion_rate'].apply(lambda x: '#10b981' if x >= 90 else '#f59e0b' if x >= 70 else '#f97316' if x >= 50 else '#ef4444')
plt.scatter(df['total_days'], df['completion_rate'], s=100, c=scatter_colors, alpha=0.6, edgecolors='black')
plt.xlabel('Total Days Tracked', fontsize=10)
plt.ylabel('Completion Rate (%)', fontsize=10)
plt.title('Completion Rate vs Tracking Duration', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(df['total_days'], df['completion_rate'], 1)
p = np.poly1d(z)
plt.plot(df['total_days'].sort_values(), p(df['total_days'].sort_values()), 
         "r--", alpha=0.8, linewidth=2, label=f'Trend line')
plt.legend()

# 6. Days with Responses vs Missing Days
ax6 = plt.subplot(3, 3, 6)
df_sorted = df.sort_values('completion_rate', ascending=False).head(20)
x = np.arange(len(df_sorted))
width = 0.35
plt.bar(x - width/2, df_sorted['days_with_responses'], width, label='Days with Responses', color='#10b981', edgecolor='black')
plt.bar(x + width/2, df_sorted['missing_days'], width, label='Missing Days', color='#ef4444', edgecolor='black')
plt.xlabel('User ID', fontsize=10)
plt.ylabel('Number of Days', fontsize=10)
plt.title('Top 20 Users: Response Days vs Missing Days', fontsize=12, fontweight='bold')
plt.xticks(x, df_sorted['user_id'].astype(str), rotation=45, ha='right', fontsize=8)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 7. Timeline: Start Dates vs Completion Rate
ax7 = plt.subplot(3, 3, 7)
scatter_colors = df['completion_rate'].apply(lambda x: '#10b981' if x >= 90 else '#f59e0b' if x >= 70 else '#f97316' if x >= 50 else '#ef4444')
scatter_sizes = df['total_days'] * 2
plt.scatter(df['start_date'], df['completion_rate'], s=scatter_sizes, c=scatter_colors, alpha=0.6, edgecolors='black')
plt.xlabel('Start Date', fontsize=10)
plt.ylabel('Completion Rate (%)', fontsize=10)
plt.title('Engagement Over Time (Size = Duration)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(-5, 105)

# 8. Box Plot by Performance Category
ax8 = plt.subplot(3, 3, 8)
category_order = ['Excellent (≥90%)', 'Good (70-89%)', 'Fair (50-69%)', 'Needs Improvement (<50%)']
df_ordered = df.copy()
df_ordered['performance_category'] = pd.Categorical(df_ordered['performance_category'], categories=category_order, ordered=True)
box_colors = ['#10b981', '#3b82f6', '#f97316', '#ef4444']
bp = plt.boxplot([df_ordered[df_ordered['performance_category'] == cat]['total_days'].values 
                   for cat in category_order], 
                  labels=['Excellent', 'Good', 'Fair', 'Low'], 
                  patch_artist=True)
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
plt.ylabel('Total Days Tracked', fontsize=10)
plt.xlabel('Performance Category', fontsize=10)
plt.title('Tracking Duration by Performance', fontsize=12, fontweight='bold')
plt.xticks(rotation=15, ha='right', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

# 9. Monthly Cohort Analysis
ax9 = plt.subplot(3, 3, 9)
monthly_stats = df.groupby('start_month').agg({
    'completion_rate': 'mean',
    'user_id': 'count'
}).reset_index()
monthly_stats.columns = ['month', 'avg_completion', 'user_count']
monthly_stats['month_str'] = monthly_stats['month'].astype(str)

bars = plt.bar(monthly_stats['month_str'], monthly_stats['avg_completion'], 
               color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Start Month', fontsize=10)
plt.ylabel('Average Completion Rate (%)', fontsize=10)
plt.title('Average Completion Rate by Start Month', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')

# Add user count labels on bars
for i, (bar, count) in enumerate(zip(bars, monthly_stats['user_count'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'n={int(count)}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/user_engagement_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Main analysis saved as 'user_engagement_analysis.png'")

# Create a second figure with additional detailed analysis
fig2 = plt.figure(figsize=(16, 10))

# 1. Heatmap: Completion Rate by User (sorted)
ax1 = plt.subplot(2, 2, 1)
df_sorted = df.sort_values('completion_rate', ascending=False)
completion_matrix = df_sorted['completion_rate'].values.reshape(-1, 1)
im = plt.imshow(completion_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
plt.colorbar(im, label='Completion Rate (%)')
plt.yticks([0], ['Completion Rate'])
plt.xticks(range(0, len(df_sorted), 5), df_sorted['user_id'].values[::5], rotation=90, fontsize=8)
plt.xlabel('User ID (sorted by performance)', fontsize=10)
plt.title('User Completion Rate Heatmap', fontsize=12, fontweight='bold')

# 2. Cumulative Distribution
ax2 = plt.subplot(2, 2, 2)
sorted_rates = np.sort(df['completion_rate'])
cumulative = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates) * 100
plt.plot(sorted_rates, cumulative, linewidth=2, color='steelblue')
plt.axhline(50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
plt.axhline(90, color='green', linestyle='--', alpha=0.5, label='90th percentile')
plt.axvline(df['completion_rate'].median(), color='orange', linestyle='--', alpha=0.5, label=f'Median: {df["completion_rate"].median():.1f}%')
plt.xlabel('Completion Rate (%)', fontsize=10)
plt.ylabel('Cumulative % of Users', fontsize=10)
plt.title('Cumulative Distribution of Completion Rates', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# 3. Engagement Efficiency (Days with Responses per Total Days)
ax3 = plt.subplot(2, 2, 3)
df_sorted = df.sort_values('completion_rate', ascending=True)
plt.barh(range(len(df_sorted)), df_sorted['completion_rate'], 
         color=df_sorted['completion_rate'].apply(
             lambda x: '#10b981' if x >= 90 else '#3b82f6' if x >= 70 else '#f97316' if x >= 50 else '#ef4444'
         ), edgecolor='black', linewidth=0.5)
plt.xlabel('Completion Rate (%)', fontsize=10)
plt.ylabel('Users (sorted by performance)', fontsize=10)
plt.title('All Users Ranked by Completion Rate', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.xlim(0, 105)

# 4. Statistical Summary Table
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['Total Users', f'{len(df)}'],
    ['Mean Completion', f'{df["completion_rate"].mean():.2f}%'],
    ['Median Completion', f'{df["completion_rate"].median():.2f}%'],
    ['Std Deviation', f'{df["completion_rate"].std():.2f}%'],
    ['', ''],
    ['Excellent (≥90%)', f'{(df["completion_rate"] >= 90).sum()} users ({(df["completion_rate"] >= 90).sum()/len(df)*100:.1f}%)'],
    ['Good (70-89%)', f'{((df["completion_rate"] >= 70) & (df["completion_rate"] < 90)).sum()} users ({((df["completion_rate"] >= 70) & (df["completion_rate"] < 90)).sum()/len(df)*100:.1f}%)'],
    ['Fair (50-69%)', f'{((df["completion_rate"] >= 50) & (df["completion_rate"] < 70)).sum()} users ({((df["completion_rate"] >= 50) & (df["completion_rate"] < 70)).sum()/len(df)*100:.1f}%)'],
    ['Low (<50%)', f'{(df["completion_rate"] < 50).sum()} users ({(df["completion_rate"] < 50).sum()/len(df)*100:.1f}%)'],
    ['', ''],
    ['Avg Days Tracked', f'{df["total_days"].mean():.1f} days'],
    ['Total Responses', f'{df["days_with_responses"].sum():,}'],
    ['Total Missing', f'{df["missing_days"].sum():,}'],
    ['', ''],
    ['Best Performer', f'User {df.loc[df["completion_rate"].idxmax(), "user_id"]} ({df["completion_rate"].max():.2f}%)'],
    ['Needs Support', f'User {df.loc[df["completion_rate"].idxmin(), "user_id"]} ({df["completion_rate"].min():.2f}%)'],
]

table = plt.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#667eea')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style alternating rows
for i in range(1, len(summary_data)):
    for j in range(2):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('/home/claude/detailed_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Detailed analysis saved as 'detailed_analysis.png'")

# Show plots
plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nKey Insights:")
print(f"• {(df['completion_rate'] >= 90).sum()} users ({(df['completion_rate'] >= 90).sum()/len(df)*100:.1f}%) achieved excellent performance (≥90%)")
print(f"• {(df['completion_rate'] < 50).sum()} users ({(df['completion_rate'] < 50).sum()/len(df)*100:.1f}%) need additional support (<50%)")
print(f"• Average engagement improves slightly with longer tracking periods")
print(f"• Users starting in {df.groupby('start_month')['completion_rate'].mean().idxmax()} had the highest average completion rate")
print("\n" + "=" * 80)