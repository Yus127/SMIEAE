"""
Analyze all ML input CSVs: completeness, days per user, data volume.
Outputs summary CSVs to the same folder.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─── CSV paths ───────────────────────────────────────────────────────────────
CSV_FILES = {
    #"data_with_exam_features": "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/data_with_exam_features.csv",
    "5min_window_enriched": "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched/ml_ready_5min_window_enriched.csv",
    "10min_window_enriched": "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched/ml_ready_10min_window_enriched.csv",
    "30min_window_enriched": "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched/ml_ready_30min_window_enriched.csv",
    "60min_window_enriched": "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched/ml_ready_60min_window_enriched.csv",
    "combined_windows_enriched": "/Users/YusMolina/Downloads/smieae/data/ml_ready/enriched/ml_ready_combined_windows_enriched.csv",
}

OUTPUT_DIR = "/Users/YusMolina/Downloads/smieae/data/data_clean/csv_joined/analyzed"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def detect_date_column(df):
    """Try to find a date/timestamp column."""
    candidates = ["date", "date_only", "timestamp", "timeStampStart", "Date", "day"]
    for col in candidates:
        if col in df.columns:
            return col
    # fallback: look for columns with 'date' or 'time' in the name
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    return None


def detect_user_column(df):
    """Try to find a user ID column."""
    
    if "userid" in df.columns:
        return "userid"
    for col in df.columns:
        if "user" in col.lower() or "participant" in col.lower():
            return col
    return None


def analyze_csv(name, path):
    """Analyze a single CSV and return summary dicts."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {path}")
    print(f"{'='*70}")

    if not os.path.exists(path):
        print(f"  FILE NOT FOUND — skipping")
        return None, None, None

    df = pd.read_csv(path, low_memory=False)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)

    # ── General overview ──
    print(f"\n  Shape:        {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  File size:    {file_size_mb:.1f} MB")

    # ── Completeness per column ──
    completeness = df.notna().mean() * 100
    overall_completeness = df.notna().mean().mean() * 100
    print(f"\n  Overall completeness: {overall_completeness:.1f}%")
    print(f"  Columns 100% complete: {(completeness == 100).sum()} / {len(completeness)}")
    print(f"  Columns <50% complete: {(completeness < 50).sum()}")

    low_cols = completeness[completeness < 50].sort_values()
    if len(low_cols) > 0:
        print(f"\n  Columns with <50% completeness:")
        for col, pct in low_cols.items():
            print(f"    {col:40s} {pct:5.1f}%")

    completeness_df = pd.DataFrame({
        "column": completeness.index,
        "completeness_pct": completeness.values,
        "non_null_count": df.notna().sum().values,
        "total_rows": df.shape[0],
        "dtype": [str(df[c].dtype) for c in df.columns],
    }).sort_values("completeness_pct")
    completeness_df.insert(0, "dataset", name)

    # ── User & date detection ──
    user_col = detect_user_column(df)
    date_col = detect_date_column(df)

    print(f"\n  User column:  {user_col}")
    print(f"  Date column:  {date_col}")

    user_summary_df = None
    if user_col:
        n_users = df[user_col].nunique()
        print(f"  Unique users: {n_users}")

        if date_col:
            date_series = pd.to_datetime(df[date_col], errors="coerce")
            df["_date_parsed"] = date_series.dt.date

            user_stats = df.groupby(user_col).agg(
                n_rows=("_date_parsed", "size"),
                n_days=("_date_parsed", "nunique"),
                first_date=("_date_parsed", "min"),
                last_date=("_date_parsed", "max"),
            ).reset_index()
            user_stats["avg_rows_per_day"] = (user_stats["n_rows"] / user_stats["n_days"]).round(1)
            user_stats.insert(0, "dataset", name)

            print(f"\n  Days per user:")
            print(f"    Mean:   {user_stats['n_days'].mean():.1f}")
            print(f"    Median: {user_stats['n_days'].median():.0f}")
            print(f"    Min:    {user_stats['n_days'].min()}")
            print(f"    Max:    {user_stats['n_days'].max()}")
            print(f"\n  Date range: {user_stats['first_date'].min()} → {user_stats['last_date'].max()}")

            user_summary_df = user_stats
            df.drop(columns=["_date_parsed"], inplace=True)
        else:
            rows_per_user = df.groupby(user_col).size().reset_index(name="n_rows")
            rows_per_user.insert(0, "dataset", name)
            print(f"\n  Rows per user:")
            print(f"    Mean:   {rows_per_user['n_rows'].mean():.1f}")
            print(f"    Median: {rows_per_user['n_rows'].median():.0f}")
            print(f"    Min:    {rows_per_user['n_rows'].min()}")
            print(f"    Max:    {rows_per_user['n_rows'].max()}")
            user_summary_df = rows_per_user

    # ── Overview row ──
    overview = {
        "dataset": name,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "overall_completeness_pct": round(overall_completeness, 1),
        "cols_100pct_complete": int((completeness == 100).sum()),
        "n_users": df[user_col].nunique() if user_col else None,
        "user_col": user_col,
        "date_col": date_col,
    }

    return overview, completeness_df, user_summary_df


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"CSV Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Analyzing {len(CSV_FILES)} files...\n")

    all_overviews = []
    all_completeness = []
    all_users = []

    for name, path in CSV_FILES.items():
        overview, comp_df, user_df = analyze_csv(name, path)
        if overview:
            all_overviews.append(overview)
        if comp_df is not None:
            all_completeness.append(comp_df)
        if user_df is not None:
            all_users.append(user_df)

    # ── Save summaries ──
    overview_df = pd.DataFrame(all_overviews)
    overview_df.to_csv(os.path.join(OUTPUT_DIR, "analysis_overview.csv"), index=False)

    if all_completeness:
        pd.concat(all_completeness, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "analysis_completeness.csv"), index=False
        )

    if all_users:
        pd.concat(all_users, ignore_index=True).to_csv(
            os.path.join(OUTPUT_DIR, "analysis_users.csv"), index=False
        )

    print(f"\n{'='*70}")
    print("  COMPARISON ACROSS ALL DATASETS")
    print(f"{'='*70}")
    print(overview_df.to_string(index=False))

    print(f"\n\nOutput saved to:")
    print(f"  - {OUTPUT_DIR}analysis_overview.csv")
    print(f"  - {OUTPUT_DIR}analysis_completeness.csv")
    print(f"  - {OUTPUT_DIR}analysis_users.csv")
