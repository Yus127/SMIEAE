#!/usr/bin/env python3
"""
03_course_averages.py

Purpose:
- Load the CLEANED daily_questions dataset
- Merge userid -> university/course mapping (embedded)
- Group stress/anxiety by class (course) and plot averages:
    1) Overall mean stress/anxiety per course (bar chart)
    2) Daily mean stress per course (time series)
    3) Daily mean anxiety per course (time series)
- Export CSV summaries

Notes:
- Uses the adjusted day logic from cleaning: date_only is derived from timestamp_for_day.
- Does NOT compute per-user daily means (assumes 1 value/day/user).
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- CONFIG (edit here) ---
CLEANED_CSV = r"/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"
USERS_COURSES_CSV = r"/Users/YusMolina/Downloads/smieae/data/original_data/users-courses.csv"
OUTPUT_DIR = r"/Users/YusMolina/Downloads/smieae/data/data_clean/stress_anxiety_by_course"
EXPECTED_COURSES = ["A1", "A2", "B", "C"]
# --------------------------



def pick_first_matching(columns: list[str], patterns: list[str]) -> str | None:
    for c in columns:
        for p in patterns:
            if re.search(p, c, flags=re.IGNORECASE):
                return c
    return None


def main() -> int:
    in_path = Path(CLEANED_CSV)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Basic required columns
    if "userid" not in df.columns:
        raise SystemExit("Missing 'userid' column in cleaned dataset.")
    if "timestamp_for_day" not in df.columns:
        raise SystemExit("Missing 'timestamp_for_day' column in cleaned dataset.")

    # Parse timestamps
    df["timestamp_for_day"] = pd.to_datetime(df["timestamp_for_day"], errors="coerce")
    df["date_only"] = df["timestamp_for_day"].dt.date

    # Detect stress/anxiety columns
    stress_col = pick_first_matching(list(df.columns), [r"stress", r"estr[eé]s"])
    anxiety_col = pick_first_matching(list(df.columns), [r"anxiety", r"ansiedad"])
    if not stress_col or not anxiety_col:
        raise SystemExit(
            "Could not auto-detect stress/anxiety columns.\n"
            f"Detected stress_col={stress_col}, anxiety_col={anxiety_col}\n"
            f"Columns:\n{chr(10).join(df.columns)}"
        )
    
    # Load mapping CSV
    df_map = pd.read_csv(USERS_COURSES_CSV)
    required_map_cols = {"userid", "university", "course"}
    if not required_map_cols.issubset(df_map.columns):
        raise SystemExit(
            f"Mapping CSV must contain columns: {sorted(required_map_cols)}\n"
            f"Found: {list(df_map.columns)}"
        )

    # Normalize userid types
    df["userid"] = pd.to_numeric(df["userid"], errors="coerce").astype("Int64")
    df_map["userid"] = pd.to_numeric(df_map["userid"], errors="coerce").astype("Int64")


    # Merge course labels
    df = df.merge(df_map[["userid", "university", "course"]], on="userid", how="left")
    df["course"] = df["course"].fillna("Unknown")


    # Keep valid rows
    d = df.dropna(subset=["date_only", stress_col, anxiety_col]).copy()
    d[stress_col] = pd.to_numeric(d[stress_col], errors="coerce")
    d[anxiety_col] = pd.to_numeric(d[anxiety_col], errors="coerce")
    d = d.dropna(subset=[stress_col, anxiety_col])

    # -------------------------
    # Aggregations
    # -------------------------

    # Overall mean per course
    course_overall = (
        d.groupby("course", as_index=False)
         .agg(
             stress_mean=(stress_col, "mean"),
             anxiety_mean=(anxiety_col, "mean"),
             n_records=(stress_col, "size"),
             n_users=("userid", "nunique"),
         )
         .sort_values("course")
    )
    course_overall.to_csv(out_dir / "course_overall_means.csv", index=False)

    # Daily mean per course
    course_daily = (
        d.groupby(["course", "date_only"], as_index=False)
         .agg(
             stress_mean=(stress_col, "mean"),
             anxiety_mean=(anxiety_col, "mean"),
             n_records=(stress_col, "size"),
             n_users=("userid", "nunique"),
         )
         .sort_values(["course", "date_only"])
    )
    course_daily.to_csv(out_dir / "course_daily_means.csv", index=False)
    course_daily = course_daily.sort_values(["course", "date_only"])


    # -------------------------
    # Plots
    # -------------------------

    # 1) Bar chart: overall averages by course
    courses = course_overall["course"].astype(str).tolist()
    x = np.arange(len(courses))
    width = 0.35

    fig = plt.figure()
    plt.bar(x - width / 2, course_overall["stress_mean"], width, label="Stress mean")
    plt.bar(x + width / 2, course_overall["anxiety_mean"], width, label="Anxiety mean")
    plt.xticks(x, courses)
    plt.xlabel("Course (Class)")
    plt.ylabel("Mean (0–100)")
    plt.title("Average Stress and Anxiety by Class")
    plt.legend()
    p1 = out_dir / "01_course_overall_means_bar.png"
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) One chart per course: daily mean stress + anxiety (two lines)

    for course in EXPECTED_COURSES:
        g = course_daily[course_daily["course"] == course].copy()
        if g.empty:
            print(f"[WARN] No data found for course={course}. Skipping plot.")
            continue

        fig = plt.figure()
        x = pd.to_datetime(g["date_only"])
        plt.plot(x, g["stress_mean"], label="Stress (daily mean)")
        plt.plot(x, g["anxiety_mean"], label="Anxiety (daily mean)")
        plt.xlabel("Date")
        plt.ylabel("Daily mean (0–100)")
        plt.title(f"Daily Mean Stress & Anxiety — Class {course}")
        plt.legend()
        fig.autofmt_xdate()

        p = out_dir / f"COURSE_{course}_daily_stress_anxiety.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Wrote: {p}")


    print("[DONE] Outputs written to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
