"""
02_eda_stress_anxiety.py

Purpose:
- Load the cleaned daily_questions CSV from Script 1
- Run exploratory analysis (EDA):
    - Overall describe (mean/std/min/max/quantiles)
    - Daily stats (by adjusted date_only derived from timestamp_for_day)
    - Weekday stats
- Generate charts:
    - Overall raw over time
    - Overall daily means
    - Histograms for stress/anxiety
    - Weekday mean plot
    - Scatter stress vs anxiety

Assumptions:
- The cleaned CSV includes:
    - userid
    - timestamp (datetime string in CSV; parsed here)
    - timestamp_for_day (datetime string in CSV; parsed here)
    - stress + anxiety columns
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- CONFIG (edit here) ---
CLEANED_CSV = r"/Users/YusMolina/Downloads/smieae/data/data_clean/daily_questions_cleaned.csv"
OUTPUT_DIR = r"/Users/YusMolina/Downloads/smieae/data/data_clean/stress_anxiety_EDA"
# --------------------------


def pick_first_matching(columns: list[str], patterns: list[str]) -> str | None:
    for c in columns:
        for p in patterns:
            if re.search(p, c, flags=re.IGNORECASE):
                return c
    return None


def agg_stats(frame: pd.DataFrame, value_col: str) -> dict:
    s = pd.to_numeric(frame[value_col], errors="coerce")
    return {
        f"{value_col}_count": int(s.count()),
        f"{value_col}_mean": float(s.mean()) if s.count() else np.nan,
        f"{value_col}_std": float(s.std(ddof=1)) if s.count() > 1 else np.nan,
        f"{value_col}_min": float(s.min()) if s.count() else np.nan,
        f"{value_col}_q25": float(s.quantile(0.25)) if s.count() else np.nan,
        f"{value_col}_median": float(s.median()) if s.count() else np.nan,
        f"{value_col}_q75": float(s.quantile(0.75)) if s.count() else np.nan,
        f"{value_col}_max": float(s.max()) if s.count() else np.nan,
    }


def main() -> int:
    in_path = Path(CLEANED_CSV)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Parse timestamps
    if "timestamp" not in df.columns:
        raise SystemExit("Missing 'timestamp' column in cleaned CSV.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "timestamp_for_day" not in df.columns:
        raise SystemExit(
            "Missing 'timestamp_for_day' in cleaned CSV. "
            "Keep it in the cleaning step to group by adjusted day."
        )
    df["timestamp_for_day"] = pd.to_datetime(df["timestamp_for_day"], errors="coerce")

    # Use adjusted day for grouping
    df["date_only"] = df["timestamp_for_day"].dt.date
    df["weekday"] = df["timestamp_for_day"].dt.day_name()

    if "userid" not in df.columns:
        raise SystemExit("Missing 'userid' column in cleaned CSV.")

    # Detect stress/anxiety columns
    stress_col = pick_first_matching(list(df.columns), [r"stress", r"estr[eé]s"])
    anxiety_col = pick_first_matching(list(df.columns), [r"anxiety", r"ansiedad"])
    if not stress_col or not anxiety_col:
        raise SystemExit(
            "Could not auto-detect stress/anxiety columns in cleaned CSV.\n"
            f"Detected stress_col={stress_col}, anxiety_col={anxiety_col}\n"
            f"Columns:\n{chr(10).join(df.columns)}"
        )

    # Valid rows
    d = df.dropna(subset=["timestamp", "date_only", stress_col, anxiety_col]).copy()

    # -------------------------
    # EDA tables
    # -------------------------
    overall_desc = (
        d[[stress_col, anxiety_col]]
        .apply(pd.to_numeric, errors="coerce")
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T
    )
    overall_desc["missing"] = d[[stress_col, anxiety_col]].isna().sum().values
    overall_desc.to_csv(out_dir / "eda_overall_describe.csv")
    print(f"[OK] Wrote: {out_dir / 'eda_overall_describe.csv'}")

    # Daily stats (across all users)
    daily_rows = []
    for date_val, g in d.groupby("date_only"):
        row = {"date_only": date_val, "n_records": len(g), "n_users": int(g["userid"].nunique())}
        row.update(agg_stats(g, stress_col))
        row.update(agg_stats(g, anxiety_col))
        daily_rows.append(row)

    daily_stats = pd.DataFrame(daily_rows).sort_values("date_only")
    daily_stats.to_csv(out_dir / "eda_daily_stats.csv", index=False)
    print(f"[OK] Wrote: {out_dir / 'eda_daily_stats.csv'}")

    # Weekday stats
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_rows = []
    for wd, g in d.groupby("weekday"):
        row = {"weekday": wd, "n_records": len(g), "n_users": int(g["userid"].nunique())}
        row.update(agg_stats(g, stress_col))
        row.update(agg_stats(g, anxiety_col))
        weekday_rows.append(row)

    weekday_stats = pd.DataFrame(weekday_rows)
    weekday_stats["weekday"] = pd.Categorical(weekday_stats["weekday"], categories=weekday_order, ordered=True)
    weekday_stats = weekday_stats.sort_values("weekday")
    weekday_stats.to_csv(out_dir / "eda_weekday_stats.csv", index=False)
    print(f"[OK] Wrote: {out_dir / 'eda_weekday_stats.csv'}")

    # -------------------------
    # Plots
    # -------------------------

  

    # 1) Overall daily means (across all users)
    overall_daily = (
        d.groupby("date_only", as_index=False)
        .agg(
            stress_mean=(stress_col, "mean"),
            anxiety_mean=(anxiety_col, "mean"),
            n_records=(stress_col, "size"),
            n_users=("userid", "nunique"),
        )
        .sort_values("date_only")
    )

    fig = plt.figure()
    x = pd.to_datetime(overall_daily["date_only"])
    plt.plot(x, overall_daily["stress_mean"], label="Stress (daily mean)")
    plt.plot(x, overall_daily["anxiety_mean"], label="Anxiety (daily mean)")
    plt.xlabel("Date")
    plt.ylabel("Daily mean (0–100)")
    plt.title("Daily Mean Stress and Anxiety (All Users)")
    plt.legend()
    fig.autofmt_xdate()
    p2 = out_dir / "01_overall_daily_means.png"
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Histograms
    fig = plt.figure()
    plt.hist(pd.to_numeric(d[stress_col], errors="coerce").dropna(), bins=30)
    plt.xlabel("Stress (0–100)")
    plt.ylabel("Count")
    plt.title("Distribution of Stress")
    p3 = out_dir / "02_hist_stress.png"
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.hist(pd.to_numeric(d[anxiety_col], errors="coerce").dropna(), bins=30)
    plt.xlabel("Anxiety (0–100)")
    plt.ylabel("Count")
    plt.title("Distribution of Anxiety")
    p4 = out_dir / "03_hist_anxiety.png"
    plt.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Weekday means plot
    fig = plt.figure()
    plt.plot(weekday_stats["weekday"].astype(str), weekday_stats[f"{stress_col}_mean"], label="Stress mean")
    plt.plot(weekday_stats["weekday"].astype(str), weekday_stats[f"{anxiety_col}_mean"], label="Anxiety mean")
    plt.xlabel("Weekday")
    plt.ylabel("Mean (0–100)")
    plt.title("Mean Stress/Anxiety by Weekday")
    plt.legend()
    p5 = out_dir / "04_weekday_means.png"
    plt.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Scatter + correlation
    s_vals = pd.to_numeric(d[stress_col], errors="coerce")
    a_vals = pd.to_numeric(d[anxiety_col], errors="coerce")
    corr = float(np.corrcoef(s_vals.fillna(0), a_vals.fillna(0))[0, 1]) if len(d) else np.nan

    fig = plt.figure()
    plt.scatter(d[stress_col], d[anxiety_col], s=10)
    plt.xlabel("Stress (0–100)")
    plt.ylabel("Anxiety (0–100)")
    plt.title(f"Stress vs Anxiety (All Records, Pearson r = {corr:.3f})")
    p6 = out_dir / "06_scatter_stress_vs_anxiety.png"
    plt.savefig(p6, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] EDA tables and plots written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
