#!/usr/bin/env python3
"""
01_clean_daily_questions.py

Purpose:
- Load SMIEAE daily_questions export (semicolon-delimited)
- Use timeStampStart as the canonical timestamp (NO timeZoneOffset adjustment)
- Add:
    - timestamp (datetime)
    - timestamp_fmt (YYYY/MM/DD HH:MM:SS)
    - date_only (date without time), with a rule:
        If timestamp time is before 02:00, assign it to the previous day.
    - weekday, hour
- Drop columns you will not use
- Export a cleaned CSV for downstream EDA
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# --- CONFIG (edit here) ---
INPUT_CSV = r"/Users/YusMolina/Downloads/smieae/data/original_data/daily_questions 2025-12-03.csv"
OUTPUT_DIR = r"/Users/YusMolina/Downloads/smieae/data/data_clean"
DELIMITER = ";"
TIMESTAMP_COL = "timeStampStart"  # canonical timestamp for analysis
CLEANED_FILENAME = "daily_questions_cleaned.csv"

DAY_CUTOFF_HOUR = 2  # before this hour -> previous day
# --------------------------


def pick_first_matching(columns: list[str], patterns: list[str]) -> str | None:
    for c in columns:
        for p in patterns:
            if re.search(p, c, flags=re.IGNORECASE):
                return c
    return None


def to_timestamp_no_offset(df: pd.DataFrame, ts_col: str) -> pd.Series:
    dt_utc = pd.to_datetime(df[ts_col], unit="s", utc=True, errors="coerce")
    return dt_utc.dt.tz_convert(None)  # naive datetime, no offset applied


def main() -> int:
    in_path = Path(INPUT_CSV)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, sep=DELIMITER)

    # Basic validation
    required = ["userid", TIMESTAMP_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{chr(10).join(df.columns)}"
        )

    # Canonical timestamp based on timeStampStart (NO timeZoneOffset considered)
    df["timestamp"] = to_timestamp_no_offset(df, TIMESTAMP_COL)
    df["timestamp_fmt"] = df["timestamp"].dt.strftime("%Y/%m/%d %H:%M:%S")
    df["hour"] = df["timestamp"].dt.hour

    # --- 2AM RULE ---
    # If the timestamp is before 02:00, treat it as belonging to the previous day.
    # We implement this by creating a "timestamp_for_day" and taking its date.
    df["timestamp_for_day"] = df["timestamp"] - pd.to_timedelta(
        (df["hour"] < DAY_CUTOFF_HOUR).astype(int), unit="D"
    )
    df["date_only"] = df["timestamp_for_day"].dt.date
    df["weekday"] = df["timestamp_for_day"].dt.day_name()
    # -----------------

    # (Optional) Ensure stress/anxiety columns exist; keep them for EDA
    stress_col = pick_first_matching(list(df.columns), [r"stress", r"estr[eé]s"])
    anxiety_col = pick_first_matching(list(df.columns), [r"anxiety", r"ansiedad"])
    if not stress_col or not anxiety_col:
        print(
            "[WARN] Could not auto-detect stress/anxiety columns.\n"
            f"Detected stress_col={stress_col}, anxiety_col={anxiety_col}\n"
            "Cleaning will continue, but EDA may fail until you confirm column names."
        )

    # Columns to drop (safe: only drop if present)
    drop_cols = [
        "deltaUTC",
        "timeStampScheduled",
        "timeStampSent",
        "timeStampStop",
        "originalTimeStampSent",
        "timeZoneOffset",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # If you don't want to keep timestamp_for_day, drop it here:
    # df = df.drop(columns=["timestamp_for_day"])

    # Sort and export
    df = df.sort_values(["userid", "timestamp"])
    out_path = out_dir / CLEANED_FILENAME
    df.to_csv(out_path, index=False)
    print(f"[OK] Cleaned CSV written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
