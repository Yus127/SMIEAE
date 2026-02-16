"""
Filter ESFM sensor records to only those that match Class B schedule and
produce exploratory stats for each CSV.

Your schema (confirmed):
timestamp, Temperature_C, Pressure_hPa, CO2_PPM, IN_Count, OUT_Count,
source_file, sensor_category, date

Class B schedule:
- Date range: 2025-02-10 to 2025-07-11 (inclusive)
- Days: Monday, Tuesday, Thursday
- Time: 09:30 to 11:00 (local time)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd


# CONFIG

BASE_PATH = "/Users/YusMolina/Downloads/smieae/data/data_clean/ESFM/"
CONTRARIO_PATH = Path(BASE_PATH+"combined_ESFM_Contrario.csv")
PUERTA_PATH    = Path(BASE_PATH+"combined_ESFM_Puerta.csv")

OUTPUT_CONTRARIO = Path(BASE_PATH+"ESFM_Contrario_B_filtered.csv")
OUTPUT_PUERTA    = Path(BASE_PATH+"ESFM_Puerta_B_filtered.csv")

# Class B schedule
CLASS_B_START_DATE = pd.Timestamp("2025-02-10").date()
CLASS_B_END_DATE   = pd.Timestamp("2025-07-11").date()

# Monday=0 ... Sunday=6
CLASS_B_WEEKDAYS = {0, 1, 3}  # Mon, Tue, Thu

CLASS_B_START_TIME = pd.Timestamp("09:30").time()
CLASS_B_END_TIME   = pd.Timestamp("11:00").time()

# Use [start, end) by default (end exclusive avoids overlaps)
END_TIME_EXCLUSIVE = False

# Known column names
TS_COL   = "timestamp"
TEMP_COL = "Temperature_C"
PRES_COL = "Pressure_hPa"
CO2_COL  = "CO2_PPM"
IN_COL   = "IN_Count"
OUT_COL  = "OUT_Count"


# HELPERS

def load_sensor_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    df = pd.read_csv(path)

    required = {TS_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    # Parse timestamp
    df["ts"] = pd.to_datetime(df[TS_COL], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["ts"]).copy()
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def p05_median_p95_max(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {"p05": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "p05": float(np.percentile(x, 5)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def median_spacing_seconds(ts: pd.Series) -> float:
    ts = ts.dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    diffs = ts.diff().dropna().dt.total_seconds()
    return float(diffs.median())


def summarize_dataset(df: pd.DataFrame, label: str) -> dict:
    # Span and unique dates
    start = df["ts"].min()
    end = df["ts"].max()
    unique_dates = int(df["ts"].dt.date.nunique())

    # Environmental presence (any of temp/pressure/co2 non-null)
    env_cols = [c for c in [TEMP_COL, PRES_COL, CO2_COL] if c in df.columns]
    env_mask = df[env_cols].notna().any(axis=1) if env_cols else pd.Series(False, index=df.index)

    # Count presence (any of in/out non-null)
    cnt_cols = [c for c in [IN_COL, OUT_COL] if c in df.columns]
    cnt_mask = df[cnt_cols].notna().any(axis=1) if cnt_cols else pd.Series(False, index=df.index)

    out = {
        "label": label,
        "rows": int(len(df)),
        "span_start": str(start),
        "span_end": str(end),
        "unique_dates": unique_dates,
        "env_rows_any": int(env_mask.sum()),
        "count_rows_any": int(cnt_mask.sum()),
        "median_env_spacing_s": median_spacing_seconds(df.loc[env_mask, "ts"]),
        "median_count_spacing_s": median_spacing_seconds(df.loc[cnt_mask, "ts"]),
    }

    # Core stats
    if TEMP_COL in df.columns:
        out["temperature_stats"] = p05_median_p95_max(df[TEMP_COL])
    if PRES_COL in df.columns:
        out["pressure_stats"] = p05_median_p95_max(df[PRES_COL])
    if CO2_COL in df.columns:
        out["co2_stats"] = p05_median_p95_max(df[CO2_COL])

    return out


def print_summary(s: dict) -> None:
    print(f"\n=== {s['label']} ===")
    print(f"Rows: {s['rows']}")
    print(f"Span: {s['span_start']} -> {s['span_end']}")
    print(f"Unique dates: {s['unique_dates']}")
    print(f"Environmental rows (any of Temp/Pressure/CO2 present): {s['env_rows_any']}")
    print(f"Count rows (any of IN/OUT present): {s['count_rows_any']}")
    print(f"Median spacing (env): {s['median_env_spacing_s']:.2f} s")
    print(f"Median spacing (counts): {s['median_count_spacing_s']:.2f} s")

    if "temperature_stats" in s:
        t = s["temperature_stats"]
        print(f"Temperature_C p05/median/p95/max: {t['p05']:.2f} / {t['median']:.2f} / {t['p95']:.2f} / {t['max']:.2f}")
    if "pressure_stats" in s:
        p = s["pressure_stats"]
        print(f"Pressure_hPa  p05/median/p95/max: {p['p05']:.2f} / {p['median']:.2f} / {p['p95']:.2f} / {p['max']:.2f}")
    if "co2_stats" in s:
        c = s["co2_stats"]
        print(f"CO2_PPM       p05/median/p95/max: {c['p05']:.2f} / {c['median']:.2f} / {c['p95']:.2f} / {c['max']:.2f}")


def filter_class_b(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["ts"]

    # Date range (inclusive)
    date_ok = (ts.dt.date >= CLASS_B_START_DATE) & (ts.dt.date <= CLASS_B_END_DATE)

    # Weekday: Mon/Tue/Thu
    weekday_ok = ts.dt.weekday.isin(CLASS_B_WEEKDAYS)

    # Time window
    t = ts.dt.time
    if END_TIME_EXCLUSIVE:
        time_ok = (t >= CLASS_B_START_TIME) & (t < CLASS_B_END_TIME)
    else:
        time_ok = (t >= CLASS_B_START_TIME) & (t <= CLASS_B_END_TIME)

    keep = date_ok & weekday_ok & time_ok
    return df.loc[keep].copy()


def overlap_correlations(df_contra: pd.DataFrame, df_puerta: pd.DataFrame, tolerance_seconds: int = 120) -> None:
    """
    Align Contrario to Puerta by nearest timestamp (within tolerance) and compute:
      - pressure correlation
      - CO2 correlation
      - mean CO2 difference (Contrario - Puerta)
    """
    left = df_contra[["ts", PRES_COL, CO2_COL]].dropna(subset=["ts"]).sort_values("ts")
    right = df_puerta[["ts", PRES_COL, CO2_COL]].dropna(subset=["ts"]).sort_values("ts")

    if left.empty or right.empty:
        print("\n[Overlap] Not enough data for overlap correlations.")
        return

    m = pd.merge_asof(
        left, right, on="ts", direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("_contra", "_puerta")
    ).dropna()

    if m.empty:
        print(f"\n[Overlap] No matches within {tolerance_seconds}s tolerance.")
        return

    pres_corr = pd.to_numeric(m[f"{PRES_COL}_contra"], errors="coerce").corr(
        pd.to_numeric(m[f"{PRES_COL}_puerta"], errors="coerce")
    )
    co2_corr = pd.to_numeric(m[f"{CO2_COL}_contra"], errors="coerce").corr(
        pd.to_numeric(m[f"{CO2_COL}_puerta"], errors="coerce")
    )
    co2_diff = pd.to_numeric(m[f"{CO2_COL}_contra"], errors="coerce") - pd.to_numeric(m[f"{CO2_COL}_puerta"], errors="coerce")

    print(f"\n[Overlap] Pressure correlation (nearest within {tolerance_seconds}s): {float(pres_corr):.4f}")
    print(f"[Overlap] CO2 correlation (nearest within {tolerance_seconds}s):      {float(co2_corr):.4f}")
    print(f"[Overlap] Mean CO2 difference (Contrario - Puerta):                 {float(co2_diff.mean()):.2f} ppm")





def main() -> None:
    # Load
    df_contra = load_sensor_csv(CONTRARIO_PATH)
    df_puerta = load_sensor_csv(PUERTA_PATH)

    # Exploratory summaries (raw)
    s1 = summarize_dataset(df_contra, "ESFM_Contrario (raw)")
    s2 = summarize_dataset(df_puerta, "ESFM_Puerta (raw)")
    print_summary(s1)
    print_summary(s2)

    # Optional overlap diagnostics (reproduces correlation-style checks)
    overlap_correlations(df_contra, df_puerta, tolerance_seconds=120)

    # Filter to Class B schedule
    df_contra_B = filter_class_b(df_contra)
    df_puerta_B = filter_class_b(df_puerta)

    print("\n=== Class B Filter Results ===")
    print(f"Contrario: {len(df_contra)} -> {len(df_contra_B)} rows kept")
    print(f"Puerta:    {len(df_puerta)} -> {len(df_puerta_B)} rows kept")

    # Save filtered
    df_contra_B.to_csv(OUTPUT_CONTRARIO, index=False)
    df_puerta_B.to_csv(OUTPUT_PUERTA, index=False)

    print("\nSaved filtered CSVs:")
    print(f" - {OUTPUT_CONTRARIO.resolve()}")
    print(f" - {OUTPUT_PUERTA.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
