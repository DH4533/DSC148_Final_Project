"""
data_loader.py
--------------
Fetches Statcast pitch-level data from Baseball Savant via pybaseball,
caches it locally, and provides a clean DataFrame ready for feature engineering.

Usage:
    python src/data_loader.py --start 2022-04-01 --end 2023-10-01
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pybaseball import statcast
from pybaseball import cache

# Enable pybaseball caching
cache.enable()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Statcast columns we actually need (keeps memory footprint small)
KEEP_COLS = [
    # Identifiers
    "game_date", "pitcher", "batter", "game_pk", "at_bat_number", "pitch_number",

    # Pitch physics
    "release_speed",       # velocity
    "release_spin_rate",   # spin rate
    "release_extension",   # extension (ft from rubber)
    "release_pos_x",       # horizontal release point
    "release_pos_z",       # vertical release point
    "pfx_x",               # horizontal movement
    "pfx_z",               # vertical movement (induced)

    # Location
    "plate_x",             # horizontal location at plate
    "plate_z",             # vertical location at plate
    "sz_top",              # top of batter's strike zone
    "sz_bot",              # bottom of batter's strike zone
    "zone",                # Statcast zone (1-14)

    # Pitch identity
    "pitch_type",          # FF, SL, CH, CU, SI, FC, etc.
    "pitch_name",

    # Count
    "balls", "strikes",

    # Handedness
    "stand",               # batter handedness (L/R)
    "p_throws",            # pitcher handedness (L/R)

    # Outcome (our label)
    "description",         # swinging_strike, called_strike, ball, hit_into_play, etc.
    "type",                # S / B / X
    "events",              # hit outcome if any
]


def fetch_statcast_range(start: str, end: str, chunk_days: int = 7) -> pd.DataFrame:
    """
    Pull Statcast data in weekly chunks to avoid timeouts.

    Parameters
    ----------
    start : str   e.g. "2022-04-01"
    end   : str   e.g. "2023-10-01"
    chunk_days : int   how many days per API request

    Returns
    -------
    pd.DataFrame of all pitches in the date range
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    chunks = []
    cursor = start_dt

    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_dt)
        fname = RAW_DIR / f"statcast_{cursor.date()}_{chunk_end.date()}.parquet"

        if fname.exists():
            print(f"  [cache] {fname.name}")
            chunks.append(pd.read_parquet(fname))
        else:
            print(f"  [fetch] {cursor.date()} → {chunk_end.date()} …", end=" ", flush=True)
            try:
                df = statcast(
                    start_dt=cursor.strftime("%Y-%m-%d"),
                    end_dt=chunk_end.strftime("%Y-%m-%d"),
                )
                # Keep only needed columns (ignore missing ones gracefully)
                available = [c for c in KEEP_COLS if c in df.columns]
                df = df[available].copy()
                df.to_parquet(fname, index=False)
                chunks.append(df)
                print(f"✓ {len(df):,} rows")
                time.sleep(2)  # be polite to the API
            except Exception as exc:
                print(f"✗ ERROR: {exc}")

        cursor = chunk_end + timedelta(days=1)

    if not chunks:
        raise RuntimeError("No data fetched. Check your date range and internet connection.")

    return pd.concat(chunks, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
    - Drop rows missing critical physics or location columns
    - Create binary label: swinging_strike = 1, else 0
    - Normalize pitch types to major families
    """
    critical = ["release_speed", "release_spin_rate", "plate_x", "plate_z",
                "balls", "strikes", "stand", "p_throws", "description"]
    df = df.dropna(subset=[c for c in critical if c in df.columns]).copy()

    # ── Label ───────────────────────────────────────────────────────────────
    swinging_descriptions = {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul_tip",         # controversial; some analysts include
    }
    df["swinging_strike"] = df["description"].isin(swinging_descriptions).astype(int)

    # ── Pitch type consolidation ─────────────────────────────────────────────
    pitch_map = {
        "FF": "FF",  "FA": "FF",
        "SI": "SI",  "FT": "SI",
        "FC": "FC",
        "SL": "SL",  "ST": "SL",
        "CU": "CU",  "KC": "CU",
        "CH": "CH",  "FS": "CH",  "FO": "CH",
        "EP": "CH",
        "CS": "CU",
        "KN": "KN",
    }
    df["pitch_family"] = df["pitch_type"].map(pitch_map).fillna("OT")

    # ── Sort for sequential pitch features ───────────────────────────────────
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    return df


def save_processed(df: pd.DataFrame, name: str = "pitches_clean.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df):,} rows → {out}")
    return out


def load_processed(name: str = "pitches_clean.parquet") -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/data_loader.py` first."
        )
    return pd.read_parquet(path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-04-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default="2023-10-01", help="End date   YYYY-MM-DD")
    args = parser.parse_args()

    print(f"\nFetching Statcast data: {args.start} → {args.end}")
    raw = fetch_statcast_range(args.start, args.end)
    print(f"\nRaw: {len(raw):,} rows, {raw.shape[1]} cols")

    clean_df = clean(raw)
    print(f"Clean: {len(clean_df):,} rows")
    print(f"Swinging strike rate: {clean_df['swinging_strike'].mean():.2%}")

    save_processed(clean_df)
