"""
feature_engineering.py
-----------------------
Transforms cleaned Statcast pitch data into a model-ready feature matrix.

Features built here:
  1. Velocity difference from pitcher's rolling season average
  2. Statcast zone → 10 binary zone indicators
  3. Pitch location normalized by strike zone
  4. Pitch tunneling distance (3D gap at ~23ft from plate)
  5. Previous-pitch information (type, result, velocity)
  6. Count leverage index
  7. Movement features
  8. Batter/pitcher handedness interaction

Usage:
    from src.feature_engineering import build_features
    X, y, feature_names = build_features(df)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── 1. Count leverage ────────────────────────────────────────────────────────
# Higher value → later in count, more leverage
COUNT_LEVERAGE = {
    (0, 0): 0.0, (1, 0): 0.1, (2, 0): 0.15, (3, 0): 0.18,
    (0, 1): 0.2, (1, 1): 0.3, (2, 1): 0.35, (3, 1): 0.40,
    (0, 2): 0.7, (1, 2): 0.8, (2, 2): 0.9,  (3, 2): 1.0,
}


def _count_leverage(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    keys = list(zip(balls.clip(0, 3), strikes.clip(0, 2)))
    return pd.Series([COUNT_LEVERAGE.get(k, 0.5) for k in keys], index=balls.index)


# ── 2. Pitcher velocity averages ─────────────────────────────────────────────
def _pitcher_pitch_avg_velocity(df: pd.DataFrame) -> pd.Series:
    """
    For each row, compute the pitcher's average velocity for that pitch type
    from all OTHER pitches in the dataset (leave-one-out approximation via
    groupby transform mean, which is close enough at scale).
    """
    return df.groupby(["pitcher", "pitch_family"])["release_speed"].transform("mean")


# ── 3. Zone indicators ───────────────────────────────────────────────────────
# Statcast zones 1-9 = strike zone, 11-14 = outer corners/balls
def _zone_dummies(df: pd.DataFrame) -> pd.DataFrame:
    zone = df["zone"].fillna(0).astype(int)
    dummies = pd.get_dummies(zone, prefix="zone").reindex(
        columns=[f"zone_{z}" for z in range(1, 15)], fill_value=0
    )
    return dummies


# ── 4. Normalized plate location ─────────────────────────────────────────────
def _normalized_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Express plate_x and plate_z relative to the batter's personal strike zone.
    plate_z_norm = 0 → bottom of zone, 1 → top of zone
    plate_x is already on a consistent scale (~-1.5 to 1.5 ft)
    """
    sz_range = (df["sz_top"] - df["sz_bot"]).replace(0, np.nan)
    sz_mid   = (df["sz_top"] + df["sz_bot"]) / 2

    plate_z_norm = (df["plate_z"] - df["sz_bot"]) / sz_range
    plate_z_dist = (df["plate_z"] - sz_mid).abs()      # distance from zone midpoint
    plate_x_abs  = df["plate_x"].abs()                 # distance from center

    return pd.DataFrame({
        "plate_z_norm":  plate_z_norm,
        "plate_z_dist":  plate_z_dist,
        "plate_x_abs":   plate_x_abs,
    }, index=df.index)


# ── 5. Pitch tunneling ───────────────────────────────────────────────────────
# Approximation: project pitch trajectory back to ~23ft from plate (decision point).
# Uses linear interpolation between release and plate.

RELEASE_Z_AVG    = 6.0   # ft, rough average release height
RELEASE_DIST_FT  = 55.0  # ft from plate to release point
TUNNEL_DIST_FT   = 23.0  # ft from plate: batter decision point
RATIO = (RELEASE_DIST_FT - TUNNEL_DIST_FT) / RELEASE_DIST_FT  # ~0.58


def _tunnel_position(df: pd.DataFrame):
    """
    Estimate x, z position at the tunnel point for each pitch.
    Returns two Series: tunnel_x, tunnel_z
    """
    # Linear interpolation: tunnel = release + ratio*(plate - release)
    tunnel_x = df["release_pos_x"] + RATIO * (df["plate_x"] - df["release_pos_x"])
    tunnel_z = df["release_pos_z"] + RATIO * (df["plate_z"] - df["release_pos_z"])
    return tunnel_x, tunnel_z


def _tunneling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each pitch, compute 3D distance at the tunnel point from the
    *previous* pitch in the same at-bat.
    """
    tun_x, tun_z = _tunnel_position(df)
    df = df.copy()
    df["_tun_x"] = tun_x
    df["_tun_z"] = tun_z

    # Shift within at-bat groups
    grp = ["game_pk", "at_bat_number"]
    df["_prev_tun_x"] = df.groupby(grp)["_tun_x"].shift(1)
    df["_prev_tun_z"] = df.groupby(grp)["_tun_z"].shift(1)

    dx = df["_tun_x"] - df["_prev_tun_x"]
    dz = df["_tun_z"] - df["_prev_tun_z"]
    tunnel_dist = np.sqrt(dx**2 + dz**2)

    # Also grab raw velocity gap
    df["_prev_velo"] = df.groupby(grp)["release_speed"].shift(1)
    velo_tunnel_gap = (df["release_speed"] - df["_prev_velo"]).abs()

    return pd.DataFrame({
        "tunnel_dist":     tunnel_dist.fillna(0),
        "velo_tunnel_gap": velo_tunnel_gap.fillna(0),
    }, index=df.index)


# ── 6. Previous pitch info ───────────────────────────────────────────────────
def _prev_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = ["game_pk", "at_bat_number"]

    prev_family = df.groupby(grp)["pitch_family"].shift(1).fillna("NONE")
    prev_result = df.groupby(grp)["swinging_strike"].shift(1).fillna(-1)   # -1 = first pitch
    prev_velo   = df.groupby(grp)["release_speed"].shift(1).fillna(df["release_speed"])

    # Encode pitch family as category int
    all_families = ["FF", "SI", "FC", "SL", "CU", "CH", "KN", "OT", "NONE"]
    fam_map = {f: i for i, f in enumerate(all_families)}
    prev_family_enc = prev_family.map(fam_map).fillna(len(all_families))

    return pd.DataFrame({
        "prev_pitch_family": prev_family_enc.astype(int),
        "prev_swinging_strike": prev_result.astype(float),
        "prev_velocity":    prev_velo,
    }, index=df.index)


# ── 7. Handedness interaction ────────────────────────────────────────────────
def _handedness_features(df: pd.DataFrame) -> pd.DataFrame:
    batter_r  = (df["stand"]    == "R").astype(int)
    pitcher_r = (df["p_throws"] == "R").astype(int)
    same_hand = (batter_r == pitcher_r).astype(int)

    # Flip plate_x so positive always = arm-side
    # For RHP vs RHB: positive plate_x = glove side for batter
    arm_side_x = df["plate_x"] * np.where(pitcher_r == 1, 1, -1)

    return pd.DataFrame({
        "batter_right":  batter_r,
        "pitcher_right": pitcher_r,
        "same_hand":     same_hand,
        "arm_side_x":    arm_side_x,
    }, index=df.index)


# ── 8. Movement features ─────────────────────────────────────────────────────
def _movement_features(df: pd.DataFrame) -> pd.DataFrame:
    pfx_x = df.get("pfx_x", pd.Series(0, index=df.index))
    pfx_z = df.get("pfx_z", pd.Series(0, index=df.index))
    total_movement = np.sqrt(pfx_x**2 + pfx_z**2)

    return pd.DataFrame({
        "pfx_x":          pfx_x.fillna(0),
        "pfx_z":          pfx_z.fillna(0),
        "total_movement": total_movement.fillna(0),
    }, index=df.index)


# ── 9. Pitch family encoding ─────────────────────────────────────────────────
def _pitch_family_dummies(df: pd.DataFrame) -> pd.DataFrame:
    families = ["FF", "SI", "FC", "SL", "CU", "CH", "KN", "OT"]
    dummies = pd.get_dummies(df["pitch_family"], prefix="pf").reindex(
        columns=[f"pf_{f}" for f in families], fill_value=0
    )
    return dummies


# ── MAIN ─────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Full feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame   cleaned Statcast data (output of data_loader.clean())

    Returns
    -------
    X            : pd.DataFrame  feature matrix (float64, no NaNs)
    y            : pd.Series     binary label (swinging_strike)
    feature_names: list[str]
    """
    print("Building features …")
    parts = []

    # Core physics
    core = pd.DataFrame({
        "release_speed":       df["release_speed"],
        "release_spin_rate":   df["release_spin_rate"].fillna(df["release_spin_rate"].median()),
        "release_extension":   df.get("release_extension", pd.Series(np.nan, index=df.index)).fillna(6.0),
        "release_pos_x":       df.get("release_pos_x",    pd.Series(0.0, index=df.index)).fillna(0),
        "release_pos_z":       df.get("release_pos_z",    pd.Series(6.0, index=df.index)).fillna(6.0),
        "plate_x":             df["plate_x"],
        "plate_z":             df["plate_z"],
    }, index=df.index)
    parts.append(core)

    # Velocity diff from pitcher/pitch-type average
    avg_velo = _pitcher_pitch_avg_velocity(df)
    parts.append(pd.DataFrame({
        "velo_diff_from_avg": df["release_speed"] - avg_velo,
    }, index=df.index))

    # Count
    parts.append(pd.DataFrame({
        "balls":           df["balls"].clip(0, 3),
        "strikes":         df["strikes"].clip(0, 2),
        "count_leverage":  _count_leverage(df["balls"], df["strikes"]),
    }, index=df.index))

    # Zone dummies
    parts.append(_zone_dummies(df))

    # Normalized location
    parts.append(_normalized_location(df))

    # Movement
    parts.append(_movement_features(df))

    # Handedness
    parts.append(_handedness_features(df))

    # Tunneling
    parts.append(_tunneling_features(df))

    # Previous pitch
    parts.append(_prev_pitch_features(df))

    # Pitch family dummies
    parts.append(_pitch_family_dummies(df))

    # ── Assemble ─────────────────────────────────────────────────────────────
    X = pd.concat(parts, axis=1).astype(float)

    # Final NaN imputation with column medians
    X = X.fillna(X.median())

    y = df["swinging_strike"].astype(int)
    feature_names = list(X.columns)

    print(f"  Features: {len(feature_names)} | Rows: {len(X):,} | "
          f"Positive rate: {y.mean():.2%}")

    return X, y, feature_names
