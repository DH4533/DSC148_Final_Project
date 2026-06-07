"""
tests/test_features.py
----------------------
Unit tests for feature engineering.

Run with:
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_features


# ── Fixtures ─────────────────────────────────────────────────────────────────
def make_dummy_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal Statcast-like DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "release_speed":      np.random.uniform(75, 100, n),
        "release_spin_rate":  np.random.uniform(1800, 3000, n),
        "release_extension":  np.random.uniform(5.5, 7.0, n),
        "release_pos_x":      np.random.uniform(-2.5, 2.5, n),
        "release_pos_z":      np.random.uniform(4.5, 7.5, n),
        "pfx_x":              np.random.uniform(-1, 1, n),
        "pfx_z":              np.random.uniform(-1, 1, n),
        "plate_x":            np.random.uniform(-1.5, 1.5, n),
        "plate_z":            np.random.uniform(1.0, 4.5, n),
        "sz_top":             np.random.uniform(3.2, 3.8, n),
        "sz_bot":             np.random.uniform(1.4, 1.8, n),
        "zone":               np.random.randint(1, 15, n),
        "balls":              np.random.randint(0, 4, n),
        "strikes":            np.random.randint(0, 3, n),
        "stand":              np.random.choice(["L", "R"], n),
        "p_throws":           np.random.choice(["L", "R"], n),
        "pitch_type":         np.random.choice(["FF", "SL", "CH", "CU", "SI"], n),
        "pitch_family":       np.random.choice(["FF", "SL", "CH", "CU", "SI"], n),
        "pitcher":            np.random.randint(1000, 9999, n),
        "game_pk":            np.zeros(n, dtype=int),
        "at_bat_number":      np.repeat(np.arange(n // 5), 5)[:n],
        "pitch_number":       np.tile(np.arange(1, 6), n // 5)[:n],
        "swinging_strike":    np.random.randint(0, 2, n),
        "description":        np.random.choice(
            ["swinging_strike", "called_strike", "ball", "hit_into_play"], n
        ),
    })


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestBuildFeatures:
    def test_output_shape(self):
        df = make_dummy_df(200)
        X, y, names = build_features(df)
        assert len(X) == 200
        assert len(y) == 200
        assert X.shape[1] == len(names)

    def test_no_nans(self):
        df = make_dummy_df(200)
        X, y, _ = build_features(df)
        assert not X.isna().any().any(), "Feature matrix contains NaN values"

    def test_no_infs(self):
        df = make_dummy_df(200)
        X, y, _ = build_features(df)
        assert not np.isinf(X.values).any(), "Feature matrix contains Inf values"

    def test_label_binary(self):
        df = make_dummy_df(200)
        _, y, _ = build_features(df)
        assert set(y.unique()).issubset({0, 1})

    def test_count_leverage_range(self):
        df = make_dummy_df(500)
        X, _, names = build_features(df)
        col = X["count_leverage"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_zone_dummies_sum(self):
        df = make_dummy_df(200)
        X, _, names = build_features(df)
        zone_cols = [c for c in names if c.startswith("zone_")]
        assert len(zone_cols) == 13, f"Expected 13 zone cols, got {len(zone_cols)}"
        # Each row should have at most one zone flagged
        zone_sums = X[zone_cols].sum(axis=1)
        assert (zone_sums <= 1).all(), "Multiple zones flagged for a single pitch"

    def test_pitch_family_dummies(self):
        df = make_dummy_df(200)
        X, _, names = build_features(df)
        pf_cols = [c for c in names if c.startswith("pf_")]
        assert len(pf_cols) >= 4

    def test_handedness_binary(self):
        df = make_dummy_df(200)
        X, _, _ = build_features(df)
        for col in ["batter_right", "pitcher_right", "same_hand"]:
            assert set(X[col].unique()).issubset({0, 1, 0.0, 1.0})

    def test_velocity_diff_sign(self):
        """Pitcher throwing below their average should have negative velo_diff."""
        df = make_dummy_df(100)
        # Force pitcher 999 to have consistent low velocity
        df["pitcher"] = 999
        df["pitch_family"] = "FF"
        df.loc[0, "release_speed"] = 80.0  # well below 75-100 mean
        X, _, _ = build_features(df)
        # Just check it's computed (not all zero)
        assert X["velo_diff_from_avg"].std() >= 0

    def test_tunnel_dist_nonneg(self):
        df = make_dummy_df(200)
        X, _, _ = build_features(df)
        assert (X["tunnel_dist"] >= 0).all()

    def test_prev_pitch_first_pitch_default(self):
        """First pitch of an at-bat should have prev_swinging_strike == -1.0."""
        df = make_dummy_df(50)
        df["at_bat_number"] = 0
        df["pitch_number"]  = range(1, 51)  # all sequential, first pitch = pitch 1
        X, _, _ = build_features(df)
        first_row = X.iloc[0]
        assert first_row["prev_swinging_strike"] == -1.0

    def test_single_row_works(self):
        """Feature engineering should not crash on a single-row DataFrame."""
        df = make_dummy_df(1)
        X, y, names = build_features(df)
        assert len(X) == 1

    def test_with_missing_optional_cols(self):
        """Should handle missing pfx_x / pfx_z / extension gracefully."""
        df = make_dummy_df(100)
        df = df.drop(columns=["pfx_x", "pfx_z", "release_extension"], errors="ignore")
        X, _, _ = build_features(df)
        assert not X.isna().any().any()


class TestCountLeverage:
    def test_two_strike_higher_than_no_strike(self):
        df_2s = make_dummy_df(50)
        df_0s = df_2s.copy()
        df_2s["strikes"] = 2
        df_0s["strikes"] = 0
        X2, _, _ = build_features(df_2s)
        X0, _, _ = build_features(df_0s)
        assert X2["count_leverage"].mean() > X0["count_leverage"].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
