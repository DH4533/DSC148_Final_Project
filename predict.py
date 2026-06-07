"""
predict.py
----------
Run inference on a single pitch from command-line arguments.

Example:
    python predict.py \\
        --velocity 94.2 \\
        --spin_rate 2400 \\
        --extension 6.2 \\
        --release_x -1.5 \\
        --release_z 5.8 \\
        --plate_x 0.3 \\
        --plate_z 2.1 \\
        --balls 1 \\
        --strikes 2 \\
        --stand R \\
        --p_throws R \\
        --pitch_type FF \\
        --model xgboost
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models import MODEL_REGISTRY
from src.feature_engineering import build_features
from src.utils import model_save_path


PITCH_FAMILY_MAP = {
    "FF": "FF", "FA": "FF",
    "SI": "SI", "FT": "SI",
    "FC": "FC",
    "SL": "SL", "ST": "SL",
    "CU": "CU", "KC": "CU", "CS": "CU",
    "CH": "CH", "FS": "CH", "FO": "CH",
    "KN": "KN",
}

# Approximate season-average velocities by pitch type for demo
AVG_VELO_BY_PITCH = {
    "FF": 93.5, "SI": 92.5, "FC": 88.5,
    "SL": 84.0, "CU": 78.0, "CH": 84.5,
    "KN": 74.0, "OT": 85.0,
}


def single_pitch_to_df(args) -> pd.DataFrame:
    """
    Build a single-row DataFrame that mimics the cleaned Statcast format,
    then run feature engineering on it.

    Since tunneling and previous-pitch features require context,
    they default to 0 (first pitch / no context).
    """
    pitch_family = PITCH_FAMILY_MAP.get(args.pitch_type, "OT")
    avg_velo     = AVG_VELO_BY_PITCH.get(pitch_family, 88.0)

    row = {
        # Physics
        "release_speed":      args.velocity,
        "release_spin_rate":  args.spin_rate,
        "release_extension":  args.extension,
        "release_pos_x":      args.release_x,
        "release_pos_z":      args.release_z,
        "pfx_x":              args.pfx_x,
        "pfx_z":              args.pfx_z,

        # Location
        "plate_x":  args.plate_x,
        "plate_z":  args.plate_z,
        "sz_top":   args.sz_top,
        "sz_bot":   args.sz_bot,
        "zone":     args.zone,

        # Count
        "balls":    args.balls,
        "strikes":  args.strikes,

        # Handedness
        "stand":    args.stand,
        "p_throws": args.p_throws,

        # Pitch identity
        "pitch_type":   args.pitch_type,
        "pitch_family": pitch_family,

        # Required for feature_engineering internals
        "pitcher": 0,
        "game_pk": 0,
        "at_bat_number": 0,
        "pitch_number":  1,

        # Label placeholder (unused during inference)
        "swinging_strike": 0,
        "description": "unknown",
    }
    return pd.DataFrame([row])


def load_model(model_name: str):
    path = model_save_path(model_name)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run `python train.py --model {model_name}` first."
        )
    cls_map = {
        "logistic":      "LogisticModel",
        "random_forest": "RandomForestModel",
        "xgboost":       "XGBoostModel",
        "neural_net":    "NeuralNetModel",
    }
    from src import models as m
    cls = getattr(m, cls_map[model_name])
    return cls.load(path)


def main():
    parser = argparse.ArgumentParser(description="Predict swing-and-miss probability")

    # Pitch inputs
    parser.add_argument("--velocity",   type=float, required=True)
    parser.add_argument("--spin_rate",  type=float, required=True)
    parser.add_argument("--extension",  type=float, default=6.2)
    parser.add_argument("--release_x",  type=float, default=-1.5)
    parser.add_argument("--release_z",  type=float, default=5.8)
    parser.add_argument("--pfx_x",      type=float, default=0.0, help="Horizontal movement (ft)")
    parser.add_argument("--pfx_z",      type=float, default=0.0, help="Vertical movement (ft)")
    parser.add_argument("--plate_x",    type=float, required=True)
    parser.add_argument("--plate_z",    type=float, required=True)
    parser.add_argument("--sz_top",     type=float, default=3.4, help="Strike zone top (ft)")
    parser.add_argument("--sz_bot",     type=float, default=1.6, help="Strike zone bottom (ft)")
    parser.add_argument("--zone",       type=int,   default=5,   help="Statcast zone (1-14)")
    parser.add_argument("--balls",      type=int,   required=True)
    parser.add_argument("--strikes",    type=int,   required=True)
    parser.add_argument("--stand",      choices=["L", "R"], required=True)
    parser.add_argument("--p_throws",   choices=["L", "R"], required=True)
    parser.add_argument("--pitch_type", default="FF",
                        help="Statcast pitch type code (FF, SL, CH, CU, SI, FC, …)")
    parser.add_argument("--model",      default="xgboost",
                        choices=list(MODEL_REGISTRY))

    args = parser.parse_args()

    print(f"\n⚾  Pitch: {args.pitch_type} {args.velocity:.1f} mph  |  "
          f"Spin: {args.spin_rate:.0f} rpm  |  "
          f"Location: ({args.plate_x:.2f}, {args.plate_z:.2f})  |  "
          f"Count: {args.balls}-{args.strikes}")

    # Build input
    df = single_pitch_to_df(args)
    X, _, feature_names = build_features(df)

    # Load model & predict
    model = load_model(args.model)
    prob  = model.predict_proba(X.values)[0]
    pred  = "SWINGING STRIKE ★" if prob >= 0.5 else "no swing"

    print(f"\n  Model:       {args.model}")
    print(f"  Probability: {prob:.1%}")
    print(f"  Prediction:  {pred}")

    # Rough context
    if prob > 0.35:
        print(f"\n  ⚠️  High whiff potential — great pitch to throw here!")
    elif prob > 0.20:
        print(f"\n  📊  Above-average swing-and-miss likelihood.")
    else:
        print(f"\n  📈  Low whiff probability for this pitch/location combo.")


if __name__ == "__main__":
    main()
