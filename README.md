# ⚾ Swing-and-Miss Probability Predictor

**DSC 148 – Course Project | UCSD**

Predicting whether a pitch results in a swinging strike using Statcast data from Baseball Savant.

---

## Project Structure

```
swing_miss_project/
├── data/                    # Raw and processed data (git-ignored)
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py       # Statcast API fetching & caching
│   ├── feature_engineering.py
│   ├── models.py            # LR, RF, XGBoost, Neural Net
│   ├── evaluate.py          # Metrics & plotting
│   └── utils.py
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
├── results/                 # Saved figures & metrics
├── tests/
│   └── test_features.py
├── train.py                 # Main training script
├── predict.py               # Inference on new pitches
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone & enter the repo
git clone https://github.com/<your-username>/swing-miss-predictor.git
cd swing-miss-predictor

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data

Data is pulled automatically from [Baseball Savant](https://baseballsavant.mlb.com/) via the `pybaseball` library, which wraps the Statcast API.

```bash
# Fetch data for a season range (downloads ~500k+ pitches)
python src/data_loader.py --start 2022-04-01 --end 2023-10-01
```

Data is cached locally in `data/raw/` to avoid re-downloading.

---

## Training

```bash
# Train all models and save results
python train.py

# Train a specific model
python train.py --model xgboost

# Options: logistic, random_forest, xgboost, neural_net, all
```

---

## Prediction

```bash
# Predict on a single pitch (example)
python predict.py \
  --velocity 94.2 \
  --spin_rate 2400 \
  --extension 6.2 \
  --release_x -1.5 \
  --release_z 5.8 \
  --plate_x 0.3 \
  --plate_z 2.1 \
  --balls 1 \
  --strikes 2 \
  --stand R \
  --p_throws R \
  --pitch_type FF
```

---

## Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline; interpretable coefficients |
| Random Forest | Ensemble; captures non-linear interactions |
| XGBoost | Gradient boosting; typically best tabular performance |
| Neural Network | MLP with batch norm; learns complex pitch tunneling |

---

## Evaluation

All models are compared on a held-out 2023 test set using:
- **Accuracy** – overall correctness
- **F1 Score** – balances precision/recall (important: class imbalance ~10% swinging strikes)
- **ROC-AUC** – probability calibration quality

Results and plots saved to `results/`.

---

## Key Features

| Feature | Description |
|---------|-------------|
| `velo_diff_from_avg` | Velocity relative to pitcher's season average |
| `zone_*` | Binary zone indicators (Statcast 9-zone + balls) |
| `tunnel_dist` | 3D distance to previous pitch at decision point |
| `prev_pitch_type` | Encoded type of pitch thrown before |
| `prev_pitch_result` | Whether previous pitch was called/swung/missed |
| `count_leverage` | Custom count-pressure index |

---

## Results Summary

Results are generated after running `train.py`. A `results/comparison.png` chart is produced comparing all four models.
