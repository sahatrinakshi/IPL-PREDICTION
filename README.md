# 🏏 IPL 2026 Championship Predictor

An ML-powered prediction dashboard for the IPL 2026 season built with XGBoost and Flask.

## Features
- **XGBoost ML model** trained on IPL 2008–2026 ball-by-ball data
- Predicts championship probabilities for all 10 IPL 2026 teams
- Beautiful interactive dashboard with Plotly charts
- Historical champions timeline, team stats, venue analysis

## Quick Start

### Requirements
- Python 3.8+
- `data/IPL.csv` (already included)

### Run (Linux / macOS)
```bash
chmod +x run.sh
./run.sh
```

### Run (Windows)
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python src/app.py
```

Then open **http://localhost:5000** in your browser.

## Project Structure
```
ipl_predictor/
├── data/
│   └── IPL.csv              # Ball-by-ball dataset 2008-2026
├── models/
│   ├── xgb_model.pkl        # Trained XGBoost model
│   ├── le_team.pkl          # Label encoder (teams)
│   ├── le_venue.pkl         # Label encoder (venues)
│   └── predictions.json     # Pre-computed predictions
├── src/
│   ├── train_model.py       # ML training pipeline
│   └── app.py               # Flask dashboard server
├── templates/
│   └── dashboard.html       # Main dashboard UI
├── requirements.txt
├── run.sh                   # One-click launcher (Unix)
└── README.md
```

## ML Model
- **Algorithm**: XGBoost Classifier
- **Features**: Team win rates, head-to-head records, venue encoding, toss outcomes, current 2026 form
- **Validation**: 5-fold cross-validation
- **Prediction**: Weighted blend of 2026 current form (45%), recent form last 2 seasons (35%), historical win rate (20%)

## Dashboard Sections
1. **Predictions** — Championship probability cards for each team
2. **Analysis** — 8 interactive Plotly charts (probability distribution, scatter, venue analysis, season trends)
3. **History** — All IPL champions 2008–2025
4. **Matches** — All 2026 matches played so far
