#!/usr/bin/env bash
# IPL 2026 Predictor - One-click setup and run

set -e
cd "$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🏏 IPL 2026 Championship Predictor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "► Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "► Installing dependencies..."
pip install -q -r requirements.txt

# Check data file
if [ ! -f "data/IPL.csv" ]; then
    echo "ERROR: data/IPL.csv not found!"
    echo "Please place IPL.csv in the data/ folder."
    exit 1
fi

echo "► Training ML model (this may take 30-60 seconds)..."
python src/train_model.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Model trained! Starting dashboard..."
echo "  🌐 Open http://localhost:5000 in your browser"
echo "  Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python src/app.py
