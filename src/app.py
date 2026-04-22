"""
IPL 2026 Prediction Dashboard - Flask App
"""

import os
import json
from flask import Flask, render_template, jsonify

app = Flask(__name__, template_folder='../templates', static_folder='../static')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

TEAM_COLORS = {
    'Mumbai Indians': '#004B9F',
    'Chennai Super Kings': '#F7A721',
    'Royal Challengers Bengaluru': '#CB2026',
    'Royal Challengers Bangalore': '#CB2026',
    'Kolkata Knight Riders': '#552E8E',
    'Delhi Capitals': '#17479E',
    'Delhi Daredevils': '#17479E',
    'Rajasthan Royals': '#C91A7B',
    'Sunrisers Hyderabad': '#F7991D',
    'Punjab Kings': '#D71921',
    'Kings XI Punjab': '#D71921',
    'Gujarat Titans': '#1B2133',
    'Lucknow Super Giants': '#2ECC71',
    'Deccan Chargers': '#FF6B00',
    'Rising Pune Supergiant': '#9B59B6',
}

TEAM_INITIALS = {
    'Mumbai Indians': 'MI',
    'Chennai Super Kings': 'CSK',
    'Royal Challengers Bengaluru': 'RCB',
    'Royal Challengers Bangalore': 'RCB',
    'Kolkata Knight Riders': 'KKR',
    'Delhi Capitals': 'DC',
    'Delhi Daredevils': 'DD',
    'Rajasthan Royals': 'RR',
    'Sunrisers Hyderabad': 'SRH',
    'Punjab Kings': 'PBKS',
    'Kings XI Punjab': 'KXIP',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG',
    'Deccan Chargers': 'DC',
    'Rising Pune Supergiant': 'RPS',
}


@app.template_global('team_color')
def team_color(team):
    return TEAM_COLORS.get(team, '#7b5cff')


@app.template_global('team_initials')
def team_initials(team):
    if team in TEAM_INITIALS:
        return TEAM_INITIALS[team]
    words = team.split()
    return ''.join(w[0] for w in words[:3]).upper()


def load_predictions():
    path = os.path.join(MODEL_DIR, 'predictions.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if 'team_predictions' in data:
        data['team_predictions_sorted'] = sorted(
            data['team_predictions'].items(),
            key=lambda x: x[1]['championship_probability'],
            reverse=True
        )
    return data


@app.route('/')
def index():
    data = load_predictions()
    if data is None:
        return (
            "<h2 style='font-family:sans-serif;padding:40px'>Model not trained yet.<br>"
            "Run: <code>python src/train_model.py</code></h2>",
            500,
        )
    return render_template('dashboard.html', data=data)


@app.route('/api/predictions')
def api_predictions():
    data = load_predictions()
    if data is None:
        return jsonify({'error': 'Model not trained'}), 500
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
