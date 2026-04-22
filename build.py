#!/usr/bin/env python3
import json
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

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


def team_color(team):
    return TEAM_COLORS.get(team, '#7b5cff')


def team_initials(team):
    if team in TEAM_INITIALS:
        return TEAM_INITIALS[team]
    words = team.split()
    return ''.join(w[0] for w in words[:3]).upper()


with open("models/predictions.json") as f:
    data = json.load(f)

if 'team_predictions' in data:
    data['team_predictions_sorted'] = sorted(
        data['team_predictions'].items(),
        key=lambda x: x[1]['championship_probability'],
        reverse=True,
    )

env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html', 'xml']),
)
env.globals['team_color'] = team_color
env.globals['team_initials'] = team_initials

html = env.get_template('dashboard.html').render(data=data)

os.makedirs('public', exist_ok=True)
with open('public/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Build complete!')
