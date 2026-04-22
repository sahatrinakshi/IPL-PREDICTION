"""
IPL 2026 Winner Prediction - ML Model Training
Uses ball-by-ball data to build features and train an XGBoost model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'IPL.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_and_prepare_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['season'] = df['season'].astype(str)
    return df


def extract_match_features(df):
    """Extract per-match features from ball-by-ball data"""
    print("Extracting match-level features...")

    match_ids = df['match_id'].unique()
    records = []

    for mid in match_ids:
        m = df[df['match_id'] == mid]
        row0 = m.iloc[0]

        team1 = row0['batting_team']
        team2 = row0['bowling_team']
        winner = row0['match_won_by']
        season = row0['season']
        venue = row0['venue']
        toss_winner = row0['toss_winner']
        toss_decision = row0['toss_decision']
        stage = row0['stage']

        # Skip matches with unknown result
        if winner not in [team1, team2]:
            continue

        # Innings stats
        inn1 = m[m['innings'] == 1]
        inn2 = m[m['innings'] == 2]

        inn1_runs = inn1['runs_total'].sum()
        inn2_runs = inn2['runs_total'].sum()
        inn1_wickets = inn1['wicket_kind'].notna().sum()
        inn2_wickets = inn2['wicket_kind'].notna().sum()
        inn1_balls = inn1['valid_ball'].sum()
        inn2_balls = inn2['valid_ball'].sum()

        batting_first_team = inn1['batting_team'].iloc[0] if not inn1.empty else team1
        batting_second_team = inn2['batting_team'].iloc[0] if not inn2.empty else team2

        batting_first_won = 1 if winner == batting_first_team else 0

        records.append({
            'match_id': mid,
            'season': season,
            'team1': batting_first_team,
            'team2': batting_second_team,
            'winner': winner,
            'venue': venue,
            'toss_winner': toss_winner,
            'toss_decision': str(toss_decision),
            'stage': stage,
            'inn1_runs': inn1_runs,
            'inn2_runs': inn2_runs,
            'inn1_wickets': inn1_wickets,
            'inn2_wickets': inn2_wickets,
            'inn1_rr': round(inn1_runs / max(inn1_balls, 1) * 6, 2),
            'inn2_rr': round(inn2_runs / max(inn2_balls, 1) * 6, 2),
            'batting_first_won': batting_first_won,
        })

    match_df = pd.DataFrame(records)
    print(f"Total valid matches: {len(match_df)}")
    return match_df


def build_team_performance_features(match_df):
    """Rolling win rates and performance stats per team per season"""
    print("Building team performance features...")

    # Season-by-season team stats
    season_order = sorted(match_df['season'].unique(), key=lambda x: str(x))
    team_season_stats = {}

    for season in season_order:
        season_matches = match_df[match_df['season'] == season]
        teams = set(season_matches['team1'].tolist() + season_matches['team2'].tolist())
        for team in teams:
            played = season_matches[(season_matches['team1'] == team) | (season_matches['team2'] == team)]
            wins = played[played['winner'] == team]
            key = (team, season)
            team_season_stats[key] = {
                'played': len(played),
                'wins': len(wins),
                'win_rate': len(wins) / max(len(played), 1)
            }

    return team_season_stats, season_order


def compute_h2h_stats(match_df):
    """Head-to-head win rates between teams"""
    h2h = {}
    for _, row in match_df.iterrows():
        t1, t2, w = row['team1'], row['team2'], row['winner']
        key = tuple(sorted([t1, t2]))
        if key not in h2h:
            h2h[key] = {key[0]: 0, key[1]: 0}
        if w in h2h[key]:
            h2h[key][w] += 1
    return h2h


def encode_features(match_df, team_season_stats, h2h):
    """Encode all features for the model"""
    print("Encoding features...")

    all_teams = sorted(set(match_df['team1'].tolist() + match_df['team2'].tolist()))
    all_venues = sorted(match_df['venue'].unique())
    seasons = sorted(match_df['season'].unique(), key=str)

    le_team = LabelEncoder().fit(all_teams)
    le_venue = LabelEncoder().fit(all_venues)

    rows = []
    for _, row in match_df.iterrows():
        t1, t2 = row['team1'], row['team2']
        s = row['season']

        t1_stats = team_season_stats.get((t1, s), {'win_rate': 0.5, 'played': 0})
        t2_stats = team_season_stats.get((t2, s), {'win_rate': 0.5, 'played': 0})

        h2h_key = tuple(sorted([t1, t2]))
        h2h_data = h2h.get(h2h_key, {t1: 1, t2: 1})
        t1_h2h = h2h_data.get(t1, 0)
        t2_h2h = h2h_data.get(t2, 0)
        h2h_total = t1_h2h + t2_h2h
        t1_h2h_rate = t1_h2h / max(h2h_total, 1)

        toss_adv = 1 if row['toss_winner'] == t1 else 0

        try:
            t1_enc = le_team.transform([t1])[0]
            t2_enc = le_team.transform([t2])[0]
            venue_enc = le_venue.transform([row['venue']])[0]
        except Exception:
            t1_enc = 0
            t2_enc = 0
            venue_enc = 0

        rows.append({
            'team1_enc': t1_enc,
            'team2_enc': t2_enc,
            'venue_enc': venue_enc,
            'team1_win_rate': t1_stats['win_rate'],
            'team2_win_rate': t2_stats['win_rate'],
            'team1_h2h_rate': t1_h2h_rate,
            'toss_advantage': toss_adv,
            'batting_first_won': row['batting_first_won'],
        })

    X = pd.DataFrame(rows).drop(columns=['batting_first_won'])
    y = pd.DataFrame(rows)['batting_first_won']

    # Target: team1 won
    return X, y, le_team, le_venue


def train_model(X, y):
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    return model, cv_scores.mean()


def compute_2026_probabilities(match_df, team_season_stats, h2h, model, le_team, le_venue):
    """Compute win probabilities for each team in 2026"""
    print("Computing 2026 win probabilities...")

    teams_2026 = [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru',
        'Kolkata Knight Riders', 'Delhi Capitals', 'Rajasthan Royals',
        'Sunrisers Hyderabad', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
    ]

    team_probs = {}

    # Use 2026 matches already in dataset (partial season)
    matches_2026 = match_df[match_df['season'] == '2026']

    for team in teams_2026:
        # Wins so far in 2026
        if len(matches_2026) > 0:
            played = matches_2026[(matches_2026['team1'] == team) | (matches_2026['team2'] == team)]
            wins = played[played['winner'] == team]
            current_wr = len(wins) / max(len(played), 1)
            games_played = len(played)
        else:
            current_wr = 0.5
            games_played = 0

        # Historical performance (all seasons)
        historical_wins = sum(
            v['wins'] for (t, s), v in team_season_stats.items() if t == team
        )
        historical_played = sum(
            v['played'] for (t, s), v in team_season_stats.items() if t == team
        )
        hist_wr = historical_wins / max(historical_played, 1)

        # Recent form (last 2 seasons)
        recent_seasons = ['2024', '2025']
        recent_wins = sum(
            team_season_stats.get((team, s), {}).get('wins', 0) for s in recent_seasons
        )
        recent_played = sum(
            team_season_stats.get((team, s), {}).get('played', 0) for s in recent_seasons
        )
        recent_wr = recent_wins / max(recent_played, 1)

        # Weighted score
        if games_played > 0:
            score = 0.45 * current_wr + 0.35 * recent_wr + 0.20 * hist_wr
        else:
            score = 0.5 * recent_wr + 0.5 * hist_wr

        team_probs[team] = {
            'score': score,
            'current_wr': round(current_wr * 100, 1),
            'historical_wr': round(hist_wr * 100, 1),
            'recent_wr': round(recent_wr * 100, 1),
            'games_played_2026': games_played,
            'wins_2026': int(len(wins)) if games_played > 0 else 0,
        }

    # Normalize to probabilities
    total = sum(v['score'] for v in team_probs.values())
    for team in team_probs:
        team_probs[team]['championship_probability'] = round(
            team_probs[team]['score'] / total * 100, 2
        )

    return team_probs


def compute_historical_stats(match_df):
    """Season-by-season champions and stats for the dashboard"""
    finals = match_df[match_df['stage'] == 'Final']
    champions = {}
    for _, row in finals.iterrows():
        champions[row['season']] = row['winner']

    # Team overall stats
    all_teams = sorted(set(match_df['team1'].tolist() + match_df['team2'].tolist()))
    team_stats = {}
    for team in all_teams:
        played = match_df[(match_df['team1'] == team) | (match_df['team2'] == team)]
        wins = played[played['winner'] == team]
        titles = sum(1 for s, w in champions.items() if w == team)
        seasons_played = played['season'].nunique()
        team_stats[team] = {
            'total_played': len(played),
            'total_wins': len(wins),
            'win_rate': round(len(wins) / max(len(played), 1) * 100, 1),
            'titles': titles,
            'seasons': seasons_played
        }

    # Season stats
    seasons_data = []
    for season in sorted(match_df['season'].unique(), key=str):
        sm = match_df[match_df['season'] == season]
        champ = champions.get(season, 'TBD')
        seasons_data.append({
            'season': season,
            'champion': champ,
            'total_matches': len(sm),
        })

    # Venue win rates (batting first)
    venue_stats = match_df.groupby('venue').agg(
        matches=('match_id', 'count'),
        batting_first_wins=('batting_first_won', 'sum')
    ).reset_index()
    venue_stats['bat_first_pct'] = round(
        venue_stats['batting_first_wins'] / venue_stats['matches'] * 100, 1
    )
    venue_stats = venue_stats[venue_stats['matches'] >= 5].nlargest(15, 'matches')

    return {
        'champions': champions,
        'team_stats': team_stats,
        'seasons': seasons_data,
        'venue_stats': venue_stats.to_dict(orient='records')
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_prepare_data()
    match_df = extract_match_features(df)
    team_season_stats, season_order = build_team_performance_features(match_df)
    h2h = compute_h2h_stats(match_df)
    X, y, le_team, le_venue = encode_features(match_df, team_season_stats, h2h)

    model, cv_acc = train_model(X, y)

    # Save model
    joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_model.pkl'))
    joblib.dump(le_team, os.path.join(MODEL_DIR, 'le_team.pkl'))
    joblib.dump(le_venue, os.path.join(MODEL_DIR, 'le_venue.pkl'))

    # Compute predictions
    team_probs = compute_2026_probabilities(match_df, team_season_stats, h2h, model, le_team, le_venue)
    historical = compute_historical_stats(match_df)

    # 2026 match results so far
    matches_2026 = match_df[match_df['season'] == '2026'][
        ['match_id', 'team1', 'team2', 'winner', 'stage']
    ].to_dict(orient='records')

    # Save all data
    output = {
        'cv_accuracy': round(cv_acc * 100, 2),
        'team_predictions': team_probs,
        'historical': historical,
        'matches_2026': matches_2026,
    }

    with open(os.path.join(MODEL_DIR, 'predictions.json'), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[SUCCESS] Model trained with {cv_acc*100:.1f}% CV accuracy")
    print("\nIPL 2026 Championship Probabilities:")
    sorted_teams = sorted(team_probs.items(), key=lambda x: x[1]['championship_probability'], reverse=True)
    for team, stats in sorted_teams:
        print(f"  {team}: {stats['championship_probability']}%")

    print(f"\nData saved to {MODEL_DIR}/predictions.json")
    return output


if __name__ == '__main__':
    main()
