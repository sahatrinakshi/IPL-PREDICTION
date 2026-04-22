import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

def handler(request):
    try:
        path = os.path.join(MODEL_DIR, 'predictions.json')
        if not os.path.exists(path):
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'text/html'},
                'body': '<html><body style="font-family:sans-serif;padding:40px"><h2>Model not trained yet.</h2></body></html>'
            }

        with open(path) as f:
            predictions = json.load(f)

        template_path = os.path.join(TEMPLATE_DIR, 'dashboard.html')
        with open(template_path) as f:
            html_content = f.read()

        if 'team_predictions' in predictions:
            sorted_teams = sorted(
                predictions['team_predictions'].items(),
                key=lambda x: x[1]['championship_probability'],
                reverse=True
            )
            predictions['team_predictions_sorted'] = sorted_teams

        html_with_data = html_content.replace(
            '</head>',
            f'<script>window.__DATA__ = {json.dumps(predictions)};</script></head>'
        )

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            },
            'body': html_with_data
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }
