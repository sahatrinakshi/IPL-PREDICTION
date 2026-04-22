import json
import os
from http.server import BaseHTTPRequestHandler

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index':
            # Load predictions
            path = os.path.join(MODEL_DIR, 'predictions.json')
            if not os.path.exists(path):
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                html = """
                <html><body style='font-family:sans-serif;padding:40px'>
                <h2>Model not trained yet.</h2>
                <p>Run: <code>python src/train_model.py</code></p>
                </body></html>
                """
                self.wfile.write(html.encode())
                return

            with open(path) as f:
                predictions = json.load(f)

            # Load dashboard HTML
            template_path = os.path.join(TEMPLATE_DIR, 'dashboard.html')
            with open(template_path) as f:
                html_content = f.read()

            # Sort teams by probability
            if 'team_predictions' in predictions:
                sorted_teams = sorted(
                    predictions['team_predictions'].items(),
                    key=lambda x: x[1]['championship_probability'],
                    reverse=True
                )
                predictions['team_predictions_sorted'] = sorted_teams

            # Inject data into HTML
            html_with_data = html_content.replace(
                '</head>',
                f'<script>window.__DATA__ = {json.dumps(predictions)};</script></head>'
            )

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html_with_data.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()
