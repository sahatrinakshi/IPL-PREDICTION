import json
import os
from http.server import BaseHTTPRequestHandler

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/predictions':
            path = os.path.join(MODEL_DIR, 'predictions.json')
            if not os.path.exists(path):
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Model not trained'}).encode())
                return

            with open(path) as f:
                data = json.load(f)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
