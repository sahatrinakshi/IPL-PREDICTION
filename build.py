#!/usr/bin/env python3
import json
import os
import subprocess

# Train model
print("Training model...")
subprocess.run(["python", "src/train_model.py"], check=True)

# Create public directory
os.makedirs("public", exist_ok=True)

# Load predictions
with open("models/predictions.json") as f:
    predictions = json.load(f)

# Load dashboard HTML
with open("templates/dashboard.html") as f:
    html = f.read()

# Inject data into HTML
html = html.replace(
    '</head>',
    f'<script>window.__DATA__ = {json.dumps(predictions)};</script></head>'
)

# Write to public
with open("public/index.html", "w") as f:
    f.write(html)

print("✅ Build complete!")
