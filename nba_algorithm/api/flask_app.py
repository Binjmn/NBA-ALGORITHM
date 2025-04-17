"""
Flask App Module for NBA Prediction System

This module provides a Flask application that can be imported by other modules.
It sets up the Flask app with proper template and static file paths.
"""

import os
from pathlib import Path
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS

# Define project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Create Flask application with explicit template and static folders
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'ui', 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'ui', 'static')
)

# Enable CORS for all routes
CORS(app)

# Configure app
app.config['JSON_SORT_KEYS'] = False  # Preserve order of JSON keys for readability
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Auto-reload templates when they are changed

# Basic routes for templates and static files
@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Render the dashboard homepage"""
    return render_template('dashboard.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, images, etc.)"""
    return send_from_directory(app.static_folder, path)

# For testing the Flask app in isolation
if __name__ == '__main__':
    print(f"\nTemplate folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    print(f"\nAvailable templates:")
    for root, dirs, files in os.walk(app.template_folder):
        for file in files:
            print(f" - {file}")
    
    print(f"\nStarting Flask app in debug mode...")
    app.run(debug=True, host='0.0.0.0', port=5000)
