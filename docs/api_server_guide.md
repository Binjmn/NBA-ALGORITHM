# NBA Prediction System API Server Guide

## Overview

This guide explains how to start, configure, and use the NBA Prediction System API server. The API server provides endpoints for managing model training, accessing performance metrics, and generating predictions.

## Prerequisites

- Python 3.8+ installed
- NBA Prediction System codebase
- Required Python packages (Flask, Flask-CORS, etc.)
- PostgreSQL database properly configured (if using database features)

## Starting the API Server

### Using PowerShell

1. **Open PowerShell**

   Right-click on the Start menu and select "Windows PowerShell" or search for PowerShell in the start menu.

2. **Navigate to your project directory**

   ```powershell
   cd "c:\Users\bcada\CascadeProjects\NBA ALGORITHM"
   ```

3. **Start the API server**

   ```powershell
   python -m src.api.server
   ```

   This will start the server on the default host (0.0.0.0) and port (5000).

4. **With custom host and port**

   ```powershell
   $env:API_HOST="127.0.0.1"
   $env:API_PORT="8080"
   python -m src.api.server
   ```

5. **With API token for authentication**

   ```powershell
   $env:API_TOKEN="your_secure_token_here"
   python -m src.api.server
   ```

### Using Command Prompt

1. **Open Command Prompt**

   Search for "cmd" in the start menu.

2. **Navigate to your project directory**

   ```cmd
   cd "c:\Users\bcada\CascadeProjects\NBA ALGORITHM"
   ```

3. **Start the API server**

   ```cmd
   python -m src.api.server
   ```

4. **With custom host and port**

   ```cmd
   set API_HOST=127.0.0.1
   set API_PORT=8080
   python -m src.api.server
   ```

5. **With API token for authentication**

   ```cmd
   set API_TOKEN=your_secure_token_here
   python -m src.api.server
   ```

## Accessing the API and Dashboard

Once the server is running, you can access:

- **API Endpoints**: http://localhost:5000/api/*
- **Dashboard**: http://localhost:5000/dashboard

If you specified a different port, replace 5000 with your port number.

## API Endpoints

The API server provides the following endpoints:

### Health Check

- `GET /api/health` - Verify the API is operational

### Training Control

- `GET /api/training/status` - Check current training status
- `POST /api/training/start` - Start model training (requires authentication)
- `POST /api/training/cancel` - Cancel ongoing training (requires authentication)

### Model Management

- `GET /api/models/list` - List all available models
- `GET /api/models/{model_name}/details` - Get detailed information about a specific model
- `POST /api/models/{model_name}/retrain` - Trigger retraining for a specific model (requires authentication)

### Performance Monitoring

- `GET /api/models/performance` - Get performance metrics for all models
- `GET /api/models/drift` - Check for model drift and see which models need retraining

### Predictions

- `GET /api/predictions/today` - Get predictions for today's games

## Using the Dashboard

The dashboard provides a visual interface for monitoring your models and controlling the training process:

1. Open your web browser and navigate to http://localhost:5000/dashboard
2. View real-time performance metrics and charts
3. Check model status and drift detection
4. To trigger model training, enter your API token and click the training button

## Testing the API

To run the API tests:

```powershell
python -m tests.test_api --host localhost --port 5000
```

To test dashboard integration:

```powershell
python -m tests.test_dashboard_integration --server --generate-data
```

## Troubleshooting

### Server Won't Start

1. **Check for port conflicts**

   If port 5000 is already in use, specify a different port using the `API_PORT` environment variable.

2. **Check for missing dependencies**

   Install required packages:
   ```powershell
   pip install flask flask-cors psycopg2-binary
   ```

3. **Check database connectivity**

   Ensure your PostgreSQL database is running and the connection parameters are correct.

### Authentication Issues

1. **Verify API token**

   Make sure you're using the correct API token for authenticated endpoints.

2. **Check Bearer format**

   When calling authenticated endpoints from external tools, use the format:
   ```
   Authorization: Bearer your_token_here
   ```

## Running in Production

For production deployments, consider:

1. Using a WSGI server like Gunicorn
2. Setting up a reverse proxy with Nginx
3. Using HTTPS for secure communication
4. Implementing proper logging and monitoring

## Stopping the Server

To stop the API server, press `Ctrl+C` in the terminal where it's running.
